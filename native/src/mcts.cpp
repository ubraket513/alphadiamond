#include "soo/mcts.hpp"

#include <cmath>
#include <stdexcept>

#include "soo/encoder.hpp"
#include "soo/rules.hpp"

namespace soo {
namespace {

// puct.exploration_bonus, in Python's operation order so the doubles agree bit
// for bit: ((c_puct * prior) * sqrt(max(1, parent))) / (1 + edge_visits).
inline double exploration_bonus(double prior, uint32_t parent_visits, uint32_t edge_visits,
                                double c_puct) {
    const double parent = static_cast<double>(parent_visits < 1 ? 1u : parent_visits);
    return c_puct * prior * std::sqrt(parent) / (1.0 + static_cast<double>(edge_visits));
}

// DiamondSearchAdapter.terminal_scalar_value
double terminal_scalar_value(const State& state, uint8_t player_id, const Match& match) {
    if (state.status != kFinished || state.finished_count != match.count) {
        throw std::runtime_error("final order is only available for a completed match");
    }
    return player_id == state.finish_order[0] ? 1.0 : -1.0;
}

}  // namespace

MCTS2P::MCTS2P(const Match& match, Evaluator& evaluator, const MCTSConfig& config)
    : match_(match), evaluator_(evaluator), config_(config) {
    if (config.simulations <= 0) throw std::invalid_argument("simulations must be positive");
    if (match.count != 2) throw std::invalid_argument("MCTS2P requires a two-seat match");
}

double MCTS2P::expand(uint32_t node_index, SearchResult& result, bool trace) {
    const State state = arena_.node(node_index).state;
    const Encoded encoded = soo::encode(state, match_);
    std::vector<int32_t> legal;
    legal.reserve(64);
    canonical_legal_action_ids(state, match_, legal);

    const EvalOutcome outcome = evaluator_.evaluate(encoded, legal);
    ++result.evaluator_calls;
    if (outcome.priors.size() != legal.size()) {
        throw std::runtime_error("evaluator priors must match authoritative legal actions");
    }
    if (trace) {
        result.trace.push_back(EvalRecord{request_hash(encoded, legal), legal});
    }

    arena_.open_edges(node_index);
    for (size_t i = 0; i < legal.size(); ++i) {
        arena_.push_edge(node_index, legal[i], outcome.priors[i]);
    }
    arena_.node(node_index).expanded = true;
    return outcome.value;
}

uint32_t MCTS2P::select(uint32_t node_index) const {
    const Node& node = arena_.node(node_index);
    const uint32_t parent_visits = node.total_visits;

    // Python takes ``min`` over the key ``(-(q + u), action)``: maximise the
    // PUCT score, break ties on the smallest action id.  Negation is exact for
    // doubles, so comparing scores directly gives the identical winner.
    uint32_t best = node.edge_begin;
    double best_score = 0.0;
    int32_t best_action = 0;
    for (uint16_t offset = 0; offset < node.edge_count; ++offset) {
        const uint32_t index = node.edge_begin + offset;
        const Edge& edge = arena_.edge(index);
        const double score =
            edge.q() + exploration_bonus(edge.prior, parent_visits, edge.visits, config_.c_puct);
        if (offset == 0 || score > best_score || (score == best_score && edge.action < best_action)) {
            best = index;
            best_score = score;
            best_action = edge.action;
        }
    }
    return best;
}

SearchResult MCTS2P::run(const State& state, double temperature, bool trace) {
    if (state.status == kFinished) throw std::invalid_argument("cannot search a terminal state");
    if (config_.dirichlet_epsilon > 0.0 || temperature > 0.0) {
        throw std::invalid_argument(
            "native MCTS is deterministic-only for now: "
            "dirichlet_epsilon and temperature must both be 0");
    }

    arena_.reset(static_cast<size_t>(config_.simulations), 64);
    SearchResult result;
    const uint32_t root = arena_.add_node(state, search_current_player(state, match_), false);
    // Root expansion with epsilon = 0: add_dirichlet_noise returns the priors
    // unchanged and draws nothing, so there is no RNG to reproduce here.
    expand(root, result, trace);

    // (owning node, edge) pairs: backup needs the node to keep its cached
    // visit aggregate in step with the edge it increments.
    std::vector<std::pair<uint32_t, uint32_t>> path;
    path.reserve(64);

    for (int simulation = 0; simulation < config_.simulations; ++simulation) {
        uint32_t node_index = root;
        path.clear();

        while (arena_.node(node_index).expanded && !arena_.node(node_index).terminal) {
            const uint32_t edge_index = select(node_index);
            path.emplace_back(node_index, edge_index);
            if (arena_.edge(edge_index).child == kNoChild) {
                const State child_state =
                    apply_action(arena_.node(node_index).state, match_,
                                 to_physical_action(arena_.edge(edge_index).action, match_,
                                                    arena_.node(node_index).state.current_player));
                const bool child_terminal = child_state.status == kFinished;
                const uint32_t child = arena_.add_node(
                    child_state, search_current_player(child_state, match_), child_terminal);
                // add_node may reallocate, so re-read the edge afterwards.
                arena_.edge(edge_index).child = static_cast<int32_t>(child);
            }
            node_index = static_cast<uint32_t>(arena_.edge(edge_index).child);
            if (!arena_.node(node_index).expanded) break;
        }

        double value;
        if (arena_.node(node_index).terminal) {
            value = terminal_scalar_value(arena_.node(node_index).state,
                                          arena_.node(node_index).player_id, match_);
        } else {
            value = expand(node_index, result, trace);
        }

        // The leaf value is from the leaf player-to-act's perspective; each 2P
        // edge crosses exactly one turn, so the sign flips once per edge.
        for (size_t i = path.size(); i-- > 0;) {
            value = -value;
            Edge& edge = arena_.edge(path[i].second);
            ++edge.visits;
            edge.value_sum += value;
            // Python re-sums the children on every selection; the cached
            // aggregate is an integer sum, so keeping it here is exact.
            ++arena_.node(path[i].first).total_visits;
        }
        ++result.simulations_run;
    }

    const Node& root_node = arena_.node(root);
    uint32_t total = 0;
    for (uint16_t offset = 0; offset < root_node.edge_count; ++offset) {
        const Edge& edge = arena_.edge(root_node.edge_begin + offset);
        result.root_actions.push_back(edge.action);
        result.visit_counts.push_back(edge.visits);
        result.q_values.push_back(edge.q());
        total += edge.visits;
    }
    for (const uint32_t visits : result.visit_counts) {
        result.policy.push_back(static_cast<double>(visits) / static_cast<double>(total));
    }

    // select_from_visits with temperature <= 0: most visits, ties to the
    // smallest action id.
    size_t best = 0;
    for (size_t i = 1; i < result.root_actions.size(); ++i) {
        if (result.visit_counts[i] > result.visit_counts[best] ||
            (result.visit_counts[i] == result.visit_counts[best] &&
             result.root_actions[i] < result.root_actions[best])) {
            best = i;
        }
    }
    result.selected_action = result.root_actions[best];
    result.nodes_created = static_cast<uint32_t>(arena_.node_count());
    return result;
}

}  // namespace soo
