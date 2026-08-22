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

SearchSession::SearchSession(const Match& match, const MCTSConfig& config)
    : match_(match), config_(config) {
    if (config.simulations <= 0) throw std::invalid_argument("simulations must be positive");
    if (match.count != 2) throw std::invalid_argument("MCTS2P requires a two-seat match");
}

void SearchSession::begin(const State& state, double temperature, bool trace) {
    if (state.status == kFinished) throw std::invalid_argument("cannot search a terminal state");
    if (config_.dirichlet_epsilon > 0.0 || temperature > 0.0) {
        throw std::invalid_argument(
            "native MCTS is deterministic-only for now: "
            "dirichlet_epsilon and temperature must both be 0");
    }
    arena_.reset(static_cast<size_t>(config_.simulations), 64);
    result_ = SearchResult{};
    trace_ = trace;
    simulation_ = 0;
    root_state_ = state;
    path_.clear();
    path_.reserve(64);
    root_ = arena_.add_node(state, search_current_player(state, match_), false);
    node_ = root_;
    phase_ = Phase::Root;
}

void SearchSession::prepare_expansion(uint32_t node_index) {
    pending_state_ = arena_.node(node_index).state;
    pending_encoded_ = soo::encode(pending_state_, match_);
    pending_actions_.clear();
    pending_actions_.reserve(64);
    canonical_legal_action_ids(pending_state_, match_, pending_actions_);
}

double SearchSession::complete_expansion(uint32_t node_index) {
    ++result_.evaluator_calls;
    if (pending_outcome_.priors.size() != pending_actions_.size()) {
        throw std::runtime_error("evaluator priors must match authoritative legal actions");
    }
    if (trace_) {
        result_.trace.push_back(
            EvalRecord{request_hash(pending_encoded_, pending_actions_), pending_actions_});
    }
    arena_.open_edges(node_index);
    for (size_t i = 0; i < pending_actions_.size(); ++i) {
        arena_.push_edge(node_index, pending_actions_[i], pending_outcome_.priors[i]);
    }
    arena_.node(node_index).expanded = true;
    return pending_outcome_.value;
}

void SearchSession::backup(double value) {
    // The leaf value is from the leaf player-to-act's perspective; each 2P edge
    // crosses exactly one turn, so the sign flips once per edge.
    for (size_t i = path_.size(); i-- > 0;) {
        value = -value;
        Edge& edge = arena_.edge(path_[i].second);
        ++edge.visits;
        edge.value_sum += value;
        // Python re-sums the children on every selection; the cached aggregate
        // is an integer sum, so keeping it here is exact.
        ++arena_.node(path_[i].first).total_visits;
    }
}

uint32_t SearchSession::select(uint32_t node_index) const {
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
        if (offset == 0 || score > best_score ||
            (score == best_score && edge.action < best_action)) {
            best = index;
            best_score = score;
            best_action = edge.action;
        }
    }
    return best;
}

void SearchSession::finalize() {
    const Node& root_node = arena_.node(root_);
    uint32_t total = 0;
    for (uint16_t offset = 0; offset < root_node.edge_count; ++offset) {
        const Edge& edge = arena_.edge(root_node.edge_begin + offset);
        result_.root_actions.push_back(edge.action);
        result_.visit_counts.push_back(edge.visits);
        result_.q_values.push_back(edge.q());
        total += edge.visits;
    }
    for (const uint32_t visits : result_.visit_counts) {
        result_.policy.push_back(static_cast<double>(visits) / static_cast<double>(total));
    }

    // select_from_visits with temperature <= 0: most visits, ties to the
    // smallest action id.
    size_t best = 0;
    for (size_t i = 1; i < result_.root_actions.size(); ++i) {
        if (result_.visit_counts[i] > result_.visit_counts[best] ||
            (result_.visit_counts[i] == result_.visit_counts[best] &&
             result_.root_actions[i] < result_.root_actions[best])) {
            best = i;
        }
    }
    result_.selected_action = result_.root_actions[best];
    result_.nodes_created = static_cast<uint32_t>(arena_.node_count());
}

SearchSession::Status SearchSession::advance() {
    for (;;) {
        switch (phase_) {
            case Phase::Root:
                // Root expansion with epsilon = 0: add_dirichlet_noise returns
                // the priors unchanged and draws nothing, so there is no RNG to
                // reproduce here.
                prepare_expansion(root_);
                phase_ = Phase::AwaitRoot;
                return Status::NeedsEvaluation;

            case Phase::AwaitRoot:
                complete_expansion(root_);
                simulation_ = 0;
                phase_ = Phase::Descend;
                break;

            case Phase::Descend: {
                if (simulation_ >= config_.simulations) {
                    finalize();
                    phase_ = Phase::Done;
                    break;
                }
                node_ = root_;
                path_.clear();
                while (arena_.node(node_).expanded && !arena_.node(node_).terminal) {
                    const uint32_t edge_index = select(node_);
                    path_.emplace_back(node_, edge_index);
                    if (arena_.edge(edge_index).child == kNoChild) {
                        const State child_state = apply_action(
                            arena_.node(node_).state, match_,
                            to_physical_action(arena_.edge(edge_index).action, match_,
                                               arena_.node(node_).state.current_player));
                        const bool child_terminal = child_state.status == kFinished;
                        const uint32_t child = arena_.add_node(
                            child_state, search_current_player(child_state, match_),
                            child_terminal);
                        // add_node may reallocate, so re-read the edge afterwards.
                        arena_.edge(edge_index).child = static_cast<int32_t>(child);
                    }
                    node_ = static_cast<uint32_t>(arena_.edge(edge_index).child);
                    if (!arena_.node(node_).expanded) break;
                }

                if (arena_.node(node_).terminal) {
                    backup(terminal_scalar_value(arena_.node(node_).state,
                                                 arena_.node(node_).player_id, match_));
                    ++simulation_;
                    result_.simulations_run = static_cast<uint32_t>(simulation_);
                    break;
                }
                prepare_expansion(node_);
                phase_ = Phase::AwaitLeaf;
                return Status::NeedsEvaluation;
            }

            case Phase::AwaitLeaf:
                backup(complete_expansion(node_));
                ++simulation_;
                result_.simulations_run = static_cast<uint32_t>(simulation_);
                phase_ = Phase::Descend;
                break;

            case Phase::Done:
                return Status::Ready;
        }
    }
}

MCTS2P::MCTS2P(const Match& match, Evaluator& evaluator, const MCTSConfig& config)
    : evaluator_(evaluator), session_(match, config) {}

SearchResult MCTS2P::run(const State& state, double temperature, bool trace) {
    session_.begin(state, temperature, trace);
    while (session_.advance() == SearchSession::Status::NeedsEvaluation) {
        session_.supply(evaluator_.evaluate(session_.pending_features(), session_.pending_actions()));
    }
    return session_.result();
}

}  // namespace soo
