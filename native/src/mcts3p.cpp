#include "soo/mcts3p.hpp"

#include <cmath>
#include <stdexcept>

#include "soo/encoder.hpp"
#include "soo/mcts.hpp"
#include "soo/rules.hpp"

namespace soo {
namespace {

// puct.exploration_bonus, in Python's operation order so the doubles agree bit
// for bit -- the same function the 2P search uses, repeated here rather than
// shared because both are transcriptions of one Python expression and a shared
// helper would hide which one drifted.
inline double exploration_bonus(double prior, uint32_t parent_visits, uint32_t edge_visits,
                                double c_puct) {
    const double parent = static_cast<double>(parent_visits < 1 ? 1u : parent_visits);
    return c_puct * prior * std::sqrt(parent) / (1.0 + static_cast<double>(edge_visits));
}

}  // namespace

SearchSession3P::SearchSession3P(const Match& match, const MCTSConfig& config)
    : match_(match), config_(config), rng_(config.seed) {
    if (config.simulations <= 0) throw std::invalid_argument("simulations must be positive");
    if (match.count != 3) throw std::invalid_argument("MCTS3P requires a three-seat match");
    player_ids_.reserve(match.count);
    for (uint8_t seat = 0; seat < match.count; ++seat) {
        player_ids_.push_back(match.players[seat].id);
    }
}

int SearchSession3P::seat_of(const State& state) const {
    return match_.seat_of(search_current_player(state, match_));
}

void SearchSession3P::begin(const State& state, double temperature) {
    if (state.status == kFinished) throw std::invalid_argument("cannot search a terminal state");
    if (config_.dirichlet_epsilon > 0.0 &&
        (config_.dirichlet_alpha <= 0.0 || config_.dirichlet_epsilon > 1.0)) {
        throw std::invalid_argument("Dirichlet alpha must be positive and epsilon in [0, 1]");
    }
    if (temperature < 0.0) throw std::invalid_argument("temperature must not be negative");

    temperature_ = temperature;
    result_ = SearchResult3P{};
    nodes_.clear();
    edges_.clear();
    nodes_.reserve(static_cast<size_t>(config_.simulations) + 2);
    edges_.reserve((static_cast<size_t>(config_.simulations) + 2) * 64);
    simulation_ = 0;
    path_.clear();

    VectorNode root;
    root.state = state;
    root.seat = seat_of(state);
    nodes_.push_back(root);
    root_ = 0;
    node_ = root_;
    phase_ = Phase::Root;
}

void SearchSession3P::prepare_expansion(uint32_t node_index) {
    pending_state_ = nodes_[node_index].state;
    pending_encoded_ = soo::encode(pending_state_, match_);
    pending_actions_.clear();
    canonical_legal_action_ids(pending_state_, match_, pending_actions_);
}

ValueVector SearchSession3P::complete_expansion(uint32_t node_index) {
    ++result_.evaluator_calls;
    if (pending_outcome_.priors.size() != pending_actions_.size()) {
        throw std::runtime_error("evaluator priors must match authoritative legal actions");
    }
    if (pending_encoded_.canonical_player_ids.size() != static_cast<size_t>(match_.count)) {
        throw std::runtime_error("3P evaluator request must identify three canonical players");
    }

    // The evaluator answers in the *node's* canonical order; the tree stores
    // values by seat. Getting this wrong is invisible in a 2P search (the
    // rotation is a sign flip) and silently trains the wrong seat in a 3P one.
    ValueVector value{};
    for (size_t index = 0; index < pending_encoded_.canonical_player_ids.size(); ++index) {
        const int seat = match_.seat_of(pending_encoded_.canonical_player_ids[index]);
        if (seat < 0) throw std::runtime_error("canonical player is not a seat in this match");
        value[static_cast<size_t>(seat)] = pending_outcome_.value[index];
    }

    VectorNode& node = nodes_[node_index];
    node.edge_begin = static_cast<uint32_t>(edges_.size());
    for (size_t i = 0; i < pending_actions_.size(); ++i) {
        VectorEdge edge;
        edge.action = pending_actions_[i];
        edge.prior = pending_outcome_.priors[i];
        edges_.push_back(edge);
    }
    node.edge_count = static_cast<uint16_t>(pending_actions_.size());
    node.expanded = true;
    return value;
}

// DiamondSearchAdapter.terminal_vector_value: placement utility, 1 / 0 / -1 in
// finishing order.
ValueVector SearchSession3P::terminal_value(const State& state) const {
    if (state.status != kFinished || state.finished_count != match_.count) {
        throw std::runtime_error("final order is only available for a completed match");
    }
    static constexpr double kPlacement[3] = {1.0, 0.0, -1.0};
    ValueVector value{};
    for (uint8_t place = 0; place < state.finished_count; ++place) {
        const int seat = match_.seat_of(state.finish_order[place]);
        if (seat < 0) throw std::runtime_error("finish order names a seat not in this match");
        value[static_cast<size_t>(seat)] = kPlacement[place];
    }
    return value;
}

void SearchSession3P::apply_dirichlet_noise(uint32_t node_index) {
    if (config_.dirichlet_epsilon <= 0.0) return;
    VectorNode& node = nodes_[node_index];
    if (node.edge_count == 0) return;

    weights_.clear();
    weights_.reserve(node.edge_count);
    double total = 0.0;
    for (uint16_t offset = 0; offset < node.edge_count; ++offset) {
        const double sample = rng_.gammavariate(config_.dirichlet_alpha);
        weights_.push_back(sample);
        total += sample;
    }
    if (!(total > 0.0)) throw std::runtime_error("failed to sample root Dirichlet noise");

    const double epsilon = config_.dirichlet_epsilon;
    for (uint16_t offset = 0; offset < node.edge_count; ++offset) {
        VectorEdge& edge = edges_[node.edge_begin + offset];
        edge.prior = (1.0 - epsilon) * edge.prior + epsilon * (weights_[offset] / total);
    }
}

// The same vector is added to every ancestor, unchanged. No negation, no
// rotation: it is indexed by seat, and a 3P game is not zero-sum between two
// sides.
void SearchSession3P::backup(const ValueVector& value) {
    for (size_t i = path_.size(); i-- > 0;) {
        VectorEdge& edge = edges_[path_[i].second];
        ++edge.visits;
        for (uint8_t seat = 0; seat < match_.count; ++seat) {
            edge.value_sum[seat] += value[seat];
        }
        ++nodes_[path_[i].first].total_visits;
    }
}

uint32_t SearchSession3P::select(uint32_t node_index) const {
    const VectorNode& node = nodes_[node_index];
    const uint32_t parent_visits = node.total_visits;

    // Python's key is ``(-(q + u), action)``: maximise, break ties on the
    // smallest action id. The q read is this node's own seat component.
    uint32_t best = node.edge_begin;
    double best_score = 0.0;
    int32_t best_action = 0;
    for (uint16_t offset = 0; offset < node.edge_count; ++offset) {
        const uint32_t index = node.edge_begin + offset;
        const VectorEdge& edge = edges_[index];
        const double score = edge.q(node.seat) + exploration_bonus(edge.prior, parent_visits,
                                                                   edge.visits, config_.c_puct);
        if (offset == 0 || score > best_score ||
            (score == best_score && edge.action < best_action)) {
            best = index;
            best_score = score;
            best_action = edge.action;
        }
    }
    return best;
}

void SearchSession3P::finalize() {
    const VectorNode& root_node = nodes_[root_];
    uint32_t total = 0;
    for (uint16_t offset = 0; offset < root_node.edge_count; ++offset) {
        const VectorEdge& edge = edges_[root_node.edge_begin + offset];
        result_.root_actions.push_back(edge.action);
        result_.visit_counts.push_back(edge.visits);
        result_.root_priors.push_back(edge.prior);
        ValueVector q{};
        for (uint8_t seat = 0; seat < match_.count; ++seat) q[seat] = edge.q(seat);
        result_.q_vectors.push_back(q);
        total += edge.visits;
    }
    for (const uint32_t visits : result_.visit_counts) {
        result_.policy.push_back(static_cast<double>(visits) / static_cast<double>(total));
    }

    // puct.select_from_visits, identical to the 2P path.
    if (temperature_ > 0.0) {
        weights_.clear();
        weights_.reserve(result_.visit_counts.size());
        double weight_total = 0.0;
        for (const uint32_t visits : result_.visit_counts) {
            const double weight = std::pow(static_cast<double>(visits), 1.0 / temperature_);
            weights_.push_back(weight);
            weight_total += weight;
        }
        if (!(weight_total > 0.0)) {
            for (double& weight : weights_) weight = 1.0;
        }
        result_.selected_action = result_.root_actions[rng_.weighted_index(weights_)];
    } else {
        size_t best = 0;
        for (size_t i = 1; i < result_.root_actions.size(); ++i) {
            if (result_.visit_counts[i] > result_.visit_counts[best] ||
                (result_.visit_counts[i] == result_.visit_counts[best] &&
                 result_.root_actions[i] < result_.root_actions[best])) {
                best = i;
            }
        }
        result_.selected_action = result_.root_actions[best];
    }
    result_.nodes_created = static_cast<uint32_t>(nodes_.size());
    result_.simulations_run = static_cast<uint32_t>(simulation_);
    phase_ = Phase::Done;
}

SearchSession3P::Status SearchSession3P::advance() {
    for (;;) {
        switch (phase_) {
            case Phase::Root:
                prepare_expansion(root_);
                phase_ = Phase::AwaitRoot;
                return Status::NeedsEvaluation;

            case Phase::AwaitRoot: {
                (void)complete_expansion(root_);
                apply_dirichlet_noise(root_);
                phase_ = Phase::Descend;
                break;
            }

            case Phase::Descend: {
                if (simulation_ >= config_.simulations) {
                    finalize();
                    return Status::Ready;
                }
                ++simulation_;
                path_.clear();
                node_ = root_;
                for (;;) {
                    const VectorNode& node = nodes_[node_];
                    if (!node.expanded || node.state.status == kFinished) break;
                    const uint32_t edge_index = select(node_);
                    path_.emplace_back(node_, edge_index);
                    if (edges_[edge_index].child < 0) {
                        const State& parent_state = nodes_[node_].state;
                        const State child_state = soo::apply_action(
                            parent_state, match_,
                            to_physical_action(edges_[edge_index].action, match_,
                                               parent_state.current_player));
                        VectorNode child;
                        child.state = child_state;
                        child.seat = seat_of(child_state);
                        child.terminal = child_state.status == kFinished;
                        nodes_.push_back(child);
                        edges_[edge_index].child = static_cast<int32_t>(nodes_.size() - 1);
                    }
                    node_ = static_cast<uint32_t>(edges_[edge_index].child);
                    if (!nodes_[node_].expanded) break;
                }

                if (nodes_[node_].state.status == kFinished) {
                    backup(terminal_value(nodes_[node_].state));
                    break;
                }
                prepare_expansion(node_);
                phase_ = Phase::AwaitLeaf;
                return Status::NeedsEvaluation;
            }

            case Phase::AwaitLeaf: {
                backup(complete_expansion(node_));
                phase_ = Phase::Descend;
                break;
            }

            case Phase::Done:
                return Status::Ready;
        }
    }
}

}  // namespace soo
