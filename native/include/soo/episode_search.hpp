// One game's search, whichever seat count the match has.
//
// The self-play pool is seat-agnostic in everything that matters -- lanes,
// batching, the job queue, the move recording -- and was two-player only for
// one reason: it named `SearchSession` directly. Min needs the same pool with
// `SearchSession3P`, whose value is a vector rather than a scalar.
//
// This is that one difference, and nothing else. It is deliberately not a
// virtual interface: there are exactly two implementations, both known here,
// and a vtable call per node on the search worker is the sort of cost this
// port exists to avoid.
#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

#include "soo/encoder.hpp"
#include "soo/mcts.hpp"
#include "soo/mcts3p.hpp"
#include "soo/state.hpp"

namespace soo {

class EpisodeSearch {
  public:
    enum class Status : uint8_t { NeedsEvaluation, Ready };

    EpisodeSearch(const Match& match, const MCTSConfig& config)
        : seats_(match.count), simulations_(config.simulations) {
        if (match.count == 2) {
            two_.emplace(match, config);
        } else if (match.count == 3) {
            three_.emplace(match, config);
        } else {
            throw std::invalid_argument("a match has two or three seats");
        }
    }

    // How many value components an evaluator must return for this match: one
    // for Soo, one per seat for Min.
    int value_width() const { return two_ ? 1 : 3; }
    int seats() const { return seats_; }

    void reseed(uint64_t seed) {
        if (two_) two_->reseed(seed);
        else three_->reseed(seed);
    }

    void set_simulations(int simulations) {
        if (two_)
            two_->set_simulations(simulations);
        else
            three_->set_simulations(simulations);
        simulations_ = simulations;
    }

    void begin(const State& state, double temperature) {
        if (two_) {
            two_->begin(state, temperature, false);
            root_ = &two_->root_features();
            return;
        }
        three_->begin(state, temperature);
        // The 3P session reports the pending node; the root's encoding is the
        // first one it asks about, captured on the first advance below.
        root_captured_ = false;
    }

    Status advance() {
        if (two_) {
            return two_->advance() == SearchSession::Status::NeedsEvaluation
                       ? Status::NeedsEvaluation
                       : Status::Ready;
        }
        const auto status = three_->advance();
        if (!root_captured_ && status == SearchSession3P::Status::NeedsEvaluation) {
            // A training sample is the position that was searched, so the
            // features recorded must be the root's -- not whichever leaf
            // happened to be expanded last. The 2P session exposes this
            // directly; here the root is simply the first request.
            root_encoded_ = three_->pending_features();
            root_ = &root_encoded_;
            root_captured_ = true;
        }
        return status == SearchSession3P::Status::NeedsEvaluation ? Status::NeedsEvaluation
                                                                  : Status::Ready;
    }

    const Encoded& pending_features() const {
        return two_ ? two_->pending_features() : three_->pending_features();
    }
    const std::vector<int32_t>& pending_actions() const {
        return two_ ? two_->pending_actions() : three_->pending_actions();
    }
    const State& pending_state() const {
        return two_ ? two_->pending_state() : three_->pending_state();
    }
    const Encoded& root_features() const { return *root_; }

    // `values` holds `value_width()` components, in the pending request's
    // canonical player order.
    void supply(const std::vector<double>& priors, const double* values) {
        if (two_) {
            EvalOutcome outcome;
            outcome.priors = priors;
            outcome.value = values[0];
            two_->supply(outcome);
            return;
        }
        EvalOutcome3P outcome;
        outcome.priors = priors;
        for (int seat = 0; seat < 3; ++seat) outcome.value[static_cast<size_t>(seat)] = values[seat];
        three_->supply(outcome);
    }

    int32_t selected_action() const {
        return two_ ? two_->result().selected_action : three_->result().selected_action;
    }
    const std::vector<int32_t>& root_actions() const {
        return two_ ? two_->result().root_actions : three_->result().root_actions;
    }
    const std::vector<uint32_t>& visit_counts() const {
        return two_ ? two_->result().visit_counts : three_->result().visit_counts;
    }
    uint32_t evaluator_calls() const {
        return two_ ? two_->result().evaluator_calls : three_->result().evaluator_calls;
    }

  private:
    int seats_ = 0;
    int simulations_ = 0;
    std::optional<SearchSession> two_;
    std::optional<SearchSession3P> three_;
    const Encoded* root_ = nullptr;
    Encoded root_encoded_;
    bool root_captured_ = false;
};

}  // namespace soo
