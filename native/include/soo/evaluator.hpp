// Evaluator interface plus the Gate B reference evaluator.
//
// The reference evaluator is a pure function of the request and deliberately
// NOT a constant: a constant makes every edge tie and hides selection bugs
// (docs/native_selfplay_phase0.md section 7, Gate B).  It is specified in
// integer arithmetic so a Python implementation can reproduce it bit-for-bit;
// tests/native/reference_evaluator.py is that implementation.
#pragma once

#include <cstdint>
#include <vector>

#include "soo/encoder.hpp"

namespace soo {

struct EvalOutcome {
    std::vector<double> priors;  // aligned with the request's legal actions
    double value = 0.0;
};

class Evaluator {
  public:
    virtual ~Evaluator() = default;
    virtual EvalOutcome evaluate(const Encoded& encoded,
                                 const std::vector<int32_t>& legal_actions) = 0;
};

// FNV-1a over the request's canonical byte serialization.  This is the value
// the Gate B request-sequence comparison is keyed on.
uint64_t request_hash(const Encoded& encoded, const std::vector<int32_t>& legal_actions);

class DeterministicEvaluator : public Evaluator {
  public:
    EvalOutcome evaluate(const Encoded& encoded,
                         const std::vector<int32_t>& legal_actions) override;
};

// Same hash-derived value, but every prior identical.
//
// This is the evaluator that exercises the PUCT tie-break.  With distinct
// priors no two selection keys are ever exactly equal, so the ``(-score,
// action)`` ordering is never consulted -- yet the production vacancy prior
// assigns equal probability to every action sharing an integer progress score,
// which ties the keys exactly on an unvisited node.  A varying value keeps the
// search from degenerating into a pure tie-break walk.
class UniformPriorEvaluator : public Evaluator {
  public:
    EvalOutcome evaluate(const Encoded& encoded,
                         const std::vector<int32_t>& legal_actions) override;
};

}  // namespace soo
