#pragma once

#include "diamond_model/soo_model.hpp"
#include "soo/evaluator.hpp"

namespace diamond_model {

class SooEvaluator final : public soo::Evaluator {
  public:
    explicit SooEvaluator(SooModel model);

    soo::EvalOutcome evaluate(const soo::Encoded& encoded,
                              const std::vector<int32_t>& legal_actions) override;

  private:
    SooModel model_;
};

}  // namespace diamond_model
