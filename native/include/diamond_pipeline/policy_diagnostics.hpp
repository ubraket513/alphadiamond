#pragma once

#include <cstdint>
#include <span>

namespace diamond_pipeline {

struct PolicyRowDiagnostics {
    double target_entropy = 0.0;
    double full_cross_entropy = 0.0;
    double legal_cross_entropy = 0.0;
    double full_kl = 0.0;
    double legal_kl = 0.0;
    double legal_probability_mass = 0.0;
    bool top1_agrees = false;
};

PolicyRowDiagnostics diagnose_policy_row(std::span<const float> full_logits,
                                         std::span<const int32_t> legal_actions,
                                         std::span<const uint32_t> visit_counts);

} // namespace diamond_pipeline
