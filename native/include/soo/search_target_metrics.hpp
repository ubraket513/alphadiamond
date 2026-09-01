#pragma once

#include <cstdint>
#include <span>

namespace soo {

struct VisitTargetObservation {
    uint32_t legal_actions = 0;
    uint64_t visits = 0;
    double entropy = 0.0;
    double normalized_entropy = 0.0;
    double max_probability = 0.0;
    double top3_mass = 0.0;
    double effective_actions = 0.0;
    double zero_visit_fraction = 0.0;
};

struct VisitTargetSummary {
    uint64_t rows = 0;
    double legal_actions_mean = 0.0;
    double entropy_mean = 0.0;
    double entropy_p50 = 0.0;
    double entropy_p90 = 0.0;
    double normalized_entropy_mean = 0.0;
    double max_probability_mean = 0.0;
    double top3_mass_mean = 0.0;
    double effective_actions_mean = 0.0;
    double zero_visit_fraction_mean = 0.0;
};

VisitTargetObservation inspect_visit_target(std::span<const uint32_t> visit_counts);
VisitTargetSummary summarize_visit_targets(std::span<const VisitTargetObservation> rows);

} // namespace soo
