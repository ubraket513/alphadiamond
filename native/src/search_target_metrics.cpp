#include "soo/search_target_metrics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace soo {
namespace {

bool finite(const VisitTargetObservation& row) {
    return std::isfinite(row.entropy) && std::isfinite(row.normalized_entropy) &&
           std::isfinite(row.max_probability) && std::isfinite(row.top3_mass) &&
           std::isfinite(row.effective_actions) && std::isfinite(row.zero_visit_fraction);
}

} // namespace

VisitTargetObservation inspect_visit_target(std::span<const uint32_t> visit_counts) {
    if (visit_counts.empty()) {
        throw std::invalid_argument("visit target has no legal actions");
    }

    uint64_t visits = 0;
    for (const uint32_t count : visit_counts) {
        if (visits > std::numeric_limits<uint64_t>::max() - count) {
            throw std::invalid_argument("visit target count overflows uint64_t");
        }
        visits += count;
    }
    if (visits == 0) {
        throw std::invalid_argument("visit target has no visits");
    }

    VisitTargetObservation result;
    result.legal_actions = static_cast<uint32_t>(visit_counts.size());
    result.visits = visits;

    std::vector<double> probabilities;
    probabilities.reserve(visit_counts.size());
    uint64_t zero_visits = 0;
    for (const uint32_t count : visit_counts) {
        const double probability = static_cast<double>(count) / static_cast<double>(visits);
        probabilities.push_back(probability);
        if (count == 0) {
            ++zero_visits;
            continue;
        }
        result.entropy -= probability * std::log(probability);
    }

    result.normalized_entropy =
        visit_counts.size() > 1 ? result.entropy / std::log(visit_counts.size()) : 0.0;
    result.effective_actions = std::exp(result.entropy);
    result.zero_visit_fraction =
        static_cast<double>(zero_visits) / static_cast<double>(visit_counts.size());
    std::sort(probabilities.begin(), probabilities.end(), std::greater<>());
    result.max_probability = probabilities.front();
    const std::size_t top_count = std::min<std::size_t>(3, probabilities.size());
    for (std::size_t index = 0; index < top_count; ++index) {
        result.top3_mass += probabilities[index];
    }

    if (!finite(result)) {
        throw std::invalid_argument("visit target produced a non-finite metric");
    }
    return result;
}

VisitTargetSummary summarize_visit_targets(std::span<const VisitTargetObservation> rows) {
    VisitTargetSummary result;
    result.rows = rows.size();
    if (rows.empty())
        return result;

    std::vector<double> entropies;
    entropies.reserve(rows.size());
    for (const VisitTargetObservation& row : rows) {
        if (!finite(row)) {
            throw std::invalid_argument("visit target summary contains a non-finite metric");
        }
        result.legal_actions_mean += row.legal_actions;
        result.entropy_mean += row.entropy;
        result.normalized_entropy_mean += row.normalized_entropy;
        result.max_probability_mean += row.max_probability;
        result.top3_mass_mean += row.top3_mass;
        result.effective_actions_mean += row.effective_actions;
        result.zero_visit_fraction_mean += row.zero_visit_fraction;
        entropies.push_back(row.entropy);
    }

    const double count = static_cast<double>(rows.size());
    result.legal_actions_mean /= count;
    result.entropy_mean /= count;
    result.normalized_entropy_mean /= count;
    result.max_probability_mean /= count;
    result.top3_mass_mean /= count;
    result.effective_actions_mean /= count;
    result.zero_visit_fraction_mean /= count;

    std::sort(entropies.begin(), entropies.end());
    const auto quantile = [&entropies](double q) {
        const auto last = static_cast<double>(entropies.size() - 1);
        return entropies[static_cast<std::size_t>(q * last)];
    };
    result.entropy_p50 = quantile(0.50);
    result.entropy_p90 = quantile(0.90);
    return result;
}

} // namespace soo
