#include "diamond_pipeline/policy_diagnostics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "soo/board.hpp"

namespace diamond_pipeline {
namespace {
double checked_kl(double value) {
    if (value < 0.0 && value > -1e-10)
        return 0.0;
    if (value < 0.0 || !std::isfinite(value))
        throw std::invalid_argument("policy diagnostics produced invalid KL divergence");
    return value;
}
} // namespace

PolicyRowDiagnostics diagnose_policy_row(std::span<const float> full_logits,
                                         std::span<const int32_t> legal_actions,
                                         std::span<const uint32_t> visit_counts) {
    if (full_logits.size() != static_cast<std::size_t>(soo::kActionSize))
        throw std::invalid_argument("policy diagnostics require the full action space");
    if (legal_actions.empty() || legal_actions.size() != visit_counts.size())
        throw std::invalid_argument("legal actions and visits must have equal non-zero width");

    double full_max = -std::numeric_limits<double>::infinity();
    for (const float logit : full_logits) {
        if (!std::isfinite(logit))
            throw std::invalid_argument("policy logits must be finite");
        full_max = std::max(full_max, static_cast<double>(logit));
    }
    double full_sum = 0.0;
    for (const float logit : full_logits)
        full_sum += std::exp(logit - full_max);
    const double full_log_z = full_max + std::log(full_sum);

    std::vector<bool> seen(soo::kActionSize, false);
    uint64_t visits = 0;
    double legal_max = -std::numeric_limits<double>::infinity();
    for (std::size_t index = 0; index < legal_actions.size(); ++index) {
        const int32_t action = legal_actions[index];
        if (action < 0 || action >= soo::kActionSize)
            throw std::invalid_argument("legal action is out of range");
        if (seen[action])
            throw std::invalid_argument("legal actions contain a duplicate");
        seen[action] = true;
        visits += visit_counts[index];
        legal_max = std::max(legal_max, static_cast<double>(full_logits[action]));
    }
    if (visits == 0)
        throw std::invalid_argument("policy target has no visits");
    double legal_sum = 0.0;
    for (const int32_t action : legal_actions)
        legal_sum += std::exp(static_cast<double>(full_logits[action]) - legal_max);
    const double legal_log_z = legal_max + std::log(legal_sum);

    PolicyRowDiagnostics result;
    int32_t target_top = legal_actions.front();
    uint32_t target_top_visits = visit_counts.front();
    int32_t network_top = legal_actions.front();
    for (std::size_t index = 0; index < legal_actions.size(); ++index) {
        const int32_t action = legal_actions[index];
        const uint32_t count = visit_counts[index];
        if (count > target_top_visits || (count == target_top_visits && action < target_top)) {
            target_top = action;
            target_top_visits = count;
        }
        if (full_logits[action] > full_logits[network_top] ||
            (full_logits[action] == full_logits[network_top] && action < network_top))
            network_top = action;
        if (count == 0)
            continue;
        const double target = static_cast<double>(count) / static_cast<double>(visits);
        result.target_entropy -= target * std::log(target);
        result.full_cross_entropy -=
            target * (static_cast<double>(full_logits[action]) - full_log_z);
        result.legal_cross_entropy -=
            target * (static_cast<double>(full_logits[action]) - legal_log_z);
    }
    result.full_kl = checked_kl(result.full_cross_entropy - result.target_entropy);
    result.legal_kl = checked_kl(result.legal_cross_entropy - result.target_entropy);
    result.legal_probability_mass = std::exp(legal_log_z - full_log_z);
    result.top1_agrees = target_top == network_top;
    if (!std::isfinite(result.legal_probability_mass) || result.legal_probability_mass <= 0.0 ||
        result.legal_probability_mass > 1.0 + 1e-12)
        throw std::invalid_argument("policy diagnostics produced invalid legal mass");
    return result;
}

} // namespace diamond_pipeline
