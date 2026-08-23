#include "search_telemetry.hpp"

#include <algorithm>

SearchComputeMetrics compute_search_metrics(const SearchTelemetry& telemetry) {
    SearchComputeMetrics result;
    result.total_ms = std::max(0.0, telemetry.total_ms);
    result.neural_ms = std::clamp(telemetry.neural_ms, 0.0, result.total_ms);
    result.mcts_rules_ms = std::max(0.0, result.total_ms - result.neural_ms);
    if (result.total_ms > 0.0) {
        result.neural_fraction = result.neural_ms / result.total_ms;
        result.mcts_rules_fraction = result.mcts_rules_ms / result.total_ms;
        const double seconds = result.total_ms / 1000.0;
        result.simulations_per_second = static_cast<double>(telemetry.simulations) / seconds;
        result.evaluations_per_second = static_cast<double>(telemetry.evaluator_calls) / seconds;
    }
    if (telemetry.evaluator_calls > 0) {
        result.average_neural_evaluation_ms =
            result.neural_ms / static_cast<double>(telemetry.evaluator_calls);
    }
    return result;
}

std::optional<ActionTelemetry> action_telemetry_for(
    const SearchTelemetry& telemetry, int32_t action) {
    const auto found = std::find_if(
        telemetry.actions.cbegin(), telemetry.actions.cend(),
        [action](const ActionTelemetry& row) { return row.action == action; });
    return found == telemetry.actions.cend() ? std::nullopt
                                             : std::optional<ActionTelemetry>(*found);
}

double normalize_soo_value(double value, int player_to_act, int perspective_player_id) {
    return player_to_act == perspective_player_id ? value : -value;
}

double soo_estimate(double fixed_perspective_value) {
    return (fixed_perspective_value + 1.0) / 2.0;
}

