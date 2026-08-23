#pragma once

#include <QMetaType>

#include <cstdint>
#include <optional>
#include <vector>

struct ActionTelemetry {
    int32_t action = -1;
    double prior = 0.0;
    double q = 0.0;
    uint32_t visits = 0;
    double visit_fraction = 0.0;
};

struct SearchTelemetry {
    // Root values retain the canonical root player-to-act convention here.
    // NativeController normalizes them to its fixed UI perspective on commit.
    double root_network_value = 0.0;
    double root_search_value = 0.0;
    double total_ms = 0.0;
    double neural_ms = 0.0;
    uint32_t simulations = 0;
    uint32_t evaluator_calls = 0;
    uint32_t nodes_created = 0;
    std::vector<ActionTelemetry> actions;
};

struct SearchComputeMetrics {
    double total_ms = 0.0;
    double neural_ms = 0.0;
    double mcts_rules_ms = 0.0;
    double neural_fraction = 0.0;
    double mcts_rules_fraction = 0.0;
    double simulations_per_second = 0.0;
    double evaluations_per_second = 0.0;
    double average_neural_evaluation_ms = 0.0;
};

struct AiSearchResult {
    int32_t selected_action = -1;
    SearchTelemetry telemetry;
};

SearchComputeMetrics compute_search_metrics(const SearchTelemetry& telemetry);
std::optional<ActionTelemetry> action_telemetry_for(
    const SearchTelemetry& telemetry, int32_t action);

// Convert a canonical player-to-act Soo value to a fixed two-player seat.
double normalize_soo_value(double value, int player_to_act, int perspective_player_id);
double soo_estimate(double fixed_perspective_value);

Q_DECLARE_METATYPE(AiSearchResult)

