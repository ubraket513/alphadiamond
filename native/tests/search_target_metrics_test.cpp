#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "check.hpp"
#include "soo/search_target_metrics.hpp"

namespace {

bool close(double lhs, double rhs, double tolerance) {
    return std::abs(lhs - rhs) <= tolerance;
}

template <typename Callable> bool throws_invalid_argument(Callable&& callable) {
    try {
        callable();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

} // namespace

int main() {
    using soo::inspect_visit_target;
    using soo::summarize_visit_targets;
    using soo::VisitTargetObservation;

    CHECK(throws_invalid_argument([] { (void)inspect_visit_target({}); }));
    CHECK(
        throws_invalid_argument([] { (void)inspect_visit_target(std::array<uint32_t, 2>{0, 0}); }));

    const auto deterministic = inspect_visit_target(std::array<uint32_t, 3>{8, 0, 0});
    CHECK_EQ(deterministic.legal_actions, 3U);
    CHECK_EQ(deterministic.visits, 8U);
    CHECK(close(deterministic.entropy, 0.0, 1e-12));
    CHECK(close(deterministic.normalized_entropy, 0.0, 1e-12));
    CHECK(close(deterministic.max_probability, 1.0, 1e-12));
    CHECK(close(deterministic.top3_mass, 1.0, 1e-12));
    CHECK(close(deterministic.effective_actions, 1.0, 1e-12));
    CHECK(close(deterministic.zero_visit_fraction, 2.0 / 3.0, 1e-12));

    const auto uniform = inspect_visit_target(std::array<uint32_t, 4>{1, 1, 1, 1});
    CHECK(close(uniform.entropy, std::log(4.0), 1e-12));
    CHECK(close(uniform.normalized_entropy, 1.0, 1e-12));
    CHECK(close(uniform.max_probability, 0.25, 1e-12));
    CHECK(close(uniform.top3_mass, 0.75, 1e-12));
    CHECK(close(uniform.effective_actions, 4.0, 1e-12));

    const std::array<VisitTargetObservation, 5> rows{{
        {.legal_actions = 1, .entropy = 5.0},
        {.legal_actions = 2, .entropy = 1.0},
        {.legal_actions = 3, .entropy = 4.0},
        {.legal_actions = 4, .entropy = 2.0},
        {.legal_actions = 5, .entropy = 3.0},
    }};
    const auto summary = summarize_visit_targets(rows);
    CHECK_EQ(summary.rows, 5U);
    CHECK(close(summary.legal_actions_mean, 3.0, 1e-12));
    CHECK(close(summary.entropy_mean, 3.0, 1e-12));
    CHECK(close(summary.entropy_p50, 3.0, 1e-12));
    CHECK(close(summary.entropy_p90, 4.0, 1e-12));

    auto non_finite = rows;
    non_finite[2].effective_actions = std::numeric_limits<double>::infinity();
    CHECK(throws_invalid_argument([&] { (void)summarize_visit_targets(non_finite); }));

    return soo_test::report("search_target_metrics_test");
}
