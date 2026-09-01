#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "check.hpp"
#include "diamond_pipeline/policy_diagnostics.hpp"
#include "soo/board.hpp"

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
    using diamond_pipeline::diagnose_policy_row;
    std::vector<float> logits(soo::kActionSize, 0.0F);
    const std::array<int32_t, 2> legal{0, 1};
    const std::array<uint32_t, 2> visits{1, 0};
    const auto row = diagnose_policy_row(logits, legal, visits);
    CHECK(close(row.target_entropy, 0.0, 1e-12));
    CHECK(close(row.legal_cross_entropy, std::log(2.0), 1e-6));
    CHECK(close(row.legal_kl, std::log(2.0), 1e-6));
    CHECK(close(row.legal_probability_mass, 2.0 / 5329.0, 1e-7));
    CHECK(close(row.full_cross_entropy - row.legal_cross_entropy,
                -std::log(row.legal_probability_mass), 1e-6));
    CHECK(close(row.full_kl - row.legal_kl, -std::log(row.legal_probability_mass), 1e-6));
    CHECK(row.top1_agrees);

    CHECK(throws_invalid_argument([&] {
        (void)diagnose_policy_row(std::span<const float>(logits).first(4), legal, visits);
    }));
    CHECK(throws_invalid_argument(
        [&] { (void)diagnose_policy_row(logits, legal, std::array<uint32_t, 1>{1}); }));
    CHECK(throws_invalid_argument(
        [&] { (void)diagnose_policy_row(logits, std::array<int32_t, 2>{1, 1}, visits); }));
    CHECK(throws_invalid_argument([&] {
        (void)diagnose_policy_row(logits, std::array<int32_t, 2>{0, soo::kActionSize}, visits);
    }));
    CHECK(throws_invalid_argument(
        [&] { (void)diagnose_policy_row(logits, legal, std::array<uint32_t, 2>{0, 0}); }));
    logits[3] = std::numeric_limits<float>::infinity();
    CHECK(throws_invalid_argument([&] { (void)diagnose_policy_row(logits, legal, visits); }));
    return soo_test::report("policy_diagnostics_test");
}
