#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

#include "check.hpp"
#include "diamond_model/soo_model.hpp"
#include "diamond_pipeline/model_pool.hpp"
#include "diamond_training/device.hpp"
#include "diamond_training/trainer.hpp"

#ifdef DIAMOND_TEST_WITH_CUDA
#include <torch/cuda.h>
#endif

namespace {

diamond_training::Compatibility soo_compatibility() {
    return diamond_training::Compatibility::soo(
        "1.0.0", {.residual_blocks = 1, .width = 8});
}

diamond_training::Compatibility min_compatibility() {
    return diamond_training::Compatibility::min(
        "1.0.0", {.residual_blocks = 1, .width = 8});
}

bool same_outcome(const soo::EvalOutcome& left, const soo::EvalOutcome& right) {
    return left.priors == right.priors && left.value == right.value &&
           left.values == right.values;
}

void check_distribution(const soo::EvalOutcome& outcome, std::size_t expected_size) {
    CHECK_EQ(outcome.priors.size(), expected_size);
    double sum = 0.0;
    for (const double prior : outcome.priors) {
        CHECK(std::isfinite(prior));
        CHECK(prior >= 0.0);
        sum += prior;
    }
    CHECK(std::abs(sum - 1.0) < 1.0e-5);
}

struct BatchFixture final {
    BatchFixture(std::size_t batch_size, int feature_count, int value_width)
        : encoded(batch_size), actions(batch_size), outcomes(batch_size), items(batch_size) {
        for (std::size_t row = 0; row < batch_size; ++row) {
            encoded[row].feature_count = feature_count;
            encoded[row].node_features.resize(73U * static_cast<std::size_t>(feature_count));
            for (std::size_t feature = 0; feature < encoded[row].node_features.size(); ++feature) {
                encoded[row].node_features[feature] =
                    static_cast<float>((feature % 97U) + row * 3U) * 0.001F;
            }

            const std::size_t legal_count = 1U + (row % 5U);
            for (std::size_t column = 0; column < legal_count; ++column) {
                actions[row].push_back(static_cast<int32_t>(
                    (row * 101U + column * 73U) % (73U * 73U)));
            }

            outcomes[row].priors = {-10.0 - static_cast<double>(row)};
            outcomes[row].value = -20.0 - static_cast<double>(row);
            outcomes[row].values = {
                -30.0 - static_cast<double>(row),
                -40.0 - static_cast<double>(row),
                -50.0 - static_cast<double>(row),
            };
            items[row] = soo::BatchItem{.encoded = &encoded[row],
                                        .actions = &actions[row],
                                        .outcome = &outcomes[row],
                                        .value_width = value_width};
        }
    }

    std::vector<soo::Encoded> encoded;
    std::vector<std::vector<int32_t>> actions;
    std::vector<soo::EvalOutcome> outcomes;
    std::vector<soo::BatchItem> items;
};

void check_outcomes_unchanged(const std::vector<soo::EvalOutcome>& actual,
                              const std::vector<soo::EvalOutcome>& expected) {
    CHECK_EQ(actual.size(), expected.size());
    for (std::size_t row = 0; row < actual.size(); ++row)
        CHECK(same_outcome(actual[row], expected[row]));
}

template <typename Mutator>
void expect_rejected_transactionally(diamond_pipeline::ModelPool& pool,
                                     int feature_count,
                                     int value_width,
                                     Mutator mutate) {
    BatchFixture fixture(2, feature_count, value_width);
    mutate(fixture);
    const auto before = fixture.outcomes;
    const auto stats_before = pool.last_evaluation_stats();
    bool rejected = false;
    try {
        pool.evaluate(fixture.items);
    } catch (const diamond_pipeline::PipelineError&) {
        rejected = true;
    }
    CHECK(rejected);
    check_outcomes_unchanged(fixture.outcomes, before);
    CHECK(pool.last_evaluation_stats() == stats_before);
}

void test_residency_contract(const std::filesystem::path& scratch,
                             const diamond_training::ResolvedDevice& device) {
    const auto resident_compatibility = soo_compatibility();
    diamond_pipeline::ModelPool pool(2, device);
    auto source = diamond_model::DiamondModel(8, 1, 4, 1);
    {
        const auto source_holder_alias = source;
        CHECK(source_holder_alias->parameters().front().data_ptr() ==
              source->parameters().front().data_ptr());
    }
    const auto first = pool.install(resident_compatibility, source);
    const auto second =
        pool.install(resident_compatibility, diamond_model::DiamondModel(8, 1, 4, 1));
    CHECK_EQ(pool.resident_count(), std::size_t{2});
    CHECK(first != second);
    pool.activate(first);
    CHECK(pool.active_key() == first);

    const auto actor = pool.active_model();
    CHECK(first.model_name == resident_compatibility.model_name);
    CHECK(first.model_version == resident_compatibility.model_version);
    CHECK(first.checkpoint_sha256 == diamond_training::canonical_model_digest(actor));
    CHECK(actor->parameters().front().data_ptr() != source->parameters().front().data_ptr());
    CHECK(actor->adjacency.data_ptr() != source->adjacency.data_ptr());
    CHECK(actor->parameters().front().device() == device.torch_device);
    CHECK(actor->adjacency.device() == device.torch_device);
    CHECK(!actor->is_training());
    for (const auto& parameter : actor->parameters()) CHECK(!parameter.requires_grad());

    const auto actor_first_parameter = actor->parameters().front().detach().clone();
    const auto actor_adjacency = actor->adjacency.detach().clone();
    {
        torch::NoGradGuard no_grad;
        source->parameters().front().add_(1.0F);
        source->adjacency.add_(1.0F);
    }
    source = nullptr;
    CHECK(torch::equal(actor->parameters().front(), actor_first_parameter));
    CHECK(torch::equal(actor->adjacency, actor_adjacency));
    CHECK(pool.active_key() == first);
    CHECK(first.checkpoint_sha256 ==
          diamond_training::canonical_model_digest(pool.active_model()));

    auto incompatible = resident_compatibility;
    incompatible.model_version = "2.0.0";
    bool saw_incompatible = false;
    try {
        pool.require_compatible(incompatible);
    } catch (const diamond_pipeline::IncompatibleCheckpointError&) {
        saw_incompatible = true;
    }
    CHECK(saw_incompatible);

    std::stop_source cancelled;
    cancelled.request_stop();
    bool saw_cancel = false;
    try {
        pool.require_ready(cancelled.get_token(),
                           std::chrono::steady_clock::now() + std::chrono::seconds(1));
    } catch (const diamond_pipeline::CancelledError&) {
        saw_cancel = true;
    }
    CHECK(saw_cancel);

    bool saw_deadline = false;
    try {
        pool.require_ready({}, std::chrono::steady_clock::now() - std::chrono::seconds(1));
    } catch (const diamond_pipeline::DeadlineExceededError&) {
        saw_deadline = true;
    }
    CHECK(saw_deadline);

    bool saw_legacy = false;
    try {
        (void)pool.install_checkpoint(resident_compatibility, scratch / "legacy.pt",
                                      diamond_model::DiamondModel(8, 1, 4, 1));
    } catch (const diamond_pipeline::IncompatibleCheckpointError&) {
        saw_legacy = true;
    }
    CHECK(saw_legacy);
}

void test_ragged_soo(const diamond_training::ResolvedDevice& device) {
    diamond_pipeline::ModelPool pool(1, device);
    const auto key = pool.install(soo_compatibility(), diamond_model::DiamondModel(8, 1, 4, 1));
    pool.activate(key);

    BatchFixture fixture(3, 4, 1);
    fixture.encoded[1] = fixture.encoded[0];
    fixture.actions[0] = {0, 74, 5328};
    fixture.actions[1] = {5328, 74, 0};
    fixture.actions[2] = {7};
    pool.evaluate(fixture.items);

    for (std::size_t row = 0; row < fixture.items.size(); ++row) {
        check_distribution(fixture.outcomes[row], fixture.actions[row].size());
        CHECK(std::isfinite(fixture.outcomes[row].value));
        CHECK(fixture.outcomes[row].value == fixture.outcomes[row].values[0]);
    }
    CHECK(std::abs(fixture.outcomes[0].priors[0] - fixture.outcomes[1].priors[2]) < 1.0e-6);
    CHECK(std::abs(fixture.outcomes[0].priors[1] - fixture.outcomes[1].priors[1]) < 1.0e-6);
    CHECK(std::abs(fixture.outcomes[0].priors[2] - fixture.outcomes[1].priors[0]) < 1.0e-6);

    const auto stats = pool.last_evaluation_stats();
    CHECK_EQ(stats.forward_calls, std::size_t{1});
    CHECK_EQ(stats.h2d_transfers, std::size_t{0});
    CHECK_EQ(stats.d2h_transfers, std::size_t{0});
    CHECK_EQ(stats.batch_size, std::size_t{3});
    CHECK_EQ(stats.max_legal_actions, std::size_t{3});
}

void test_ragged_min(const diamond_training::ResolvedDevice& device) {
    diamond_pipeline::ModelPool pool(1, device);
    const auto key = pool.install(min_compatibility(), diamond_model::DiamondModel(8, 1, 6, 3));
    pool.activate(key);

    BatchFixture fixture(4, 6, 3);
    fixture.actions[0] = {0};
    fixture.actions[1] = {74, 1};
    fixture.actions[2] = {5328, 200, 7, 81};
    fixture.actions[3] = {9, 17, 73};
    pool.evaluate(fixture.items);

    for (std::size_t row = 0; row < fixture.items.size(); ++row) {
        check_distribution(fixture.outcomes[row], fixture.actions[row].size());
        CHECK(fixture.outcomes[row].value == fixture.outcomes[row].values[0]);
        for (const double value : fixture.outcomes[row].values) CHECK(std::isfinite(value));
    }
    const auto stats = pool.last_evaluation_stats();
    CHECK_EQ(stats.forward_calls, std::size_t{1});
    CHECK_EQ(stats.h2d_transfers, std::size_t{0});
    CHECK_EQ(stats.d2h_transfers, std::size_t{0});
    CHECK_EQ(stats.batch_size, std::size_t{4});
    CHECK_EQ(stats.max_legal_actions, std::size_t{4});
}

void test_transactional_rejections(const diamond_training::ResolvedDevice& device) {
    diamond_pipeline::ModelPool pool(1, device);
    const auto key = pool.install(soo_compatibility(), diamond_model::DiamondModel(8, 1, 4, 1));
    pool.activate(key);

    BatchFixture successful(1, 4, 1);
    pool.evaluate(successful.items);

    expect_rejected_transactionally(pool, 4, 1,
        [](BatchFixture& fixture) { fixture.items[1].encoded = nullptr; });
    expect_rejected_transactionally(pool, 4, 1,
        [](BatchFixture& fixture) { fixture.items[1].actions = nullptr; });
    expect_rejected_transactionally(pool, 4, 1,
        [](BatchFixture& fixture) { fixture.items[1].outcome = nullptr; });
    expect_rejected_transactionally(pool, 4, 1,
        [](BatchFixture& fixture) { fixture.encoded[1].feature_count = 3; });
    expect_rejected_transactionally(pool, 4, 1,
        [](BatchFixture& fixture) { fixture.encoded[1].node_features.pop_back(); });
    expect_rejected_transactionally(pool, 4, 1,
        [](BatchFixture& fixture) { fixture.actions[1].clear(); });
    expect_rejected_transactionally(pool, 4, 1,
        [](BatchFixture& fixture) { fixture.actions[1] = {1, 1}; });
    expect_rejected_transactionally(pool, 4, 1,
        [](BatchFixture& fixture) { fixture.actions[1] = {-1}; });
    expect_rejected_transactionally(pool, 4, 1,
        [](BatchFixture& fixture) { fixture.actions[1] = {5329}; });
    expect_rejected_transactionally(pool, 4, 1,
        [](BatchFixture& fixture) {
            fixture.encoded[1].node_features[0] = std::numeric_limits<float>::quiet_NaN();
        });
    expect_rejected_transactionally(pool, 4, 1,
        [](BatchFixture& fixture) {
            fixture.encoded[1].node_features[0] = std::numeric_limits<float>::infinity();
        });
    expect_rejected_transactionally(pool, 4, 1,
        [](BatchFixture& fixture) { fixture.items[1].value_width = 3; });
}

#ifdef DIAMOND_TEST_WITH_CUDA
void test_cuda_batches(const diamond_training::ResolvedDevice& device) {
    REQUIRE(torch::cuda::is_available(),
            "CUDA-labelled inference test requires an available CUDA device");
    diamond_pipeline::ModelPool pool(1, device);
    const auto key = pool.install(soo_compatibility(), diamond_model::DiamondModel(8, 1, 4, 1));
    pool.activate(key);

    for (const std::size_t batch_size : std::array<std::size_t, 4>{1, 17, 64, 256}) {
        BatchFixture fixture(batch_size, 4, 1);
        std::size_t expected_max_legal = 0;
        for (std::size_t row = 0; row < batch_size; ++row) {
            fixture.actions[row].clear();
            const std::size_t legal_count = 1U + (row % 9U);
            expected_max_legal = std::max(expected_max_legal, legal_count);
            for (std::size_t column = 0; column < legal_count; ++column) {
                fixture.actions[row].push_back(static_cast<int32_t>(
                    (row * 101U + column * 73U) % (73U * 73U)));
            }
        }

        pool.evaluate(fixture.items);
        const auto stats = pool.last_evaluation_stats();
        CHECK_EQ(stats.forward_calls, std::size_t{1});
        CHECK_EQ(stats.h2d_transfers, std::size_t{3});
        CHECK_EQ(stats.d2h_transfers, std::size_t{1});
        CHECK_EQ(stats.batch_size, batch_size);
        CHECK_EQ(stats.max_legal_actions, expected_max_legal);
        for (std::size_t row = 0; row < batch_size; ++row) {
            check_distribution(fixture.outcomes[row], fixture.actions[row].size());
            CHECK(std::isfinite(fixture.outcomes[row].value));
        }
    }
}
#endif

}  // namespace

int main(int argc, char** argv) {
    const bool run_cuda = argc == 3 && std::string_view(argv[2]) == "--cuda";
    REQUIRE(argc == 2 || run_cuda,
            "usage: inference_coordinator_test <scratch> [--cuda]");
    const auto scratch = std::filesystem::path(argv[1]);
    std::filesystem::remove_all(scratch);

    if (run_cuda) {
#ifdef DIAMOND_TEST_WITH_CUDA
        test_cuda_batches(diamond_training::resolve_device("cuda"));
        return soo_test::report("inference_coordinator_test_cuda");
#else
        REQUIRE(false, "CUDA inference coverage was not compiled for this build");
#endif
    }

    const auto cpu = diamond_training::resolve_device("cpu");
    test_residency_contract(scratch, cpu);
    test_ragged_soo(cpu);
    test_ragged_min(cpu);
    test_transactional_rejections(cpu);
    return soo_test::report("inference_coordinator_test");
}
