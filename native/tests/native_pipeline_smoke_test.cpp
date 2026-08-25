#include <filesystem>
#include <stop_token>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.hpp"
#include "diamond_model/soo_model.hpp"
#include "diamond_pipeline/pipeline.hpp"
#include "diamond_training/trainer.hpp"

namespace {

diamond_training::TrainingSample training_sample(
    const diamond_training::Compatibility& compatibility) {
    diamond_training::TrainingSample sample;
    sample.compatibility = compatibility;
    sample.node_features.assign(73U * 4U, 0.0F);
    sample.sparse_policy.emplace_back(0, 1.0F);
    sample.value_target.push_back(0.0F);
    return sample;
}

}  // namespace

int main(int argc, char** argv) {
    REQUIRE(argc == 2, "usage: native_pipeline_smoke_test <scratch>");
    const auto scratch = std::filesystem::path(argv[1]);
    std::filesystem::remove_all(scratch);

    const auto compatibility = diamond_training::Compatibility::soo(
        "1.0.0", {.residual_blocks = 1, .width = 8});
    const diamond_pipeline::ModelKey key{"Soo", "1.0.0", std::string(64, 'a')};
    auto model = diamond_model::DiamondModel(8, 1, 4, 1);
    diamond_pipeline::ModelPool models(1);
    models.install(key, model);
    models.activate(key);
    diamond_pipeline::ReplayStore replay(scratch / "replay", compatibility, 8, 7);
    diamond_training::Trainer trainer(model, compatibility,
                                      {.learning_rate = 1e-3, .weight_decay = 1e-4});

    diamond_pipeline::IterationRequest request;
    request.operation_id = "smoke-v2";
    request.model_key = key;
    request.compatibility = compatibility;
    request.training_steps = 0;
    const auto result = diamond_pipeline::run_iteration(request, models, replay, trainer, {});
    CHECK_EQ(result.operation_id, std::string("smoke-v2"));
    CHECK_EQ(result.completed_games, std::size_t{0});
    CHECK_EQ(result.aborted_games, std::size_t{0});
    CHECK_EQ(result.training_step, uint64_t{0});

    diamond_pipeline::Episode seed_episode;
    seed_episode.game_id = "seed-replay";
    seed_episode.seed = 11;
    seed_episode.retry_id = "attempt-0";
    seed_episode.model_key = key;
    seed_episode.compatibility = compatibility;
    seed_episode.samples.assign(4, training_sample(compatibility));
    seed_episode.final_order = {0, 1};
    seed_episode.move_count = 1;
    const std::vector<diamond_pipeline::Episode> seed_episodes{seed_episode};
    CHECK_EQ(replay.ingest(seed_episodes), std::size_t{1});

    diamond_pipeline::IterationRequest training_request;
    training_request.operation_id = "batch-contract-v2";
    training_request.model_key = key;
    training_request.compatibility = compatibility;
    training_request.training_batch_size = 2;
    training_request.training_steps = 2;
    const auto trained =
        diamond_pipeline::run_iteration(training_request, models, replay, trainer, {});
    CHECK_EQ(trained.requested_training_steps, std::size_t{2});
    CHECK_EQ(trained.completed_training_steps, std::size_t{2});
    CHECK_EQ(trained.training_batch_sizes.size(), std::size_t{2});
    CHECK_EQ(trained.training_batch_sizes[0], std::size_t{2});
    CHECK_EQ(trained.training_batch_sizes[1], std::size_t{2});
    CHECK_EQ(trained.replay_size, std::size_t{4});
    CHECK_EQ(trained.training_step, uint64_t{2});

    training_request.operation_id = "insufficient-replay-v2";
    training_request.training_batch_size = 5;
    training_request.training_steps = 1;
    bool rejected = false;
    try {
        (void)diamond_pipeline::run_iteration(training_request, models, replay, trainer, {});
    } catch (const std::exception& error) {
        rejected = true;
        CHECK_EQ(std::string(error.what()),
                 std::string("insufficient replay samples: requested 5, available 4"));
    }
    REQUIRE(rejected, "insufficient replay was accepted");
    CHECK_EQ(trainer.training_step(), uint64_t{2});
    return soo_test::report("native_pipeline_smoke_test");
}
