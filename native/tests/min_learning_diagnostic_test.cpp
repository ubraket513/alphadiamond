#include <cmath>
#include <filesystem>
#include "check.hpp"
#include "diamond_model/soo_model.hpp"
#include "diamond_pipeline/learning_diagnostic.hpp"
#include "diamond_pipeline/replay_store.hpp"
#include "diamond_training/device.hpp"
#include "diamond_training/trainer.hpp"
#include "soo/board.hpp"

int main(int argc, char** argv) {
    REQUIRE(argc == 2, "usage: min_learning_diagnostic_test <scratch>");
    const auto scratch = std::filesystem::path(argv[1]);
    std::filesystem::remove_all(scratch);
    const auto compatibility =
        diamond_training::Compatibility::min("2.0.0", {.residual_blocks = 1, .width = 8});
    auto model = diamond_model::DiamondModel(8, 1, 6, 3);
    auto adjacency = torch::zeros({6, soo::kBoardSize, soo::kBoardSize});
    for (int64_t node = 0; node < soo::kBoardSize; ++node)
        adjacency[0][node][(node + 1) % soo::kBoardSize] = 1.0F;
    model->set_adjacency(adjacency);
    diamond_pipeline::ReplayStore replay(scratch / "replay", compatibility, 32, 7);
    diamond_pipeline::Episode episode;
    episode.game_id = "diagnostic";
    episode.retry_id = "attempt-0";
    episode.model_key = diamond_pipeline::ModelKey{
        .model_name = "Min", .model_version = "2.0.0", .checkpoint_sha256 = std::string(64, 'a')};
    episode.compatibility = compatibility;
    episode.final_order = {0, 1, 2};
    for (int index = 0; index < 8; ++index) {
        diamond_training::TrainingSample sample;
        sample.compatibility = compatibility;
        sample.node_features.assign(soo::kBoardSize * 6, 0.0F);
        sample.canonical_player_ids = {0, 1, 2};
        sample.sparse_policy = {{0, 0.6F}, {1, 0.4F}};
        sample.value_target = {1.0F, 0.0F, -1.0F};
        episode.samples.push_back(std::move(sample));
    }
    CHECK_EQ(replay.ingest(std::span<const diamond_pipeline::Episode>(&episode, 1)),
             std::size_t{1});
    const auto manifest_before = replay.manifest_digest();
    diamond_training::Trainer trainer(model, compatibility,
                                      {.learning_rate = 1e-3, .weight_decay = 1e-4},
                                      diamond_training::resolve_device("cpu"));
    const auto result = diamond_pipeline::run_min_learning_diagnostic(trainer, replay,
                                                                      {.iteration = 26,
                                                                       .steps = 2,
                                                                       .batch_size = 4,
                                                                       .evaluation_samples = 4,
                                                                       .evaluation_batch = 2,
                                                                       .log_every = 1,
                                                                       .seed = 9});
    CHECK_EQ(trainer.training_step(), uint64_t{2});
    CHECK_EQ(result.steps.size(), std::size_t{2});
    for (const auto group : {diamond_training::ParameterGroup::policy_source,
                             diamond_training::ParameterGroup::policy_destination}) {
        CHECK(result.steps.back().groups.at(group).gradient_l2 > 0.0);
        CHECK(result.steps.back().groups.at(group).update_l2 > 0.0);
    }
    CHECK(std::isfinite(result.initial.full_kl));
    CHECK(std::isfinite(result.final.full_kl));
    CHECK_EQ(replay.manifest_digest(), manifest_before);
    return soo_test::report("min_learning_diagnostic_test");
}
