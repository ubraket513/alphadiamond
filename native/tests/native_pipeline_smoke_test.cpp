#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stop_token>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.hpp"
#include "diamond_model/soo_model.hpp"
#include "diamond_pipeline/pipeline.hpp"
#include "diamond_training/trainer.hpp"
#include "soo/board.hpp"

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

soo::Match soo_match() {
    soo::ensure_topology_configured();
    soo::Match match;
    match.count = 2;
    match.players[0] = {1, 0, 3};
    match.players[1] = {2, 3, 0};
    return match;
}

soo::State opening(const soo::Match& match) {
    soo::State state;
    for (std::size_t seat = 0; seat < match.count; ++seat) {
        for (const uint8_t position : soo::topology().camp_positions[match.players[seat].camp])
            state.occupancy[position] = match.players[seat].id;
    }
    state.current_player = match.players[0].id;
    return state;
}

std::string replay_manifest(const std::filesystem::path& replay_root) {
    std::error_code error;
    std::filesystem::recursive_directory_iterator entry(replay_root, error);
    const std::filesystem::recursive_directory_iterator end;
    for (; entry != end && !error; entry.increment(error)) {
        if (entry->path().filename() != "manifest.json") continue;
        std::ifstream input(entry->path(), std::ios::binary);
        REQUIRE(static_cast<bool>(input), "cannot open replay manifest");
        return {std::istreambuf_iterator<char>(input), {}};
    }
    REQUIRE(!error, error.message().c_str());
    return {};
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

    diamond_pipeline::IterationRequest deadline_request;
    deadline_request.operation_id = "deadline-abort-v2";
    deadline_request.model_key = key;
    deadline_request.compatibility = compatibility;
    deadline_request.match = soo_match();
    deadline_request.jobs = {{opening(deadline_request.match), 19}};
    deadline_request.selfplay.lanes = 1;
    deadline_request.selfplay.threads = 1;
    deadline_request.selfplay.max_batch = 1;
    deadline_request.selfplay.max_wait_us = 1;
    deadline_request.selfplay.simulations = 1;
    deadline_request.selfplay.max_moves = 1000;
    deadline_request.selfplay.max_game_duration = std::chrono::nanoseconds(1);
    diamond_pipeline::IterationResult aborted;
    try {
        aborted = diamond_pipeline::run_iteration(deadline_request, models, replay, trainer, {});
    } catch (const std::exception& error) {
        REQUIRE(false, error.what());
    }
    CHECK_EQ(aborted.completed_games, std::size_t{0});
    CHECK_EQ(aborted.aborted_games, std::size_t{1});
    CHECK_EQ(aborted.new_samples, std::size_t{0});
    CHECK_EQ(aborted.replay_size, std::size_t{0});
    CHECK(replay_manifest(scratch / "replay").find("max_game_seconds") != std::string::npos);

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
