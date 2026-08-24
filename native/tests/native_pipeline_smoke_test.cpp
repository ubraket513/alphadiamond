#include <filesystem>
#include <stop_token>
#include <string>

#include "check.hpp"
#include "diamond_model/soo_model.hpp"
#include "diamond_pipeline/pipeline.hpp"
#include "diamond_training/trainer.hpp"

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
    return soo_test::report("native_pipeline_smoke_test");
}
