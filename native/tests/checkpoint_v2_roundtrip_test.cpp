#include <filesystem>
#include "check.hpp"
#include "diamond_training/checkpoint.hpp"

int main(int argc, char** argv) {
    REQUIRE(argc == 2, "usage: checkpoint_v2_roundtrip_test <scratch>");
    const auto scratch = std::filesystem::path(argv[1]);
    std::filesystem::remove_all(scratch);
    auto compatibility = diamond_training::Compatibility::soo("1.0.0", {.residual_blocks = 1, .width = 8});
    auto model = diamond_model::DiamondModel(8, 1, 4, 1);
    diamond_training::Trainer saved(model, compatibility, {.learning_rate = 1e-3, .weight_decay = 1e-4});
    {
        torch::NoGradGuard no_grad;
        saved.model()->parameters().front().fill_(0.25);
    }
    const auto written = diamond_training::save_checkpoint_v2(scratch, saved);
    CHECK(std::filesystem::exists(written.generation / "state.pt"));
    CHECK(std::filesystem::exists(written.generation / "optimizer.pt"));
    CHECK_EQ(diamond_training::inspect_checkpoint_v2(scratch).training_step, uint64_t{0});
    diamond_training::Trainer restored(diamond_model::DiamondModel(8, 1, 4, 1), compatibility, {.learning_rate = 1e-3, .weight_decay = 1e-4});
    {
        torch::NoGradGuard no_grad;
        restored.model()->parameters().front().zero_();
    }
    diamond_training::load_checkpoint_v2(scratch, restored);
    CHECK(torch::allclose(saved.model()->parameters().front(), restored.model()->parameters().front()));
    CHECK_EQ(restored.training_step(), uint64_t{0});
    std::filesystem::remove_all(scratch);
    return soo_test::report("checkpoint_v2_roundtrip_test");
}
