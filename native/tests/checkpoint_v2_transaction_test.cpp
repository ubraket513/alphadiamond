#include <cstdlib>
#include <filesystem>
#include <fstream>

#include "check.hpp"
#include "diamond_training/checkpoint.hpp"

int main(int argc, char** argv) {
    REQUIRE(argc == 2, "usage: checkpoint_v2_transaction_test <scratch>");
    const auto scratch = std::filesystem::path(argv[1]); std::filesystem::remove_all(scratch);
    auto compatibility = diamond_training::Compatibility::soo("1.0.0", {.residual_blocks = 1, .width = 8});
    diamond_training::Trainer trainer(diamond_model::DiamondModel(8, 1, 4, 1), compatibility, {.learning_rate = 1e-3, .weight_decay = 1e-4});
    const auto first = diamond_training::save_checkpoint_v2(scratch, trainer);
    const auto current_before = [&] { std::ifstream in(scratch / "CURRENT"); std::string value; std::getline(in, value); return value; }();
    _putenv_s("DIAMOND_CHECKPOINT_FAIL_ACTIVATE", "1");
    bool failed = false; try { (void)diamond_training::save_checkpoint_v2(scratch, trainer); } catch (const diamond_training::CheckpointError&) { failed = true; }
    _putenv_s("DIAMOND_CHECKPOINT_FAIL_ACTIVATE", "");
    CHECK(failed);
    const auto current_after = [&] { std::ifstream in(scratch / "CURRENT"); std::string value; std::getline(in, value); return value; }();
    CHECK_EQ(current_after, current_before); CHECK(std::filesystem::exists(first.generation));
    std::filesystem::remove_all(scratch);
    return soo_test::report("checkpoint_v2_transaction_test");
}
