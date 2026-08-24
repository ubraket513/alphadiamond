#include <filesystem>
#include <fstream>

#include "check.hpp"
#include "diamond_training/checkpoint.hpp"

int main(int argc, char** argv) {
    REQUIRE(argc == 2, "usage: checkpoint_v2_reject_test <scratch>");
    const auto scratch = std::filesystem::path(argv[1]);
    std::filesystem::remove_all(scratch); std::filesystem::create_directories(scratch);
    std::ofstream(scratch / "legacy.pt") << "v1";
    bool rejected = false;
    try { (void)diamond_training::inspect_checkpoint_v2(scratch / "legacy.pt"); }
    catch (const diamond_training::CheckpointError&) { rejected = true; }
    CHECK(rejected);
    std::filesystem::remove_all(scratch);
    return soo_test::report("checkpoint_v2_reject_test");
}
