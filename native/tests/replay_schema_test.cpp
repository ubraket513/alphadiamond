#include <filesystem>

#include "check.hpp"
#include "diamond_pipeline/replay_store.hpp"

int main(int argc, char** argv) {
    REQUIRE(argc == 2, "usage: replay_schema_test <fixture-dir>");
    const diamond_pipeline::Compatibility compatibility{"soo", "1.2.3"};
    const auto fixture_dir = std::filesystem::path(argv[1]);
    REQUIRE(std::filesystem::exists(fixture_dir / "completed"), "replay-v1 fixtures missing");
    const auto scratch = std::filesystem::temp_directory_path() / "alphadiamond-replay-schema-test";
    std::filesystem::remove_all(scratch);
    diamond_pipeline::ReplayStore store(scratch, compatibility, 8, 3);
    CHECK(store.sample(0).empty());
    std::filesystem::remove_all(scratch);
    return soo_test::report("replay_schema_test");
}
