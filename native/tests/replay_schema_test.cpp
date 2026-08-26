#include <algorithm>
#include <filesystem>
#include <fstream>

#include "check.hpp"
#include "diamond_pipeline/replay_store.hpp"

namespace {
void copy_fixture(const std::filesystem::path& source, const std::filesystem::path& destination) {
    for (const auto& entry : std::filesystem::recursive_directory_iterator(source)) {
        if (!entry.is_regular_file()) continue;
        const auto target = destination / std::filesystem::relative(entry.path(), source);
        std::filesystem::create_directories(target.parent_path());
        std::ifstream input(entry.path(), std::ios::binary);
        std::ofstream output(target, std::ios::binary | std::ios::trunc);
        output << input.rdbuf();
    }
}
}

int main(int argc, char** argv) {
    try {
    REQUIRE(argc == 3, "usage: replay_schema_test <fixture-dir> <scratch-dir>");
    const diamond_pipeline::Compatibility compatibility =
        diamond_pipeline::Compatibility::soo("1.2.3", {.residual_blocks = 1, .width = 16});
    const auto fixture_dir = std::filesystem::path(argv[1]);
    const auto scratch = std::filesystem::path(argv[2]);
    std::filesystem::remove_all(scratch);
    copy_fixture(fixture_dir / "completed", scratch / "completed");
    diamond_pipeline::ReplayStore completed(scratch / "completed", compatibility, 8, 3);
    const auto rows = completed.sample(2, 3);
    REQUIRE(rows.size() == 2, "completed fixture sample count");
    // Drawing the whole pool must yield exactly the fixture's rows.  Their
    // order is a property of the sampling seed, not of the stored contents,
    // so the contract is on the set of actions, not on the permutation.
    CHECK_EQ(rows[0].canonical_player_ids[0], 1);
    CHECK_EQ(rows[1].canonical_player_ids[0], 1);
    const auto first = rows[0].sparse_policy[0].first;
    const auto second = rows[1].sparse_policy[0].first;
    CHECK(std::min(first, second) == 4 && std::max(first, second) == 8);
    return soo_test::report("replay_schema_test");
    } catch (const std::exception& error) { soo_test::fail(__FILE__, __LINE__, error.what()); return soo_test::report("replay_schema_test"); }
}
