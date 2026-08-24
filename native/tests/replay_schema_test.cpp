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
    const diamond_pipeline::Compatibility compatibility{"soo", "1.2.3"};
    const auto fixture_dir = std::filesystem::path(argv[1]);
    const auto scratch = std::filesystem::path(argv[2]);
    std::filesystem::remove_all(scratch);
    copy_fixture(fixture_dir / "completed", scratch / "completed");
    diamond_pipeline::ReplayStore completed(scratch / "completed", compatibility, 8, 3);
    const auto rows = completed.sample(2);
    REQUIRE(rows.size() == 2, "completed fixture sample count");
    CHECK_EQ(rows[0].canonical_player_ids[0], 1);
    CHECK_EQ(rows[0].sparse_policy[0].first, 4);
    CHECK_EQ(rows[1].sparse_policy[0].first, 8);
    return soo_test::report("replay_schema_test");
    } catch (const std::exception& error) { soo_test::fail(__FILE__, __LINE__, error.what()); return soo_test::report("replay_schema_test"); }
}
