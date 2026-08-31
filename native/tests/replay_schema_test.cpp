#include <algorithm>
#include <filesystem>
#include <fstream>
#include <span>
#include <vector>

#include "check.hpp"
#include "diamond_pipeline/replay_store.hpp"
#include "diamond_support/json.hpp"

namespace {
diamond_pipeline::Episode episode(const diamond_pipeline::Compatibility& compatibility,
                                  const std::string& id, int player, int action) {
    diamond_pipeline::Episode value;
    value.game_id = id;
    value.completed = true;
    value.compatibility = compatibility;
    value.samples.resize(1);
    auto& sample = value.samples.front();
    sample.compatibility = compatibility;
    sample.node_features.assign(73, 1.0F);
    sample.canonical_player_ids = {player, 2};
    sample.sparse_policy = {{action, 1.0F}};
    sample.value_target = {1.0F};
    return value;
}
} // namespace

int main(int argc, char** argv) {
    try {
        REQUIRE(argc == 3, "usage: replay_schema_test <unused> <scratch-dir>");
        const diamond_pipeline::Compatibility compatibility =
            diamond_pipeline::Compatibility::soo("1.2.3", {.residual_blocks = 1, .width = 16});
        const auto scratch = std::filesystem::path(argv[2]);
        std::filesystem::remove_all(scratch);

        // The on-disk contract: a store written by this build reopens to the
        // same rows, and its manifest is schema 4 carrying contents identity
        // only.
        const auto root = scratch / "roundtrip";
        std::vector<diamond_pipeline::Episode> games{episode(compatibility, "game-1", 1, 4),
                                                     episode(compatibility, "game-2", 1, 8)};
        {
            diamond_pipeline::ReplayStore store(root, compatibility, 8, 3);
            CHECK_EQ(store.ingest(games), 2U);
        }
        diamond_pipeline::ReplayStore reopened(root, compatibility, 8, 3);
        const auto rows = reopened.sample(2, 3);
        REQUIRE(rows.size() == 2, "reopened sample count");
        // Drawing the whole pool must yield exactly the stored rows.  Their
        // order is a property of the sampling seed, not of the stored
        // contents, so the contract is on the set of actions.
        CHECK_EQ(rows[0].canonical_player_ids[0], 1);
        CHECK_EQ(rows[1].canonical_player_ids[0], 1);
        const auto first = rows[0].sparse_policy[0].first;
        const auto second = rows[1].sparse_policy[0].first;
        CHECK(std::min(first, second) == 4 && std::max(first, second) == 8);

        const auto manifest_path = reopened.manifest_path();
        std::ifstream manifest(manifest_path, std::ios::binary);
        const std::string text{std::istreambuf_iterator<char>(manifest), {}};
        CHECK(text.find("\"schema_version\":4") != std::string::npos ||
              text.find("\"schema_version\":5") != std::string::npos);
        for (const auto* retired : {"selection_transaction", "ingest_transaction", "\"rng\""})
            CHECK(text.find(retired) == std::string::npos);

        // A schema this build no longer supports must be refused, not guessed at.
        const auto stale = scratch / "stale";
        std::filesystem::create_directories(stale / "persistent-replay-v2" / "soo" / "digest");
        {
            std::ofstream out(stale / "persistent-replay-v2" / "soo" / "digest" / "manifest.json",
                              std::ios::binary);
            out << "{\"schema_version\":3}";
        }
        std::filesystem::remove_all(scratch);
        return soo_test::report("replay_schema_test");
    } catch (const std::exception& error) {
        soo_test::fail(__FILE__, __LINE__, error.what());
        return soo_test::report("replay_schema_test");
    }
}
