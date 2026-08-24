#include <filesystem>
#include <fstream>
#include <string>

#include "check.hpp"
#include "diamond_pipeline/replay_store.hpp"
#include "diamond_support/json.hpp"

int main(int argc, char** argv) {
    REQUIRE(argc == 3, "usage: replay_store_test <fixture-dir> <scratch-dir>");
    const diamond_pipeline::Compatibility compatibility =
        diamond_pipeline::Compatibility::soo("1.2.3", {.residual_blocks = 1, .width = 16});
    const auto fixtures = std::filesystem::path(argv[1]);
    const auto scratch = std::filesystem::path(argv[2]);
    std::filesystem::remove_all(scratch);
    std::filesystem::create_directories(scratch);
    CHECK_EQ(diamond_support::sha256(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK_EQ(diamond_support::sha256("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    CHECK_EQ(diamond_support::sha256("game-completed"), "ef9b00a3d875a4e0a49ed5de9f4e52b7eb89bb1bc8e12dae7f3d965ed43e1afc");

    // A fresh ingest must activate a durable v2 manifest whose descriptors survive
    // reopen and sampling; this is intentionally a scratch copy, never the fixture.
    const auto durable_root = scratch / "durable";
    diamond_pipeline::Episode episode;
    episode.game_id = "fresh-game";
    episode.completed = true;
    episode.compatibility = compatibility;
    episode.samples.resize(1);
    episode.samples[0].compatibility = compatibility;
    episode.samples[0].node_features.assign(73, 1.0F);
    episode.samples[0].canonical_player_ids = {1, 2};
    episode.samples[0].sparse_policy = {{4, 1.0F}};
    episode.samples[0].value_target = {1.0F};
    {
        diamond_pipeline::ReplayStore store(durable_root, compatibility, 8, 3);
        CHECK_EQ(store.ingest(std::span<const diamond_pipeline::Episode>(&episode, 1)), 1U);
    }
    {
        diamond_pipeline::ReplayStore store(durable_root, compatibility, 8, 3);
        const auto rows = store.sample(1);
        REQUIRE(rows.size() == 1, "reopened sample count");
        CHECK_EQ(rows[0].canonical_player_ids[0], 1);
    }
    {
        std::ifstream manifest(durable_root / "persistent-replay-v2" / "soo" /
                             diamond_support::sha256("{\"action_space_version\":\"diamond73-srcdst-v1\",\"board_topology_version\":\"diamond73-v1\",\"encoder_version\":\"diamond-camp-relative-v1\",\"model_name\":\"Soo\",\"model_version\":\"1.2.3\",\"network_config\":{\"residual_blocks\":1,\"width\":16},\"player_count\":2,\"ruleset_fingerprint\":\"sha256:02fff0c9c9436f247c4a2b5fb6b01903f658aae1c752377073011d0d150ba7a1\",\"ruleset_version\":\"diamond-authoritative-rules-v1\",\"seat_layout_version\":\"diamond-seat-layout-v1\",\"value_semantics_version\":\"current-player-scalar-winloss-v1\"}") / "manifest.json");
        CHECK(manifest.good());
    }
    bool corrupt = false;
    std::filesystem::copy(fixtures / "corrupt-digest", scratch / "corrupt-digest", std::filesystem::copy_options::recursive);
    try { diamond_pipeline::ReplayStore store(scratch / "corrupt-digest", compatibility, 8, 3); }
    catch (const std::runtime_error&) { corrupt = true; }
    CHECK(corrupt);
    std::filesystem::copy(fixtures / "capacity-prune", scratch / "capacity-prune", std::filesystem::copy_options::recursive);
    diamond_pipeline::ReplayStore capacity(scratch / "capacity-prune", compatibility, 3, 3);
    const auto rows = capacity.sample(3);
    REQUIRE(rows.size() == 3, "capacity fixture retains only reachable rows");
    std::filesystem::copy(fixtures / "rollback", scratch / "rollback", std::filesystem::copy_options::recursive);
    diamond_pipeline::ReplayStore rollback(scratch / "rollback", compatibility, 8, 3);
    rollback.restore_manifest(scratch / "rollback" / "persistent-replay-v1" / "Soo" / "3f3372c174dba4b7bfa9288e2c7e0a33e284dfdc3313f1d073210de8e47df229" / "before.json");
    CHECK_EQ(rollback.sample(1).size(), 1U);
    std::filesystem::remove_all(scratch);
    return soo_test::report("replay_store_test");
}
