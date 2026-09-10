#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "check.hpp"
#include "diamond_pipeline/replay_store.hpp"
#include "diamond_support/json.hpp"

namespace {
void set_environment(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    if (*value)
        setenv(name, value, 1);
    else
        unsetenv(name);
#endif
}

diamond_pipeline::Episode episode(const diamond_pipeline::Compatibility& compatibility, int id) {
    diamond_pipeline::Episode value;
    value.game_id = "game-" + std::to_string(id);
    value.completed = true;
    value.compatibility = compatibility;
    value.samples.resize(1);
    auto& sample = value.samples.front();
    sample.compatibility = compatibility;
    sample.node_features.assign(73, 1.0F);
    sample.canonical_player_ids = {id, 2};
    sample.sparse_policy = {{4, 1.0F}};
    sample.value_target = {1.0F};
    return value;
}

std::vector<int> ids(const std::vector<diamond_pipeline::TrainingSample>& rows) {
    std::vector<int> result;
    for (const auto& row : rows)
        result.push_back(row.canonical_player_ids.front());
    return result;
}

std::size_t chunk_file_count(const std::filesystem::path& root) {
    std::size_t count = 0;
    std::error_code error;
    for (std::filesystem::recursive_directory_iterator it(root, error), end; !error && it != end;
         it.increment(error)) {
        if (it->is_regular_file(error) && !error && it->path().parent_path().filename() == "chunks")
            ++count;
    }
    if (error)
        throw std::runtime_error("cannot inspect replay chunks");
    return count;
}

void remove_tree(const std::filesystem::path& path, std::error_code& error) {
#ifdef _WIN32
    std::wstring native = std::filesystem::absolute(path).wstring();
    if (native.rfind(L"\\\\?\\", 0) != 0) {
        if (native.rfind(L"\\\\", 0) == 0)
            native = L"\\\\?\\UNC\\" + native.substr(2);
        else
            native = L"\\\\?\\" + native;
    }
    std::filesystem::remove_all(std::filesystem::path(native), error);
#else
    std::filesystem::remove_all(path, error);
#endif
}
} // namespace

int main(int argc, char** argv) {
    REQUIRE(argc == 3, "usage: replay_store_test <unused> <scratch-dir>");
    const diamond_pipeline::Compatibility compatibility =
        diamond_pipeline::Compatibility::soo("1.2.3", {.residual_blocks = 1, .width = 16});
    const auto scratch = std::filesystem::path(argv[2]);
    std::error_code cleanup_error;
    remove_tree(scratch, cleanup_error);
    REQUIRE(!cleanup_error, "remove scratch before replay test");
    std::filesystem::create_directories(scratch);
    const auto missing_read_only = scratch / "missing-read-only";
    bool missing_threw = false;
    try {
        diamond_pipeline::ReplayStore missing(missing_read_only, compatibility, 8, 3,
                                              diamond_pipeline::ReplayContents::full,
                                              diamond_pipeline::ReplayOpenMode::must_exist);
    } catch (const std::runtime_error&) {
        missing_threw = true;
    }
    CHECK(missing_threw);
    CHECK(!std::filesystem::exists(missing_read_only));

    CHECK_EQ(diamond_support::sha256(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK_EQ(diamond_support::sha256("abc"),
             "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");

    const auto durable_root = scratch / "durable";
    auto fresh = episode(compatibility, 1);
    std::vector<int> expected_first;
    {
        diamond_pipeline::ReplayStore store(durable_root, compatibility, 8, 3);
        CHECK_EQ(store.ingest(std::span<const diamond_pipeline::Episode>(&fresh, 1)), 1U);
    }
    {
        diamond_pipeline::ReplayStore store(durable_root, compatibility, 8, 3);
        const auto rows = store.sample(1, 3);
        REQUIRE(rows.size() == 1, "reopened sample count");
        CHECK_EQ(rows[0].canonical_player_ids[0], 1);
    }

#ifdef _WIN32
    auto long_root = scratch / "windows-long-path";
    while ((std::filesystem::absolute(long_root) / "persistent-replay-v2" / "soo" /
            std::string(64, 'a') / "chunks" / (std::string(64, 'b') + ".json"))
               .native()
               .size() <= 270) {
        long_root /= "long-segment-0123456789abcdef";
    }
    {
        diamond_pipeline::ReplayStore store(long_root, compatibility, 8, 3);
        CHECK_EQ(store.ingest(std::span<const diamond_pipeline::Episode>(&fresh, 1)), 1U);
    }
    {
        diamond_pipeline::ReplayStore reopened(long_root, compatibility, 8, 3);
        REQUIRE(reopened.sample(1, 3).size() == 1, "Windows extended-path replay reopen");
    }
#endif

    std::vector<diamond_pipeline::Episode> pool;
    for (int id = 10; id < 74; ++id)
        pool.push_back(episode(compatibility, id));
    const auto selection_root = scratch / "selection";
    {
        diamond_pipeline::ReplayStore store(selection_root, compatibility, 128, 17);
        const auto report = store.ingest_iteration(pool);
        CHECK_EQ(report.accepted_games, 64U);
        CHECK_EQ(report.accepted_samples, 64U);
        CHECK_EQ(store.ingest_iteration(pool).duplicate_games, 64U);
        const auto step_seed = diamond_pipeline::replay_sampling_seed(17, 5, 0);
        expected_first = ids(store.sample(4, step_seed));
        std::unordered_set<int> unique(expected_first.begin(), expected_first.end());
        CHECK_EQ(unique.size(), 4U);
        const auto stats = store.last_sampling_stats();
        CHECK(stats.selection_slots <= 8U);
        CHECK_EQ(stats.copied_samples, 4U);
        // Stateless: the same seed reselects the same rows, a different local
        // step selects different ones, and neither writes anything.
        const auto manifest_before = store.manifest_digest();
        CHECK(ids(store.sample(4, step_seed)) == expected_first);
        CHECK(ids(store.sample(4, diamond_pipeline::replay_sampling_seed(17, 5, 1))) !=
              expected_first);
        CHECK_EQ(store.manifest_digest(), manifest_before);
    }
    {
        // A reopened store with the same contents replays the same sequence.
        diamond_pipeline::ReplayStore reopened(selection_root, compatibility, 128, 17);
        CHECK(ids(reopened.sample(4, diamond_pipeline::replay_sampling_seed(17, 5, 0))) ==
              expected_first);
    }
    const auto pre_activation_root = scratch / "pre-activation";
    {
        // Sampling touches no file, so an activation failure injected around it
        // cannot be reached at all: the draw succeeds and the manifest written
        // by ingest is untouched.
        diamond_pipeline::ReplayStore store(pre_activation_root, compatibility, 128, 17);
        (void)store.ingest(pool);
        const auto manifest_before = store.manifest_digest();
        set_environment("DIAMOND_REPLAY_FAIL_ACTIVATE", "1");
        const auto drawn = ids(store.sample(4, diamond_pipeline::replay_sampling_seed(17, 5, 0)));
        set_environment("DIAMOND_REPLAY_FAIL_ACTIVATE", "");
        CHECK(drawn == expected_first);
        CHECK_EQ(store.manifest_digest(), manifest_before);
    }

    const auto rollback_root = scratch / "in-memory-rollback";
    {
        diamond_pipeline::ReplayStore store(rollback_root, compatibility, 8, 23);
        CHECK_EQ(store.ingest(std::span<const diamond_pipeline::Episode>(&fresh, 1)), 1U);
        const auto size_before = store.size();
        const auto manifest_before = store.manifest_digest();
        auto next = episode(compatibility, 2);
        set_environment("DIAMOND_REPLAY_FAIL_BEFORE_MANIFEST_COMMIT", "1");
        bool failed = false;
        try {
            (void)store.ingest(std::span<const diamond_pipeline::Episode>(&next, 1));
        } catch (const std::runtime_error&) {
            failed = true;
        }
        set_environment("DIAMOND_REPLAY_FAIL_BEFORE_MANIFEST_COMMIT", "");
        CHECK(failed);
        CHECK_EQ(store.size(), size_before);
        CHECK_EQ(store.manifest_digest(), manifest_before);
        CHECK_EQ(store.ingest(std::span<const diamond_pipeline::Episode>(&next, 1)), 1U);
        CHECK_EQ(store.size(), size_before + 1U);
    }

    const auto orphan_root = scratch / "orphan-recovery";
    {
        diamond_pipeline::ReplayStore store(orphan_root, compatibility, 8, 19);
        set_environment("DIAMOND_REPLAY_FAIL_AFTER_CHUNK_ACTIVATE", "1");
        bool failed = false;
        try {
            (void)store.ingest(std::span<const diamond_pipeline::Episode>(&fresh, 1));
        } catch (const std::runtime_error&) {
            failed = true;
        }
        set_environment("DIAMOND_REPLAY_FAIL_AFTER_CHUNK_ACTIVATE", "");
        CHECK(failed);
        const auto orphan_chunks = chunk_file_count(orphan_root);
        if (orphan_chunks != 1U)
            std::fprintf(stderr, "orphan-recovery: expected 1 chunk, found %zu\n", orphan_chunks);
        CHECK_EQ(orphan_chunks, 1U);
    }
    {
        diamond_pipeline::ReplayStore recovered(orphan_root, compatibility, 8, 19);
        CHECK_EQ(chunk_file_count(orphan_root), 0U);
        CHECK_EQ(recovered.size(), 0U);
    }
    {
        std::ifstream manifest(
            selection_root / "persistent-replay-v2" / "soo" /
            diamond_support::sha256(
                "{\"action_space_version\":\"diamond73-srcdst-v1\",\"board_topology_version\":"
                "\"diamond73-v1\",\"encoder_version\":\"diamond-camp-relative-v1\",\"model_name\":"
                "\"Soo\",\"model_version\":\"1.2.3\",\"network_config\":{\"residual_blocks\":1,"
                "\"width\":16},\"player_count\":2,\"ruleset_fingerprint\":\"sha256:"
                "02fff0c9c9436f247c4a2b5fb6b01903f658aae1c752377073011d0d150ba7a1\",\"ruleset_"
                "version\":\"diamond-authoritative-rules-v1\",\"seat_layout_version\":\"diamond-"
                "seat-layout-v1\",\"value_semantics_version\":\"current-player-scalar-winloss-"
                "v1\"}") /
            "manifest.json");
        const std::string contents{std::istreambuf_iterator<char>(manifest), {}};
        // Contents identity only: no sampler state, no transaction records.
        CHECK(contents.find("\"schema_version\":4") != std::string::npos ||
              contents.find("\"schema_version\":5") != std::string::npos);
        CHECK(contents.find("selection_transaction") == std::string::npos);
        CHECK(contents.find("ingest_transaction") == std::string::npos);
        CHECK(contents.find("\"rng\"") == std::string::npos);
    }

    // Corrupt chunk detection, built natively rather than from a frozen v1
    // fixture: ingest, then flip a byte inside a chunk body so its content no
    // longer hashes to the name the manifest recorded.
    {
        const auto corrupt_root = scratch / "corrupt-digest";
        {
            diamond_pipeline::ReplayStore store(corrupt_root, compatibility, 8, 3);
            CHECK_EQ(store.ingest(std::span<const diamond_pipeline::Episode>(&fresh, 1)), 1U);
        }
        std::filesystem::path chunk;
        for (std::filesystem::recursive_directory_iterator it(corrupt_root), end; it != end; ++it)
            if (it->is_regular_file() && it->path().parent_path().filename() == "chunks")
                chunk = it->path();
        REQUIRE(!chunk.empty(), "corrupt-digest chunk located");
        std::string body;
        {
            std::ifstream input(chunk, std::ios::binary);
            body.assign(std::istreambuf_iterator<char>(input), {});
        }
        REQUIRE(!body.empty(), "corrupt-digest payload is editable");
        body[body.size() / 2] ^= char{1};
        {
            std::ofstream output(chunk, std::ios::binary | std::ios::trunc);
            output << body;
        }
        bool corrupt = false;
        try {
            diamond_pipeline::ReplayStore store(corrupt_root, compatibility, 8, 3);
        } catch (const std::runtime_error&) {
            corrupt = true;
        }
        CHECK(corrupt);
    }
    // Capacity bounds the durable manifest as well as the sampling pool. Old
    // chunks must become unreachable during ordinary ingest so legacy JSON is
    // progressively replaced instead of being parsed forever on cold open.
    {
        const auto capacity_root = scratch / "capacity";
        {
            diamond_pipeline::ReplayStore store(capacity_root, compatibility, 3, 3);
            CHECK_EQ(store.ingest(pool), 64U);
            CHECK_EQ(chunk_file_count(capacity_root), 3U);
        }
        diamond_pipeline::ReplayStore capacity(capacity_root, compatibility, 3, 3);
        CHECK_EQ(capacity.size(), 3U);
        REQUIRE(capacity.sample(3, 3).size() == 3, "capacity bounds the reopened pool");
        CHECK_EQ(chunk_file_count(capacity_root), 3U);
    }
    // Metadata-only opens the manifest and nothing else: it reports the same
    // size and digest as a full open, refuses to sample, and never reads a
    // chunk -- which is what lets a digest-only stage skip rehydrating 1M rows.
    {
        const auto meta_root = scratch / "metadata-only";
        std::string full_digest;
        std::size_t full_size = 0;
        {
            diamond_pipeline::ReplayStore store(meta_root, compatibility, 128, 3);
            CHECK_EQ(store.ingest(pool), 64U);
            full_digest = store.manifest_digest();
            full_size = store.size();
        }
        diamond_pipeline::ReplayStore meta(meta_root, compatibility, 128, 3,
                                           diamond_pipeline::ReplayContents::metadata_only);
        CHECK_EQ(meta.manifest_digest(), full_digest);
        CHECK_EQ(meta.size(), full_size);
        bool refused = false;
        try {
            (void)meta.sample(1, 3);
        } catch (const std::logic_error&) {
            refused = true;
        }
        CHECK(refused);
    }
    remove_tree(scratch, cleanup_error);
    CHECK(!cleanup_error);
    return soo_test::report("replay_store_test");
}
