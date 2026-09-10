#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "check.hpp"
#include "diamond_pipeline/replay_segment.hpp"

namespace {

diamond_pipeline::Episode fixture(const diamond_pipeline::Compatibility& compatibility) {
    diamond_pipeline::Episode episode;
    episode.game_id = "binary-game-17";
    episode.seed = 0x0102030405060708ULL;
    episode.retry_id = "retry-2";
    episode.move_count = 41;
    episode.completed = true;
    episode.compatibility = compatibility;
    episode.samples.resize(2);
    episode.samples[0].compatibility = compatibility;
    episode.samples[0].node_features = {-1.25F, 0.0F, 2.5F};
    episode.samples[0].canonical_player_ids = {2, 0, 1};
    episode.samples[0].sparse_policy = {{4, 0.25F}, {72, 0.75F}};
    episode.samples[0].value_target = {-1.0F, 0.5F, 1.0F};
    episode.samples[1].compatibility = compatibility;
    episode.samples[1].node_features = {3.0F};
    episode.samples[1].canonical_player_ids = {1, 2, 0};
    episode.samples[1].sparse_policy = {{0, 1.0F}};
    episode.samples[1].value_target = {0.0F, 1.0F, -1.0F};
    return episode;
}

bool throws_decode(std::span<const std::byte> bytes,
                   const diamond_pipeline::Compatibility& compatibility) {
    try {
        (void)diamond_pipeline::decode_replay_segment(bytes, compatibility);
        return false;
    } catch (const std::runtime_error&) {
        return true;
    }
}

}  // namespace

int main() {
    const auto compatibility = diamond_pipeline::Compatibility::min(
        "binary-test-1", {.residual_blocks = 2, .width = 16});
    const auto source = fixture(compatibility);
    const auto first = diamond_pipeline::encode_replay_segment(source, compatibility);
    const auto second = diamond_pipeline::encode_replay_segment(source, compatibility);
    CHECK(first == second);

    const auto decoded = diamond_pipeline::decode_replay_segment(first, compatibility);
    CHECK_EQ(decoded.game_id, source.game_id);
    CHECK_EQ(decoded.seed, source.seed);
    CHECK_EQ(decoded.retry_id, source.retry_id);
    CHECK_EQ(decoded.move_count, source.move_count);
    CHECK_EQ(decoded.samples.size(), source.samples.size());
    for (std::size_t i = 0; i < source.samples.size(); ++i) {
        CHECK(decoded.samples[i].compatibility == compatibility);
        CHECK(decoded.samples[i].node_features == source.samples[i].node_features);
        CHECK(decoded.samples[i].canonical_player_ids == source.samples[i].canonical_player_ids);
        CHECK(decoded.samples[i].sparse_policy == source.samples[i].sparse_policy);
        CHECK(decoded.samples[i].value_target == source.samples[i].value_target);
    }

    for (std::size_t length = 0; length < first.size(); ++length)
        CHECK(throws_decode(std::span<const std::byte>(first).first(length), compatibility));

    auto corrupt = first;
    corrupt[corrupt.size() / 2] ^= std::byte{1};
    CHECK(throws_decode(corrupt, compatibility));

    auto incompatible = compatibility;
    incompatible.model_version = "different";
    CHECK(throws_decode(first, incompatible));

    auto non_finite = source;
    non_finite.samples[0].node_features[0] = std::numeric_limits<float>::infinity();
    bool rejected_non_finite = false;
    try {
        (void)diamond_pipeline::encode_replay_segment(non_finite, compatibility);
    } catch (const std::invalid_argument&) {
        rejected_non_finite = true;
    }
    CHECK(rejected_non_finite);

    return soo_test::report("replay_segment_test");
}
