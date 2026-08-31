#include "diamond_pipeline/replay_segment.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

#include "diamond_support/json.hpp"

namespace diamond_pipeline {
namespace {

constexpr std::array<std::byte, 8> magic = {
    std::byte{'A'}, std::byte{'D'}, std::byte{'R'}, std::byte{'P'},
    std::byte{'B'}, std::byte{'I'}, std::byte{'N'}, std::byte{'1'}};
constexpr std::uint32_t format_version = 1;
constexpr std::uint32_t header_size = 72;
constexpr std::size_t digest_size = 32;

void append_u32(std::vector<std::byte>& out, std::uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8)
        out.push_back(std::byte((value >> shift) & 0xffU));
}

void append_u64(std::vector<std::byte>& out, std::uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8)
        out.push_back(std::byte((value >> shift) & 0xffU));
}

void append_bytes(std::vector<std::byte>& out, std::string_view value) {
    out.insert(out.end(), reinterpret_cast<const std::byte*>(value.data()),
               reinterpret_cast<const std::byte*>(value.data() + value.size()));
}

void append_string(std::vector<std::byte>& out, std::string_view value) {
    if (value.size() > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("replay segment string is too large");
    append_u32(out, static_cast<std::uint32_t>(value.size()));
    append_bytes(out, value);
}

void append_float(std::vector<std::byte>& out, float value) {
    if (!std::isfinite(value))
        throw std::invalid_argument("replay segment contains a non-finite float");
    append_u32(out, std::bit_cast<std::uint32_t>(value));
}

std::array<std::byte, digest_size> digest(std::span<const std::byte> bytes) {
    const auto hex = diamond_support::sha256(std::string_view(
        reinterpret_cast<const char*>(bytes.data()), bytes.size()));
    std::array<std::byte, digest_size> result{};
    auto nibble = [](char value) -> unsigned {
        if (value >= '0' && value <= '9') return unsigned(value - '0');
        if (value >= 'a' && value <= 'f') return unsigned(value - 'a' + 10);
        throw std::logic_error("sha256 returned invalid hexadecimal");
    };
    for (std::size_t i = 0; i < result.size(); ++i)
        result[i] = std::byte((nibble(hex[i * 2]) << 4) | nibble(hex[i * 2 + 1]));
    return result;
}

std::array<std::byte, digest_size> compatibility_digest(const Compatibility& value) {
    std::vector<std::byte> bytes;
    append_string(bytes, value.model_name);
    append_string(bytes, value.model_version);
    append_u64(bytes, static_cast<std::uint64_t>(value.player_count));
    append_string(bytes, value.ruleset_version);
    append_string(bytes, value.board_topology_version);
    append_string(bytes, value.ruleset_fingerprint);
    append_string(bytes, value.encoder_version);
    append_string(bytes, value.action_space_version);
    append_string(bytes, value.seat_layout_version);
    append_string(bytes, value.value_semantics_version);
    append_u64(bytes, static_cast<std::uint64_t>(value.network_config.residual_blocks));
    append_u64(bytes, static_cast<std::uint64_t>(value.network_config.width));
    return digest(bytes);
}

void append_count(std::vector<std::byte>& out, std::size_t count) {
    if (count > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("replay segment array is too large");
    append_u32(out, static_cast<std::uint32_t>(count));
}

std::vector<std::byte> encode_sample(const TrainingSample& sample,
                                     const Compatibility& compatibility) {
    if (sample.compatibility != compatibility)
        throw std::invalid_argument("replay sample compatibility mismatch");
    std::vector<std::byte> out;
    append_count(out, sample.canonical_player_ids.size());
    for (const auto value : sample.canonical_player_ids)
        append_u32(out, std::bit_cast<std::uint32_t>(value));
    append_count(out, sample.node_features.size());
    for (const auto value : sample.node_features) append_float(out, value);
    append_count(out, sample.sparse_policy.size());
    for (const auto& [action, probability] : sample.sparse_policy) {
        append_u32(out, std::bit_cast<std::uint32_t>(action));
        append_float(out, probability);
    }
    append_count(out, sample.value_target.size());
    for (const auto value : sample.value_target) append_float(out, value);
    return out;
}

class Reader {
public:
    explicit Reader(std::span<const std::byte> bytes) : bytes_(bytes) {}

    std::uint32_t u32() {
        require(4);
        std::uint32_t value = 0;
        for (unsigned shift = 0; shift < 32; shift += 8)
            value |= std::uint32_t(std::to_integer<unsigned>(bytes_[position_++])) << shift;
        return value;
    }

    std::uint64_t u64() {
        require(8);
        std::uint64_t value = 0;
        for (unsigned shift = 0; shift < 64; shift += 8)
            value |= std::uint64_t(std::to_integer<unsigned>(bytes_[position_++])) << shift;
        return value;
    }

    float finite_float() {
        const auto value = std::bit_cast<float>(u32());
        if (!std::isfinite(value)) throw std::runtime_error("non-finite replay segment float");
        return value;
    }

    std::string string() {
        const auto count = u32();
        require(count);
        std::string result(reinterpret_cast<const char*>(bytes_.data() + position_), count);
        position_ += count;
        return result;
    }

    std::span<const std::byte> take(std::size_t count) {
        require(count);
        auto result = bytes_.subspan(position_, count);
        position_ += count;
        return result;
    }

    std::size_t remaining() const { return bytes_.size() - position_; }

private:
    void require(std::size_t count) const {
        if (count > bytes_.size() - position_)
            throw std::runtime_error("truncated replay segment");
    }

    std::span<const std::byte> bytes_;
    std::size_t position_ = 0;
};

template <class T, class Read>
std::vector<T> read_array(Reader& reader, std::size_t element_bytes, Read read) {
    const auto count = reader.u32();
    if (count > reader.remaining() / element_bytes)
        throw std::runtime_error("invalid replay segment array length");
    std::vector<T> result;
    result.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) result.push_back(read());
    return result;
}

TrainingSample decode_sample(std::span<const std::byte> bytes,
                             const Compatibility& compatibility) {
    Reader reader(bytes);
    TrainingSample sample;
    sample.compatibility = compatibility;
    sample.canonical_player_ids = read_array<std::int32_t>(reader, 4, [&] {
        return std::bit_cast<std::int32_t>(reader.u32());
    });
    sample.node_features = read_array<float>(reader, 4, [&] { return reader.finite_float(); });
    sample.sparse_policy = read_array<std::pair<std::int32_t, float>>(reader, 8, [&] {
        return std::pair{std::bit_cast<std::int32_t>(reader.u32()), reader.finite_float()};
    });
    sample.value_target = read_array<float>(reader, 4, [&] { return reader.finite_float(); });
    if (reader.remaining() != 0) throw std::runtime_error("trailing replay sample bytes");
    return sample;
}

}  // namespace

std::vector<std::byte> encode_replay_segment(const Episode& episode,
                                             const Compatibility& compatibility) {
    if (!episode.completed) throw std::invalid_argument("cannot encode an aborted replay segment");
    if (episode.compatibility != compatibility)
        throw std::invalid_argument("replay episode compatibility mismatch");
    if (episode.samples.size() > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("replay segment has too many samples");

    std::vector<std::byte> metadata;
    append_string(metadata, episode.game_id);
    append_string(metadata, episode.retry_id);
    append_u64(metadata, episode.seed);
    append_u64(metadata, episode.move_count);

    std::vector<std::vector<std::byte>> records;
    records.reserve(episode.samples.size());
    std::uint64_t payload_size = 0;
    for (const auto& sample : episode.samples) {
        records.push_back(encode_sample(sample, compatibility));
        if (records.back().size() > std::numeric_limits<std::uint64_t>::max() - payload_size)
            throw std::invalid_argument("replay segment payload is too large");
        payload_size += records.back().size();
    }

    std::vector<std::byte> out;
    out.insert(out.end(), magic.begin(), magic.end());
    append_u32(out, format_version);
    append_u32(out, 0);
    append_u32(out, header_size);
    append_u32(out, static_cast<std::uint32_t>(records.size()));
    append_u64(out, metadata.size());
    append_u64(out, payload_size);
    const auto compatibility_hash = compatibility_digest(compatibility);
    out.insert(out.end(), compatibility_hash.begin(), compatibility_hash.end());
    out.insert(out.end(), metadata.begin(), metadata.end());
    std::uint64_t offset = 0;
    for (const auto& record : records) {
        append_u64(out, offset);
        append_u64(out, record.size());
        offset += record.size();
    }
    for (const auto& record : records) out.insert(out.end(), record.begin(), record.end());
    const auto footer = digest(out);
    out.insert(out.end(), footer.begin(), footer.end());
    return out;
}

Episode decode_replay_segment(std::span<const std::byte> bytes,
                              const Compatibility& compatibility) {
    if (bytes.size() < header_size + digest_size)
        throw std::runtime_error("truncated replay segment header");
    const auto body = bytes.first(bytes.size() - digest_size);
    const auto expected_footer = digest(body);
    if (!std::equal(expected_footer.begin(), expected_footer.end(), bytes.end() - digest_size))
        throw std::runtime_error("replay segment checksum mismatch");

    Reader reader(body);
    const auto actual_magic = reader.take(magic.size());
    if (!std::equal(magic.begin(), magic.end(), actual_magic.begin()))
        throw std::runtime_error("invalid replay segment magic");
    if (reader.u32() != format_version) throw std::runtime_error("unsupported replay segment version");
    if (reader.u32() != 0) throw std::runtime_error("unsupported replay segment flags");
    if (reader.u32() != header_size) throw std::runtime_error("invalid replay segment header size");
    const auto sample_count = reader.u32();
    const auto metadata_size = reader.u64();
    const auto payload_size = reader.u64();
    const auto recorded_compatibility = reader.take(digest_size);
    const auto expected_compatibility = compatibility_digest(compatibility);
    if (!std::equal(expected_compatibility.begin(), expected_compatibility.end(),
                    recorded_compatibility.begin()))
        throw std::runtime_error("replay segment compatibility mismatch");
    if (sample_count > reader.remaining() / 16)
        throw std::runtime_error("invalid replay segment sample count");
    if (metadata_size > reader.remaining()) throw std::runtime_error("invalid replay metadata length");
    const auto metadata_bytes = reader.take(static_cast<std::size_t>(metadata_size));
    if (sample_count > reader.remaining() / 16)
        throw std::runtime_error("truncated replay segment index");
    std::vector<std::pair<std::uint64_t, std::uint64_t>> index;
    index.reserve(sample_count);
    for (std::uint32_t i = 0; i < sample_count; ++i) {
        const auto offset = reader.u64();
        const auto length = reader.u64();
        index.emplace_back(offset, length);
    }
    if (payload_size != reader.remaining()) throw std::runtime_error("invalid replay payload length");
    const auto payload = reader.take(static_cast<std::size_t>(payload_size));

    Reader metadata(metadata_bytes);
    Episode episode;
    episode.game_id = metadata.string();
    episode.retry_id = metadata.string();
    episode.seed = metadata.u64();
    episode.move_count = metadata.u64();
    episode.completed = true;
    episode.compatibility = compatibility;
    if (metadata.remaining() != 0) throw std::runtime_error("trailing replay metadata bytes");
    episode.samples.reserve(sample_count);
    std::uint64_t expected_offset = 0;
    for (const auto& [offset, length] : index) {
        if (offset != expected_offset || length > payload.size() - std::min<std::uint64_t>(offset, payload.size()))
            throw std::runtime_error("invalid replay segment sample index");
        episode.samples.push_back(decode_sample(payload.subspan(static_cast<std::size_t>(offset),
                                                                 static_cast<std::size_t>(length)),
                                                compatibility));
        expected_offset += length;
    }
    if (expected_offset != payload.size()) throw std::runtime_error("incomplete replay segment index");
    return episode;
}

}  // namespace diamond_pipeline
