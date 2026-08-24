#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace diamond_training {

struct NetworkConfig {
    int64_t residual_blocks = 0;
    int64_t width = 0;

    bool operator==(const NetworkConfig&) const = default;
};

// This is the compatibility gate persisted by the Python v1 replay format.
// Keep the members typed: turning nested JSON or numeric fields into strings
// changes the canonical bytes, and therefore the namespace and chunk hashes.
struct Compatibility {
    std::string model_name;
    std::string model_version;
    int64_t player_count = 0;
    std::string ruleset_version;
    std::string board_topology_version;
    std::string ruleset_fingerprint;
    std::string encoder_version;
    std::string action_space_version;
    std::string seat_layout_version;
    std::string value_semantics_version;
    NetworkConfig network_config;

    static Compatibility soo(std::string version, NetworkConfig network);
    static Compatibility min(std::string version, NetworkConfig network);
    std::string family() const;
    void validate() const;

    bool operator==(const Compatibility&) const = default;
};

struct TrainingSample {
    Compatibility compatibility;
    std::vector<float> node_features;
    std::vector<int32_t> canonical_player_ids;
    std::vector<std::pair<int32_t, float>> sparse_policy;
    std::vector<float> value_target;
};

}  // namespace diamond_training
