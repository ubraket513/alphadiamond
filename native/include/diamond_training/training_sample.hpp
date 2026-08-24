#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace diamond_training {

struct Compatibility {
    std::string family;
    std::string model_version;
    // A legacy v1 replay persists the complete compatibility gate.  New native
    // callers may leave this empty; readers retain a parsed legacy gate here.
    std::map<std::string, std::string> metadata;

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
