#pragma once

#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#include <torch/types.h>

namespace diamond_training {

class DeviceResolutionError : public std::invalid_argument {
  public:
    using std::invalid_argument::invalid_argument;
};

struct DeviceRequest final {
    std::string requested_name;
    std::optional<int> cuda_index;
};

struct ResolvedDevice final {
    torch::Device torch_device;
    std::string requested_name;
    std::string canonical_name;
    std::optional<int> cuda_index;
};

DeviceRequest parse_device_request(std::string_view requested);
ResolvedDevice resolve_device(std::string_view requested);

}  // namespace diamond_training
