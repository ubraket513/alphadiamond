#include "diamond_training/device.hpp"

#include <charconv>
#include <string>

#include <torch/cuda.h>

namespace diamond_training {
namespace {

constexpr std::string_view kCudaPrefix = "cuda:";

[[noreturn]] void invalid_device_name() {
    throw DeviceResolutionError("runtime.device must be cpu, cuda, or cuda:N");
}

} // namespace

DeviceRequest parse_device_request(std::string_view requested) {
    if (requested == "cpu")
        return {.requested_name = std::string(requested), .cuda_index = std::nullopt};
    if (requested == "cuda")
        return {.requested_name = std::string(requested), .cuda_index = 0};
    if (!requested.starts_with(kCudaPrefix))
        invalid_device_name();

    const auto index_text = requested.substr(kCudaPrefix.size());
    if (index_text.empty())
        invalid_device_name();
    for (const unsigned char character : index_text)
        if (character < '0' || character > '9')
            invalid_device_name();

    int index = 0;
    const auto [end, error] =
        std::from_chars(index_text.data(), index_text.data() + index_text.size(), index);
    if (error != std::errc{} || end != index_text.data() + index_text.size())
        invalid_device_name();
    return {.requested_name = std::string(requested), .cuda_index = index};
}

ResolvedDevice resolve_device(std::string_view requested) {
    const auto parsed = parse_device_request(requested);
    if (!parsed.cuda_index) {
        return {.torch_device = torch::Device(torch::kCPU),
                .requested_name = parsed.requested_name,
                .canonical_name = "cpu",
                .cuda_index = std::nullopt};
    }

    const int available = static_cast<int>(torch::cuda::device_count());
    if (!torch::cuda::is_available() || available <= 0) {
        throw DeviceResolutionError("runtime.device " + parsed.requested_name +
                                    " requires CUDA, but no CUDA device is available");
    }
    if (*parsed.cuda_index >= available) {
        throw DeviceResolutionError(
            "runtime.device " + parsed.requested_name +
            " is out of range; available CUDA devices: " + std::to_string(available));
    }
    return {.torch_device = torch::Device(torch::kCUDA, *parsed.cuda_index),
            .requested_name = parsed.requested_name,
            .canonical_name = "cuda:" + std::to_string(*parsed.cuda_index),
            .cuda_index = parsed.cuda_index};
}

} // namespace diamond_training
