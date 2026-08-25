#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "check.hpp"
#include "diamond_training/device.hpp"
#include <torch/cuda.h>

namespace {

using diamond_training::DeviceResolutionError;

bool rejects(std::string_view requested) {
    try {
        (void)diamond_training::parse_device_request(requested);
    } catch (const DeviceResolutionError&) {
        return true;
    }
    return false;
}

std::string quote(const std::filesystem::path& path) {
    const auto text = path.string();
    REQUIRE(text.find('"') == std::string::npos, "test path contains a quote");
    return '"' + text + '"';
}

void write_cuda_config(const std::filesystem::path& path) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    REQUIRE(static_cast<bool>(output), "cannot write CUDA test config");
    output << R"({"arena":{"games":2,"max_moves":10,"promotion_threshold":0.55,"seed":7},"inference":{"max_batch_size":1,"max_wait_us":1,"request_queue_capacity":1,"response_timeout_s":5.0},"mcts":{"c_puct":1.5,"dirichlet_alpha":0.3,"dirichlet_epsilon":0.25,"seed":7,"simulations":128},"model_name":"Soo","model_version":"2.0.0","network":{"residual_blocks":6,"width":128},"opening_suite":{"count":1,"id":"production-openings-v1","max_depth":0,"seed":7,"version":1},"promotion_statistics":{"bootstrap_replicates":10000,"confidence_level":0.95,"method":"opening-block-bootstrap-v1","resampling_unit":"opening_block","seed":7},"replay":{"capacity":128,"seed":7},"run_budget":{"checkpoint_every_iterations":1,"max_iterations":1,"max_wall_clock_seconds":null},"run_seed":7,"runtime":{"device":"cuda","precision":"fp32"},"schema_version":2,"self_play":{"bootstrap_prior":"none","max_game_seconds":null,"max_moves":10,"seed":7,"temperature":1.0,"temperature_moves":1},"training":{"batch_size":1,"learning_rate":0.001,"seed":7,"train_steps_per_iteration":1,"weight_decay":0.0},"workers":{"games_per_iteration":1,"logical_lanes":1,"retry_id":"attempt-0","search_threads":1}})";
}

}  // namespace

int main(int argc, char** argv) {
    REQUIRE(argc == 3 || (argc == 4 && std::string_view(argv[3]) == "--cuda"),
            "usage: device_test <scratch> <alphadiamond-train> [--cuda]");
    const auto scratch = std::filesystem::path(argv[1]);
    const auto training_executable = std::filesystem::path(argv[2]);
    std::filesystem::remove_all(scratch);
    std::filesystem::create_directories(scratch);

    if (argc == 4) {
        REQUIRE(torch::cuda::is_available(), "CUDA-labelled test requires an available CUDA device");
        const int available = static_cast<int>(torch::cuda::device_count());
        REQUIRE(available > 0, "CUDA-labelled test requires at least one CUDA device");

        const auto resolved = diamond_training::resolve_device("cuda");
        CHECK_EQ(resolved.requested_name, std::string("cuda"));
        CHECK_EQ(resolved.canonical_name, std::string("cuda:0"));
        CHECK_EQ(resolved.cuda_index, std::optional<int>{0});
        CHECK(resolved.torch_device.is_cuda());

        const auto last_name = "cuda:" + std::to_string(available - 1);
        const auto last = diamond_training::resolve_device(last_name);
        CHECK_EQ(last.requested_name, last_name);
        CHECK_EQ(last.canonical_name, last_name);
        CHECK_EQ(last.cuda_index, std::optional<int>{available - 1});

        const auto out_of_range = "cuda:" + std::to_string(available);
        try {
            (void)diamond_training::resolve_device(out_of_range);
            CHECK(false);
        } catch (const DeviceResolutionError& error) {
            CHECK_EQ(std::string(error.what()),
                     "runtime.device " + out_of_range +
                         " is out of range; available CUDA devices: " + std::to_string(available));
        }
        return soo_test::report("device_test_cuda");
    }

    const auto cpu = diamond_training::resolve_device("cpu");
    CHECK_EQ(cpu.requested_name, std::string("cpu"));
    CHECK_EQ(cpu.canonical_name, std::string("cpu"));
    CHECK(!cpu.cuda_index.has_value());
    CHECK(cpu.torch_device.is_cpu());

    const auto cuda = diamond_training::parse_device_request("cuda:17");
    CHECK_EQ(cuda.requested_name, std::string("cuda:17"));
    CHECK_EQ(cuda.cuda_index, std::optional<int>{17});
    CHECK_EQ(diamond_training::parse_device_request("cuda").cuda_index, std::optional<int>{0});
    for (const auto* malformed : {"CPU", "cuda:", "cuda:-1", "cuda:1x", "gpu", "cuda:999999999999999999999"})
        CHECK(rejects(malformed));

    if (!torch::cuda::is_available()) {
        try {
            (void)diamond_training::resolve_device("cuda");
            CHECK(false);
        } catch (const DeviceResolutionError& error) {
            CHECK_EQ(std::string(error.what()), std::string("runtime.device cuda requires CUDA, but no CUDA device is available"));
        }

        const auto config = scratch / "cuda-config.json";
        const auto runtime_dir = scratch / "unavailable-cuda-run";
        const auto report = scratch / "unavailable-cuda-report.json";
        write_cuda_config(config);
        const std::string command = quote(training_executable) + " train --runtime-dir " + quote(runtime_dir) +
            " --model Soo --run-id unavailable-cuda --config " + quote(config) +
            " --checkpoint " + quote(scratch / "missing.pt") + " > " + quote(report) + " 2>&1";
#ifdef _WIN32
        const std::string shell_command = '"' + command + '"';
#else
        const std::string& shell_command = command;
#endif
        CHECK(std::system(shell_command.c_str()) != 0);
        CHECK(!std::filesystem::exists(runtime_dir));
    }

    return soo_test::report("device_test");
}
