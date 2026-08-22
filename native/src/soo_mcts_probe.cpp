#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

#include "diamond_model/soo_evaluator.hpp"
#include "diamond_model/deployment_artifact.hpp"
#include "soo/board.hpp"
#include "soo/encoder.hpp"
#include "soo/mcts.hpp"
#include "soo/rules.hpp"

namespace {

template <typename T>
std::vector<T> read_values(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open " + path.string());
    file.seekg(0, std::ios::end);
    const auto bytes = file.tellg();
    file.seekg(0, std::ios::beg);
    if (bytes < 0 || bytes % static_cast<std::streamoff>(sizeof(T)) != 0) {
        throw std::runtime_error("invalid binary fixture size: " + path.string());
    }
    std::vector<T> values(static_cast<size_t>(bytes) / sizeof(T));
    file.read(reinterpret_cast<char*>(values.data()), bytes);
    return values;
}

void configure_topology(const std::filesystem::path& root) {
    auto& topo = soo::mutable_topology();
    const auto neighbour = read_values<int8_t>(root / "topology_neighbour.i8");
    const auto camps = read_values<int32_t>(root / "topology_camp_positions.i32");
    const auto pairwise = read_values<int32_t>(root / "topology_pairwise_distance.i32");
    const auto physical = read_values<int32_t>(root / "topology_physical_to_canonical.i32");
    const auto canonical = read_values<int32_t>(root / "topology_canonical_to_physical.i32");
    if (neighbour.size() != 73 * 6 || camps.size() != 6 * 10 || pairwise.size() != 73 * 73 ||
        physical.size() != 6 * 73 || canonical.size() != 6 * 73) {
        throw std::runtime_error("invalid topology fixture dimensions");
    }
    for (int position = 0; position < 73; ++position)
        for (int direction = 0; direction < 6; ++direction)
            topo.neighbour[position][direction] = neighbour[position * 6 + direction];
    for (int camp = 0; camp < 6; ++camp)
        for (int offset = 0; offset < 10; ++offset)
            topo.camp_positions[camp][offset] = static_cast<uint8_t>(camps[camp * 10 + offset]);
    for (int row = 0; row < 73; ++row)
        for (int column = 0; column < 73; ++column)
            topo.pairwise[row][column] = static_cast<uint8_t>(pairwise[row * 73 + column]);
    for (int camp = 0; camp < 6; ++camp)
        for (int position = 0; position < 73; ++position) {
            topo.physical_to_canonical[camp][position] =
                static_cast<uint8_t>(physical[camp * 73 + position]);
            topo.canonical_to_physical[camp][position] =
                static_cast<uint8_t>(canonical[camp * 73 + position]);
        }
    topo.configured = true;
}

double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const auto index = static_cast<size_t>(
        std::ceil(fraction * static_cast<double>(values.size()))) - 1;
    return values.at(std::min(index, values.size() - 1));
}

double process_cpu_seconds() {
#ifdef _WIN32
    FILETIME created{}, exited{}, kernel{}, user{};
    if (!GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user)) return 0.0;
    ULARGE_INTEGER kernel_ticks{}, user_ticks{};
    kernel_ticks.LowPart = kernel.dwLowDateTime;
    kernel_ticks.HighPart = kernel.dwHighDateTime;
    user_ticks.LowPart = user.dwLowDateTime;
    user_ticks.HighPart = user.dwHighDateTime;
    return static_cast<double>(kernel_ticks.QuadPart + user_ticks.QuadPart) / 10'000'000.0;
#else
    return static_cast<double>(std::clock()) / CLOCKS_PER_SEC;
#endif
}

void validate_result(const soo::SearchResult& result,
                     const std::vector<int32_t>& expected_actions) {
    if (result.root_actions != expected_actions) {
        throw std::runtime_error("native MCTS root legal-action order differs from Python");
    }
    if (result.evaluator_calls == 0 || result.root_priors.size() != expected_actions.size()) {
        throw std::runtime_error("native MCTS did not receive root evaluator priors");
    }
    double prior_sum = 0.0;
    for (double prior : result.root_priors) prior_sum += prior;
    if (std::abs(prior_sum - 1.0) > 1e-6) {
        throw std::runtime_error("native MCTS root priors are not normalized");
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 5) return 2;
    try {
        const int simulations = argc >= 3 ? std::stoi(argv[2]) : 2;
        const int repeats = argc >= 4 ? std::stoi(argv[3]) : 1;
        const int threads = argc >= 5 ? std::stoi(argv[4]) : 1;
        if (simulations <= 0 || repeats <= 0 || threads <= 0)
            throw std::invalid_argument("simulations, repeats, and threads must be positive");
        torch::set_num_threads(threads);
        torch::set_num_interop_threads(1);
        const std::filesystem::path root(argv[1]);
        const auto artifact = diamond_model::validate_soo_deployment_artifact(root);
        configure_topology(root);
        const auto occupancy = read_values<uint8_t>(root / "mcts_occupancy.u8");
        const auto players = read_values<int32_t>(root / "mcts_players.i32");
        const auto expected_actions = read_values<int32_t>(root / "mcts_legal_actions.i32");
        const auto current = read_values<uint8_t>(root / "mcts_current_player.u8");
        if (occupancy.size() != 73 || players.size() != 6 || current.size() != 1) {
            throw std::runtime_error("invalid MCTS fixture dimensions");
        }

        soo::Match match;
        match.count = 2;
        for (int seat = 0; seat < 2; ++seat) {
            match.players[seat] = soo::PlayerSpec{
                static_cast<uint8_t>(players[seat * 3]),
                static_cast<uint8_t>(players[seat * 3 + 1]),
                static_cast<uint8_t>(players[seat * 3 + 2])};
        }
        soo::State state;
        std::copy(occupancy.begin(), occupancy.end(), state.occupancy.begin());
        state.current_player = current[0];

        diamond_model::SooModel model(artifact.width, artifact.residual_blocks);
        model->load_weights(artifact.weights);
        diamond_model::SooEvaluator evaluator(model);

        const auto encoded = soo::encode(state, match);
        torch::NoGradGuard no_grad;
        auto features = torch::from_blob(
            const_cast<float*>(encoded.node_features.data()), {1, 73, 4}, torch::kFloat32).clone();
        model->eval();
        (void)model->forward(features);
        constexpr int kInferenceSamples = 50;
        std::vector<double> raw_forward_ms;
        raw_forward_ms.reserve(kInferenceSamples);
        for (int iteration = 0; iteration < kInferenceSamples; ++iteration) {
            const auto started = std::chrono::steady_clock::now();
            (void)model->forward(features);
            const auto elapsed = std::chrono::steady_clock::now() - started;
            raw_forward_ms.push_back(std::chrono::duration<double, std::milli>(elapsed).count());
        }

        (void)evaluator.evaluate(encoded, expected_actions);
        std::vector<double> evaluator_ms;
        evaluator_ms.reserve(kInferenceSamples);
        for (int iteration = 0; iteration < kInferenceSamples; ++iteration) {
            const auto started = std::chrono::steady_clock::now();
            (void)evaluator.evaluate(encoded, expected_actions);
            const auto elapsed = std::chrono::steady_clock::now() - started;
            evaluator_ms.push_back(std::chrono::duration<double, std::milli>(elapsed).count());
        }

        soo::MCTSConfig config;
        config.simulations = simulations;
        config.seed = 17;
        soo::MCTS2P correctness_search(match, evaluator, config);
        soo::SearchResult result = correctness_search.run(state, 0.0, true);
        validate_result(result, expected_actions);

        std::vector<double> mcts_ms;
        mcts_ms.reserve(static_cast<size_t>(repeats));
        const double cpu_started = process_cpu_seconds();
        const auto wall_started = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < repeats; ++iteration) {
            soo::MCTS2P search(match, evaluator, config);
            const auto started = std::chrono::steady_clock::now();
            result = search.run(state, 0.0, false);
            const auto elapsed = std::chrono::steady_clock::now() - started;
            mcts_ms.push_back(std::chrono::duration<double, std::milli>(elapsed).count());
            validate_result(result, expected_actions);
        }
        const double wall_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - wall_started).count();
        const double cpu_seconds = process_cpu_seconds() - cpu_started;
        const unsigned logical_cpus = std::max(1U, std::thread::hardware_concurrency());
        const double normalized_cpu = wall_seconds > 0.0
            ? 100.0 * cpu_seconds / wall_seconds / static_cast<double>(logical_cpus)
            : 0.0;

        // The GUI currently validates the artifact and constructs/loads the native
        // model for each proposal. Measure that complete warm-cache path separately.
        std::vector<double> gui_proposal_ms;
        gui_proposal_ms.reserve(static_cast<size_t>(repeats));
        for (int iteration = 0; iteration < repeats; ++iteration) {
            const auto started = std::chrono::steady_clock::now();
            const auto runtime_artifact =
                diamond_model::validate_soo_deployment_artifact(root);
            diamond_model::SooModel runtime_model(
                runtime_artifact.width, runtime_artifact.residual_blocks);
            runtime_model->load_weights(runtime_artifact.weights);
            diamond_model::SooEvaluator runtime_evaluator(runtime_model);
            soo::MCTS2P runtime_search(match, runtime_evaluator, config);
            const auto runtime_result = runtime_search.run(state, 0.0, false);
            validate_result(runtime_result, expected_actions);
            const auto elapsed = std::chrono::steady_clock::now() - started;
            gui_proposal_ms.push_back(
                std::chrono::duration<double, std::milli>(elapsed).count());
        }
        std::cout << "Soo native MCTS evaluator integration passed; root_actions="
                  << result.root_actions.size() << ", evaluator_calls="
                  << result.evaluator_calls << ", simulations=" << simulations
                  << ", repeats=" << repeats << ", threads=" << threads
                  << ", raw_forward_p50_ms=" << percentile(raw_forward_ms, 0.50)
                  << ", raw_forward_p95_ms=" << percentile(raw_forward_ms, 0.95)
                  << ", evaluator_p50_ms=" << percentile(evaluator_ms, 0.50)
                  << ", evaluator_p95_ms=" << percentile(evaluator_ms, 0.95)
                  << ", mcts_p50_ms=" << percentile(mcts_ms, 0.50)
                  << ", mcts_p95_ms=" << percentile(mcts_ms, 0.95)
                  << ", gui_proposal_p50_ms=" << percentile(gui_proposal_ms, 0.50)
                  << ", gui_proposal_p95_ms=" << percentile(gui_proposal_ms, 0.95)
                  << ", mcts_normalized_cpu_percent=" << normalized_cpu << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "probe error: " << error.what() << "\n";
        return 1;
    }
}
