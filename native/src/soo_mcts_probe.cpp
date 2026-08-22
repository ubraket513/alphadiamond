#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "diamond_model/soo_evaluator.hpp"
#include "soo/board.hpp"
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

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    try {
        const std::filesystem::path root(argv[1]);
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

        diamond_model::SooModel model(128, 6);
        model->load_weights(root / "weights");
        diamond_model::SooEvaluator evaluator(model);
        soo::MCTSConfig config;
        config.simulations = 2;
        config.seed = 17;
        soo::MCTS2P search(match, evaluator, config);
        const auto result = search.run(state, 0.0, true);
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
        std::cout << "Soo native MCTS evaluator integration passed; root_actions="
                  << result.root_actions.size() << ", evaluator_calls="
                  << result.evaluator_calls << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "probe error: " << error.what() << "\n";
        return 1;
    }
}
