#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "diamond_model/soo_evaluator.hpp"

namespace {

std::vector<float> read_f32(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open " + path.string());
    file.seekg(0, std::ios::end);
    const auto bytes = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<float> values(static_cast<size_t>(bytes) / sizeof(float));
    file.read(reinterpret_cast<char*>(values.data()), bytes);
    return values;
}

std::vector<int32_t> read_i32(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open " + path.string());
    file.seekg(0, std::ios::end);
    const auto bytes = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<int32_t> values(static_cast<size_t>(bytes) / sizeof(int32_t));
    file.read(reinterpret_cast<char*>(values.data()), bytes);
    return values;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    try {
        const std::filesystem::path root(argv[1]);
        const auto input = read_f32(root / "inputs.f32");
        const auto expected = read_f32(root / "expected_legal_priors.f32");
        const auto actions = read_i32(root / "legal_actions.i32");
        soo::Encoded encoded;
        encoded.node_features.assign(input.begin(), input.begin() + 73 * 4);
        encoded.feature_count = 4;

        diamond_model::SooModel model(128, 6);
        model->load_weights(root / "weights");
        diamond_model::SooEvaluator evaluator(model);
        const auto result = evaluator.evaluate(encoded, actions);
        if (result.priors.size() != actions.size()) throw std::runtime_error("prior count mismatch");
        float max_error = 0.0f;
        for (size_t i = 0; i < result.priors.size(); ++i) {
            max_error = std::max(max_error, std::fabs(static_cast<float>(result.priors[i]) - expected[i]));
        }
        if (max_error > 1e-5f) {
            std::cerr << "legal prior parity failure: " << max_error << "\n";
            return 1;
        }
        std::cout << "Soo legal-prior parity passed; max_error=" << max_error << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "probe error: " << error.what() << "\n";
        return 1;
    }
}
