#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include <torch/torch.h>

#include "diamond_model/soo_model.hpp"
#include "diamond_model/deployment_artifact.hpp"

namespace {

std::vector<float> read_f32(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open " + path.string());
    file.seekg(0, std::ios::end);
    const auto bytes = file.tellg();
    file.seekg(0, std::ios::beg);
    if (bytes < 0 || bytes % static_cast<std::streamoff>(sizeof(float)) != 0) {
        throw std::runtime_error("invalid float32 file: " + path.string());
    }
    std::vector<float> values(static_cast<size_t>(bytes) / sizeof(float));
    file.read(reinterpret_cast<char*>(values.data()), bytes);
    return values;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: soo_native_model_probe <artifact-directory>\n";
        return 2;
    }
    try {
        const std::filesystem::path root(argv[1]);
        const auto artifact = diamond_model::validate_soo_deployment_artifact(root);
        auto input_data = read_f32(root / "inputs.f32");
        const auto expected_policy = read_f32(root / "expected_policy.f32");
        const auto expected_value = read_f32(root / "expected_value.f32");
        auto input = torch::from_blob(input_data.data(), {2, 73, 4}, torch::kFloat32).clone();

        diamond_model::SooModel model(artifact.width, artifact.residual_blocks);
        model->load_weights(artifact.weights);
        model->eval();
        const auto [policy, value] = model->forward(input);
        const auto policy_flat = policy.contiguous().view(-1);
        const auto value_flat = value.contiguous().view(-1);
        const auto* policy_data = policy_flat.data_ptr<float>();
        const auto* value_data = value_flat.data_ptr<float>();
        float max_policy_error = 0.0f;
        float max_value_error = 0.0f;
        for (size_t i = 0; i < expected_policy.size(); ++i) {
            max_policy_error = std::max(max_policy_error, std::fabs(policy_data[i] - expected_policy[i]));
        }
        for (size_t i = 0; i < expected_value.size(); ++i) {
            max_value_error = std::max(max_value_error, std::fabs(value_data[i] - expected_value[i]));
        }
        if (max_policy_error > 1e-5f || max_value_error > 1e-5f) {
            std::cerr << "native model parity failure: policy=" << max_policy_error
                      << " value=" << max_value_error << "\n";
            return 1;
        }
        std::cout << "Soo native model parity passed; max_policy_error=" << max_policy_error
                  << " max_value_error=" << max_value_error << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "probe error: " << error.what() << "\n";
        return 1;
    }
}
