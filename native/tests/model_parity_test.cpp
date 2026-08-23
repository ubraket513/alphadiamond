// The native model must reproduce the Python model's own outputs.
//
// Every artifact ships a small deterministic corpus -- the inputs the exporter
// ran, and the policy and value PyTorch produced for them. This test loads the
// raw weights into the native model and compares. It is the only thing that
// proves the C++ inference path is the same function as the trained network,
// and it now runs for both families from the same code: Soo and Min differ
// only in the declared input features and value-head width.
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "check.hpp"
#include "diamond_model/deployment_artifact.hpp"
#include "diamond_model/soo_model.hpp"

namespace {

constexpr float kTolerance = 1e-5f;

std::vector<float> read_f32(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    REQUIRE(static_cast<bool>(file), path.string().c_str());
    file.seekg(0, std::ios::end);
    const auto bytes = file.tellg();
    file.seekg(0, std::ios::beg);
    REQUIRE(bytes > 0 && bytes % static_cast<std::streamoff>(sizeof(float)) == 0,
            "invalid float32 file");
    std::vector<float> values(static_cast<size_t>(bytes) / sizeof(float));
    file.read(reinterpret_cast<char*>(values.data()), bytes);
    return values;
}

float max_error(const float* actual, const std::vector<float>& expected) {
    float worst = 0.0f;
    for (size_t index = 0; index < expected.size(); ++index) {
        worst = std::max(worst, std::fabs(actual[index] - expected[index]));
    }
    return worst;
}

void check_family(const std::filesystem::path& root, const std::string& family) {
    const auto artifact = diamond_model::validate_deployment_artifact(root, family);

    auto input_data = read_f32(root / "inputs.f32");
    const auto expected_policy = read_f32(root / "expected_policy.f32");
    const auto expected_value = read_f32(root / "expected_value.f32");

    const int64_t batch =
        static_cast<int64_t>(input_data.size()) / (73 * artifact.input_features);
    REQUIRE(batch > 0, "artifact corpus is empty");
    auto input = torch::from_blob(input_data.data(), {batch, 73, artifact.input_features},
                                  torch::kFloat32)
                     .clone();

    diamond_model::DiamondModel model(artifact.width, artifact.residual_blocks,
                                      artifact.input_features, artifact.value_size);
    model->load_weights(artifact.weights);
    model->eval();

    torch::NoGradGuard no_grad;
    const auto [policy, value] = model->forward(input);
    const auto policy_flat = policy.contiguous().view(-1);
    const auto value_flat = value.contiguous().view(-1);

    CHECK_EQ(static_cast<size_t>(policy_flat.numel()), expected_policy.size());
    CHECK_EQ(static_cast<size_t>(value_flat.numel()), expected_value.size());

    const float policy_error = max_error(policy_flat.data_ptr<float>(), expected_policy);
    const float value_error = max_error(value_flat.data_ptr<float>(), expected_value);
    if (policy_error > kTolerance || value_error > kTolerance) {
        soo_test::fail(__FILE__, __LINE__,
                       family + ": native inference differs from PyTorch (policy=" +
                           std::to_string(policy_error) + " value=" +
                           std::to_string(value_error) + ")");
        return;
    }
    std::fprintf(stderr, "%s: policy_error=%g value_error=%g\n", family.c_str(),
                 static_cast<double>(policy_error), static_cast<double>(value_error));
}

}  // namespace

int main(int argc, char** argv) {
    REQUIRE(argc >= 3, "usage: model_parity_test <family> <artifact-dir> [<family> <dir>]...");
    for (int index = 1; index + 1 < argc; index += 2) {
        check_family(argv[index + 1], argv[index]);
    }
    return soo_test::report("model_parity_test");
}
