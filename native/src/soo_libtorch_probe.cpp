#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include <torch/script.h>

namespace {

std::vector<float> read_f32(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open " + path.string());
    file.seekg(0, std::ios::end);
    const auto size = file.tellg();
    if (size < 0 || size % static_cast<std::streamoff>(sizeof(float)) != 0) {
        throw std::runtime_error("invalid float32 file size: " + path.string());
    }
    file.seekg(0, std::ios::beg);
    std::vector<float> values(static_cast<size_t>(size) / sizeof(float));
    file.read(reinterpret_cast<char*>(values.data()), size);
    return values;
}

std::string read_text(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot open " + path.string());
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

void require_metadata(const std::string& metadata, const char* key, const char* value) {
    const std::string expected = std::string("\"") + key + "\": " + value;
    if (metadata.find(expected) == std::string::npos) {
        throw std::runtime_error("metadata mismatch or missing field: " + std::string(key));
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: soo_libtorch_probe <artifact-directory>\n";
        return 2;
    }

    try {
        const std::filesystem::path root(argv[1]);
        const auto metadata = read_text(root / "metadata.json");
        require_metadata(metadata, "format_version", "1");
        require_metadata(metadata, "model_name", "\"Soo\"");
        require_metadata(metadata, "dtype", "\"float32\"");
        require_metadata(metadata, "width", "128");
        require_metadata(metadata, "residual_blocks", "6");

        constexpr int64_t batch = 2;
        constexpr int64_t board = 73;
        constexpr int64_t features = 4;
        constexpr int64_t policy_size = board * board;

        auto input_data = read_f32(root / "inputs.f32");
        auto expected_policy = read_f32(root / "expected_policy.f32");
        auto expected_value = read_f32(root / "expected_value.f32");
        if (input_data.size() != batch * board * features ||
            expected_policy.size() != batch * policy_size || expected_value.size() != batch) {
            throw std::runtime_error("parity corpus has an unexpected shape");
        }

        auto input = torch::from_blob(input_data.data(), {batch, board, features}, torch::kFloat32)
                         .clone();
        auto module = torch::jit::load((root / "model.ts").string(), torch::kCPU);
        module.eval();
        const auto output = module.forward({input}).toTuple();
        const auto policy = output->elements()[0].toTensor().contiguous().view(-1);
        const auto value = output->elements()[1].toTensor().contiguous().view(-1);
        const auto policy_data = policy.data_ptr<float>();
        const auto value_data = value.data_ptr<float>();

        constexpr float tolerance = 1e-5f;
        float max_policy_error = 0.0f;
        float max_value_error = 0.0f;
        for (size_t i = 0; i < expected_policy.size(); ++i) {
            max_policy_error = std::max(max_policy_error, std::fabs(policy_data[i] - expected_policy[i]));
        }
        for (size_t i = 0; i < expected_value.size(); ++i) {
            max_value_error = std::max(max_value_error, std::fabs(value_data[i] - expected_value[i]));
        }
        if (max_policy_error > tolerance || max_value_error > tolerance) {
            std::cerr << "parity failure: policy=" << max_policy_error
                      << " value=" << max_value_error << "\n";
            return 1;
        }
        std::cout << "Soo LibTorch parity passed; max_policy_error=" << max_policy_error
                  << " max_value_error=" << max_value_error << "\n";
        return 0;
    } catch (const c10::Error& error) {
        std::cerr << "LibTorch error: " << error.what() << "\n";
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "probe error: " << error.what() << "\n";
        return 1;
    }
}
