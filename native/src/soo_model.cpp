#include "diamond_model/soo_model.hpp"

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace diamond_model {
namespace {

torch::Tensor apply_gelu(const torch::Tensor& value) {
    return torch::gelu(value);
}

torch::Tensor read_tensor(const std::filesystem::path& path, torch::IntArrayRef shape) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open model tensor: " + path.string());
    file.seekg(0, std::ios::end);
    const auto bytes = file.tellg();
    if (bytes < 0 || bytes % static_cast<std::streamoff>(sizeof(float)) != 0) {
        throw std::runtime_error("invalid tensor byte length: " + path.string());
    }
    file.seekg(0, std::ios::beg);
    std::vector<float> values(static_cast<size_t>(bytes) / sizeof(float));
    file.read(reinterpret_cast<char*>(values.data()), bytes);
    int64_t expected = 1;
    for (const auto dimension : shape) expected *= dimension;
    if (static_cast<int64_t>(values.size()) != expected) {
        throw std::runtime_error("tensor shape mismatch: " + path.string());
    }
    return torch::from_blob(values.data(), shape, torch::TensorOptions().dtype(torch::kFloat32))
        .clone();
}

void copy_parameter(torch::Tensor parameter, const std::filesystem::path& path,
                    torch::IntArrayRef shape) {
    parameter.copy_(read_tensor(path, shape));
}

std::filesystem::path weight(const std::filesystem::path& root, const std::string& name) {
    return root / (name + ".f32");
}

}  // namespace

DirectionalResidualBlockImpl::DirectionalResidualBlockImpl(int64_t width)
    : self_projection(torch::nn::LinearOptions(width, width)),
      norm(torch::nn::LayerNormOptions({width})) {
    register_module("self_projection", self_projection);
    for (int direction = 0; direction < 6; ++direction) {
        auto projection = torch::nn::Linear(torch::nn::LinearOptions(width, width).bias(false));
        direction_projections.push_back(projection);
        register_module("direction_projection_" + std::to_string(direction), projection);
    }
    register_module("norm", norm);
}

torch::Tensor DirectionalResidualBlockImpl::forward(const torch::Tensor& nodes,
                                                    const torch::Tensor& adjacency) {
    const auto neighbours = torch::einsum("dij,bjw->bdiw", {adjacency, nodes});
    std::vector<torch::Tensor> weights;
    for (const auto& module : direction_projections) {
        weights.push_back(module->weight);
    }
    const auto message = self_projection->forward(nodes) +
                         torch::einsum("bdiw,dvw->biv", {neighbours, torch::stack(weights)});
    return nodes + apply_gelu(norm->forward(message));
}

DiamondModelImpl::DiamondModelImpl(int64_t width, int64_t residual_blocks,
                                   int64_t input_features, int64_t value_size)
    : width_(width),
      residual_blocks_(residual_blocks),
      input_features_(input_features),
      value_size_(value_size),
      input_projection(torch::nn::LinearOptions(input_features, width)),
      output_norm(torch::nn::LayerNormOptions({width})),
      policy_source(torch::nn::LinearOptions(width, width)),
      policy_destination(torch::nn::LinearOptions(width, width)),
      value_linear1(torch::nn::LinearOptions(width, width)),
      value_linear2(torch::nn::LinearOptions(width, value_size)) {
    if (width <= 0 || residual_blocks <= 0 || input_features <= 0 || value_size <= 0)
        throw std::invalid_argument("invalid model dimensions");
    register_module("input_projection", input_projection);
    for (int64_t index = 0; index < residual_blocks; ++index) {
        blocks.push_back(DirectionalResidualBlock(width));
        register_module("block_" + std::to_string(index), blocks.back());
    }
    register_module("output_norm", output_norm);
    register_module("policy_source", policy_source);
    register_module("policy_destination", policy_destination);
    register_module("value_linear1", value_linear1);
    register_module("value_linear2", value_linear2);
    adjacency = torch::zeros({6, 73, 73});
    register_buffer("adjacency", adjacency);
}

void DiamondModelImpl::set_adjacency(const torch::Tensor& value) {
    if (value.sizes() != torch::IntArrayRef({6, 73, 73})) {
        throw std::invalid_argument("adjacency must have shape [6,73,73]");
    }
    adjacency.copy_(value);
}

std::tuple<torch::Tensor, torch::Tensor> DiamondModelImpl::forward(const torch::Tensor& features) {
    if (features.sizes().size() != 3 || features.size(1) != 73 ||
        features.size(2) != input_features_) {
        throw std::invalid_argument("features must have shape [B,73,input_features]");
    }
    auto nodes = input_projection->forward(features);
    for (auto& module : blocks) {
        nodes = module->forward(nodes, adjacency);
    }
    nodes = output_norm->forward(nodes);
    const auto source = policy_source->forward(nodes);
    const auto destination = policy_destination->forward(nodes);
    const auto policy = torch::matmul(source, destination.transpose(-1, -2)) /
                        std::sqrt(static_cast<double>(width_));
    const auto value = torch::tanh(value_linear2->forward(torch::gelu(value_linear1->forward(nodes.mean(1)))));
    return {policy.flatten(1), value};
}

void DiamondModelImpl::load_weights(const std::filesystem::path& root) {
    torch::NoGradGuard no_grad;
    copy_parameter(input_projection->weight, weight(root, "trunk__input_projection__weight"),
                   {width_, input_features_});
    copy_parameter(input_projection->bias, weight(root, "trunk__input_projection__bias"), {width_});
    for (int64_t index = 0; index < residual_blocks_; ++index) {
        auto block = blocks.at(static_cast<size_t>(index));
        const auto prefix = "trunk__blocks__" + std::to_string(index) + "__";
        copy_parameter(block->self_projection->weight, weight(root, prefix + "self_projection__weight"),
                       {width_, width_});
        copy_parameter(block->self_projection->bias, weight(root, prefix + "self_projection__bias"),
                       {width_});
        for (int direction = 0; direction < 6; ++direction) {
            const auto projection = block->direction_projections.at(static_cast<size_t>(direction));
            copy_parameter(projection->weight,
                           weight(root, prefix + "direction_projections__" + std::to_string(direction) + "__weight"),
                           {width_, width_});
        }
        copy_parameter(block->norm->weight, weight(root, prefix + "norm__weight"), {width_});
        copy_parameter(block->norm->bias, weight(root, prefix + "norm__bias"), {width_});
    }
    copy_parameter(output_norm->weight, weight(root, "trunk__output_norm__weight"), {width_});
    copy_parameter(output_norm->bias, weight(root, "trunk__output_norm__bias"), {width_});
    copy_parameter(policy_source->weight, weight(root, "policy_head__source__weight"), {width_, width_});
    copy_parameter(policy_source->bias, weight(root, "policy_head__source__bias"), {width_});
    copy_parameter(policy_destination->weight, weight(root, "policy_head__destination__weight"), {width_, width_});
    copy_parameter(policy_destination->bias, weight(root, "policy_head__destination__bias"), {width_});
    copy_parameter(value_linear1->weight,
                   weight(root, "value_head__0__weight"), {width_, width_});
    copy_parameter(value_linear1->bias,
                   weight(root, "value_head__0__bias"), {width_});
    copy_parameter(value_linear2->weight,
                   weight(root, "value_head__2__weight"), {value_size_, width_});
    copy_parameter(value_linear2->bias,
                   weight(root, "value_head__2__bias"), {value_size_});
    set_adjacency(read_tensor(root / "trunk__adjacency.f32", {6, 73, 73}));
}

}  // namespace diamond_model
