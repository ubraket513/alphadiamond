#pragma once

#include <filesystem>
#include <vector>

#include <torch/torch.h>

namespace diamond_model {

class DirectionalResidualBlockImpl : public torch::nn::Module {
  public:
    explicit DirectionalResidualBlockImpl(int64_t width);
    torch::Tensor forward(const torch::Tensor& nodes, const torch::Tensor& adjacency);

    torch::nn::Linear self_projection{nullptr};
    std::vector<torch::nn::Linear> direction_projections;
    torch::nn::LayerNorm norm{nullptr};
};
TORCH_MODULE(DirectionalResidualBlock);

class SooModelImpl : public torch::nn::Module {
  public:
    explicit SooModelImpl(int64_t width = 128, int64_t residual_blocks = 6);

    std::tuple<torch::Tensor, torch::Tensor> forward(const torch::Tensor& features);
    void set_adjacency(const torch::Tensor& adjacency);
    void load_weights(const std::filesystem::path& weights_dir);

    int64_t width() const { return width_; }
    int64_t residual_blocks() const { return residual_blocks_; }

    torch::nn::Linear input_projection{nullptr};
    std::vector<DirectionalResidualBlock> blocks;
    torch::nn::LayerNorm output_norm{nullptr};
    torch::nn::Linear policy_source{nullptr};
    torch::nn::Linear policy_destination{nullptr};
    torch::nn::Linear value_linear1{nullptr};
    torch::nn::Linear value_linear2{nullptr};
    torch::Tensor adjacency;

  private:
    int64_t width_;
    int64_t residual_blocks_;
};
TORCH_MODULE(SooModel);

}  // namespace diamond_model
