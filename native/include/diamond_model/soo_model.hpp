#pragma once

#include <filesystem>
#include <vector>

#include <torch/torch.h>

namespace diamond_model {

class DirectionalResidualBlockImpl : public torch::nn::Module {
  public:
    explicit DirectionalResidualBlockImpl(int64_t width);
    // Takes the neighbour gather index and the "no neighbour here" mask rather
    // than the dense adjacency. See DiamondModelImpl::set_adjacency.
    torch::Tensor forward(const torch::Tensor& nodes, const torch::Tensor& neighbour_index,
                          const torch::Tensor& neighbour_missing);

    torch::nn::Linear self_projection{nullptr};
    std::vector<torch::nn::Linear> direction_projections;
    torch::nn::LayerNorm norm{nullptr};
};
TORCH_MODULE(DirectionalResidualBlock);

// One graph trunk, two model families. Soo and Min differ only in how many
// features a hole carries and how many values the head predicts -- Min is the
// three-player model, so it predicts one value per seat. Both numbers come
// from the deployment artifact rather than from constants here.
class DiamondModelImpl : public torch::nn::Module {
  public:
    explicit DiamondModelImpl(int64_t width = 128, int64_t residual_blocks = 6,
                              int64_t input_features = 4, int64_t value_size = 1);

    std::tuple<torch::Tensor, torch::Tensor> forward(const torch::Tensor& features);
    void set_adjacency(const torch::Tensor& adjacency);
    void load_weights(const std::filesystem::path& weights_dir);

    int64_t width() const { return width_; }
    int64_t residual_blocks() const { return residual_blocks_; }
    int64_t input_features() const { return input_features_; }
    int64_t value_size() const { return value_size_; }

    torch::nn::Linear input_projection{nullptr};
    std::vector<DirectionalResidualBlock> blocks;
    torch::nn::LayerNorm output_norm{nullptr};
    torch::nn::Linear policy_source{nullptr};
    torch::nn::Linear policy_destination{nullptr};
    torch::nn::Linear value_linear1{nullptr};
    torch::nn::Linear value_linear2{nullptr};
    torch::Tensor adjacency;

  private:
    // Derived from `adjacency`, which is 0/1 with at most one neighbour per
    // (direction, hole) -- so the dense [6,73,73] contraction it was written as
    // is a gather wearing a matrix. Kept as a flat [6*73] index into the hole
    // dimension plus a [1,6,73,1] mask marking the board edges where a
    // direction runs off. `adjacency` itself stays: it is a shipped weight file
    // and the artifact format's business, not this optimisation's.
    torch::Tensor neighbour_index_;
    torch::Tensor neighbour_missing_;
    int64_t neighbour_tables_version_ = -1;
    void refresh_neighbour_tables();
    int64_t width_;
    int64_t residual_blocks_;
    int64_t input_features_;
    int64_t value_size_;
};
TORCH_MODULE(DiamondModel);

// The Soo spelling, kept so the existing probes and the Qt runtime read the
// same as before. Soo is the default configuration of the same model.
using SooModelImpl = DiamondModelImpl;
using SooModel = DiamondModel;

}  // namespace diamond_model
