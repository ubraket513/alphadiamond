#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include "diamond_release/promotion.hpp"
#include "diamond_training/checkpoint.hpp"

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("expected scratch directory");
        const std::filesystem::path root = argv[1];
        std::error_code ignored;
        std::filesystem::remove_all(root, ignored);
        auto compatibility = diamond_training::Compatibility::soo(
            "1.0.0", {.residual_blocks = 1, .width = 8});
        const auto device = diamond_training::resolve_device("cpu");
        diamond_training::Trainer trainer(diamond_model::DiamondModel(8, 1, 4, 1), compatibility,
                                          {.learning_rate = 1e-3, .weight_decay = 1e-4}, device);
        const auto checkpoint = diamond_training::save_checkpoint_v2(root, trainer);
        const auto archival = diamond_release::initialize_promotion(root, "soo");
        require(archival.state == diamond_release::PromotionState::archival,
                "release did not initialize as archival");
        require(archival.checkpoint_generation == checkpoint.generation.filename().string(),
                "release did not bind the active checkpoint generation");

        const auto candidate = diamond_release::promote_checkpoint(
            root, diamond_release::PromotionState::candidate);
        require(candidate.state == diamond_release::PromotionState::candidate,
                "promotion did not reach candidate");
        require(diamond_release::load_promotion_manifest(root).state ==
                    diamond_release::PromotionState::candidate,
                "candidate state was not persisted");

        (void)diamond_release::promote_checkpoint(
            root, diamond_release::PromotionState::candidate);
        std::filesystem::remove_all(root, ignored);
        std::cout << "promotion_test: ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "promotion_test: " << error.what() << '\n';
        return 1;
    }
}
