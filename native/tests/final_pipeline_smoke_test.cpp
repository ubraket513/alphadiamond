#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.hpp"
#include "diamond_model/deployment_artifact.hpp"
#include "diamond_model/soo_model.hpp"
#include "diamond_orchestration/rating.hpp"
#include "diamond_release/promotion.hpp"
#include "diamond_training/checkpoint.hpp"
#include "diamond_training/trainer.hpp"

namespace {

using diamond_training::Compatibility;
using diamond_training::NetworkConfig;

constexpr NetworkConfig kNetwork{.residual_blocks = 1, .width = 8};
constexpr diamond_training::TrainingConfig kTraining{
    .learning_rate = 1e-3,
    .weight_decay = 1e-4,
};

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

diamond_training::TrainingSample sample_for(const Compatibility& compatibility,
                                            int64_t input_features,
                                            int64_t value_size) {
    diamond_training::TrainingSample sample;
    sample.compatibility = compatibility;
    sample.node_features.assign(static_cast<size_t>(73 * input_features), 0.0F);
    sample.sparse_policy.emplace_back(0, 1.0F);
    sample.value_target.assign(static_cast<size_t>(value_size), 0.0F);
    return sample;
}

void train_save_and_resume(const std::filesystem::path& checkpoint_root,
                           const Compatibility& compatibility,
                           int64_t input_features, int64_t value_size) {
    const auto sample = sample_for(compatibility, input_features, value_size);
    const std::vector<diamond_training::TrainingSample> samples{sample};
    const auto device = diamond_training::resolve_device("cpu");
    diamond_training::Trainer trainer(
        diamond_model::DiamondModel(8, 1, input_features, value_size), compatibility,
        kTraining, device);
    require(trainer.train(samples).training_step == 1, "initial native training step failed");
    const auto saved = diamond_training::save_checkpoint_v2(checkpoint_root, trainer);
    require(saved.training_step == 1, "checkpoint did not record initial step");

    diamond_training::Trainer resumed(
        diamond_model::DiamondModel(8, 1, input_features, value_size), compatibility,
        kTraining, device);
    require(diamond_training::load_checkpoint_v2(checkpoint_root, resumed).training_step == 1,
            "checkpoint restore did not recover initial step");
    require(resumed.train(samples).training_step == 2, "resumed native training step failed");
    require(diamond_training::save_checkpoint_v2(checkpoint_root, resumed).training_step == 2,
            "checkpoint did not record resumed step");
}

void record_rating_events() {
    using namespace diamond_orchestration;

    RatingRegistry soo{"final-pipeline-soo-v1"};
    soo.add_participant("soo-candidate", "Soo candidate");
    soo.add_participant("soo-baseline", "Soo baseline");
    require(soo.record_event(make_soo_rating_event(
                0, "final-pipeline-soo-v1", {"soo-candidate", "soo-baseline"},
                {1, 2}, {1, 2}, "final-smoke-opening", true,
                "soo-candidate", "soo-baseline")),
            "Soo rating event was not recorded");
    require(soo.events().size() == 1, "Soo rating event count");

    RatingRegistry min{"final-pipeline-min-v1", TrueSkillConfig{}};
    min.add_participant("min-candidate", "Min candidate");
    min.add_participant("min-baseline-a", "Min baseline A");
    min.add_participant("min-baseline-b", "Min baseline B");
    require(min.record_event(make_min_rating_event(
                0, "final-pipeline-min-v1",
                {"min-candidate", "min-baseline-a", "min-baseline-b"},
                {1, 2, 3}, {1, 2, 3}, "final-smoke-opening", true,
                {"min-candidate", "min-baseline-a", "min-baseline-b"})),
            "Min rating event was not recorded");
    require(min.events().size() == 1, "Min rating event count");
}

void promote(const std::filesystem::path& checkpoint_root, const std::string& family,
             const std::filesystem::path& artifact) {
    require(diamond_release::initialize_promotion(checkpoint_root, family).state ==
                diamond_release::PromotionState::archival,
            "release initialization did not create archival manifest");
    require(diamond_release::promote_checkpoint(
                checkpoint_root, diamond_release::PromotionState::candidate).state ==
                diamond_release::PromotionState::candidate,
            "release did not promote checkpoint to candidate");
    require(diamond_release::promote_checkpoint(
                checkpoint_root, diamond_release::PromotionState::promoted, artifact).state ==
                diamond_release::PromotionState::promoted,
            "release did not promote checkpoint to promoted");
}

}  // namespace

int main(int argc, char** argv) {
    REQUIRE(argc == 4,
            "usage: final_pipeline_smoke_test <scratch> <soo-artifact> <min-artifact>");
    try {
        const std::filesystem::path scratch = argv[1];
        const auto soo_artifact = diamond_model::validate_deployment_artifact(argv[2], "soo");
        const auto min_artifact = diamond_model::validate_deployment_artifact(argv[3], "min");
        std::error_code ignored;
        std::filesystem::remove_all(scratch, ignored);

        torch::manual_seed(0);
        const auto soo_checkpoint = scratch / "soo" / "checkpoint-v2";
        const auto min_checkpoint = scratch / "min" / "checkpoint-v2";
        train_save_and_resume(soo_checkpoint, Compatibility::soo("1.0.0", kNetwork), 4, 1);
        train_save_and_resume(min_checkpoint, Compatibility::min("1.0.0", kNetwork), 6, 3);

        record_rating_events();
        promote(soo_checkpoint, "soo", soo_artifact.root);
        promote(min_checkpoint, "min", min_artifact.root);

        const auto staged_root = scratch / "runtime-models";
        const auto index = diamond_release::stage_release_models(
            staged_root, {soo_artifact.root, min_artifact.root}, {"soo", "min"});
        const auto* staged_soo = index.default_for("soo");
        const auto* staged_min = index.default_for("min");
        require(staged_soo && staged_soo->runtime_sha256 == soo_artifact.runtime_sha256,
                "staged Soo runtime artifact mismatch");
        require(staged_min && staged_min->runtime_sha256 == min_artifact.runtime_sha256,
                "staged Min runtime artifact mismatch");
        require(!std::filesystem::exists(staged_root / "soo" / soo_artifact.model_version / "model.ts"),
                "runtime staging copied the Soo development graph");
        require(!std::filesystem::exists(staged_root / "min" / min_artifact.model_version / "model.ts"),
                "runtime staging copied the Min development graph");

        std::filesystem::remove_all(scratch, ignored);
        return soo_test::report("final_pipeline_smoke_test");
    } catch (const std::exception& error) {
        soo_test::fail(__FILE__, __LINE__, error.what());
        return soo_test::report("final_pipeline_smoke_test");
    }
}
