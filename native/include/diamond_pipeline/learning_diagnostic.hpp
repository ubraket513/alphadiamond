#pragma once
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>
#include "diamond_pipeline/replay_store.hpp"
#include "diamond_training/parameter_diagnostics.hpp"
#include "diamond_training/trainer.hpp"
namespace diamond_pipeline {
struct LearningDiagnosticConfig {
    uint64_t iteration = 0;
    std::size_t steps = 0;
    std::size_t batch_size = 0;
    std::size_t evaluation_samples = 0;
    std::size_t evaluation_batch = 0;
    std::size_t log_every = 0;
    uint64_t seed = 0;
};
struct HeldOutMetrics {
    double target_entropy = 0.0;
    double full_cross_entropy = 0.0;
    double full_kl = 0.0;
    double top1_agreement = 0.0;
    double value_mse = 0.0;
};
struct LearningStepDiagnostic {
    std::size_t local_step = 0;
    diamond_training::TrainingMetrics losses;
    std::map<diamond_training::ParameterGroup, diamond_training::GroupNorms> groups;
};
struct LearningDrift {
    double policy_kl = 0.0;
    double logit_rms_delta = 0.0;
    double value_rms_delta = 0.0;
    double top1_agreement_delta = 0.0;
    double full_cross_entropy_delta = 0.0;
    double value_mse_delta = 0.0;
};
struct LearningDiagnosticResult {
    HeldOutMetrics initial;
    HeldOutMetrics final;
    LearningDrift drift;
    std::vector<LearningStepDiagnostic> steps;
};
LearningDiagnosticResult run_min_learning_diagnostic(diamond_training::Trainer& trainer,
                                                     const ReplayStore& replay,
                                                     const LearningDiagnosticConfig& config);
} // namespace diamond_pipeline
