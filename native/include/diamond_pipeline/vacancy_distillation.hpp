#pragma once

#include <cstdint>
#include <cstddef>
#include <span>
#include <vector>

#include "diamond_training/trainer.hpp"
#include "diamond_training/training_sample.hpp"

namespace diamond_pipeline {

struct VacancyTarget {
    std::vector<int32_t> actions;
    std::vector<double> probabilities;
    std::vector<double> progress;
};

VacancyTarget vacancy_target(const diamond_training::TrainingSample& sample);

enum class DistillationArm {
    policy_head,
    trunk_policy,
};

struct VacancyDistillationConfig {
    DistillationArm arm = DistillationArm::policy_head;
    std::size_t steps = 0;
    std::size_t batch_size = 0;
    std::size_t evaluation_batch = 0;
};

struct VacancyMetrics {
    double legal_kl = 0.0;
    double expected_progress = 0.0;
    double teacher_expected_progress = 0.0;
    double expected_progress_ratio = 0.0;
    double top1_agreement = 0.0;
    double top3_teacher_mass = 0.0;
};

struct VacancyDistillationResult {
    VacancyMetrics initial;
    VacancyMetrics final;
    double policy_kl = 0.0;
};

VacancyDistillationResult
run_vacancy_distillation(diamond_training::Trainer& trainer,
                         std::span<const diamond_training::TrainingSample> training_samples,
                         std::span<const diamond_training::TrainingSample> held_out_samples,
                         const VacancyDistillationConfig& config);

} // namespace diamond_pipeline
