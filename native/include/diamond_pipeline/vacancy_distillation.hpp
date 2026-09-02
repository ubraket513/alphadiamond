#pragma once

#include <cstdint>
#include <vector>

#include "diamond_training/training_sample.hpp"

namespace diamond_pipeline {

struct VacancyTarget {
    std::vector<int32_t> actions;
    std::vector<double> probabilities;
    std::vector<double> progress;
};

VacancyTarget vacancy_target(const diamond_training::TrainingSample& sample);

} // namespace diamond_pipeline
