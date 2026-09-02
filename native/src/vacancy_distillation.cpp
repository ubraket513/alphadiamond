#include "diamond_pipeline/vacancy_distillation.hpp"

#include <cmath>
#include <stdexcept>
#include <unordered_set>

#include "soo/action.hpp"
#include "soo/board.hpp"
#include "soo/prior.hpp"

namespace diamond_pipeline {

VacancyTarget vacancy_target(const diamond_training::TrainingSample& sample) {
    if (sample.compatibility.model_name != "Min" || sample.compatibility.player_count != 3)
        throw std::invalid_argument("vacancy distillation requires a Min sample");
    constexpr std::size_t feature_width = 6;
    if (sample.node_features.size() != soo::kBoardSize * feature_width)
        throw std::invalid_argument("vacancy distillation requires Min feature width 6");
    if (sample.sparse_policy.empty())
        throw std::invalid_argument("vacancy distillation requires legal actions");

    soo::PieceSet occupied;
    for (std::size_t position = 0; position < soo::kBoardSize; ++position) {
        const float value = sample.node_features[position * feature_width];
        if (value != 0.0F && value != 1.0F)
            throw std::invalid_argument("canonical self occupancy must be binary");
        if (value == 1.0F)
            occupied.set(position);
    }

    VacancyTarget target;
    target.actions.reserve(sample.sparse_policy.size());
    std::unordered_set<int32_t> unique;
    unique.reserve(sample.sparse_policy.size());
    for (const auto& [action, probability] : sample.sparse_policy) {
        (void)probability;
        if (action < 0 || action >= soo::kActionSize)
            throw std::invalid_argument("vacancy distillation action is out of range");
        if (!unique.insert(action).second)
            throw std::invalid_argument("vacancy distillation action is duplicated");
        target.actions.push_back(action);
    }

    soo::vacancy_prior(target.actions, occupied, target.probabilities);
    const auto& camp = soo::target_camp_set();
    const double before = soo::vacancy_potential(occupied, camp);
    target.progress.reserve(target.actions.size());
    for (const int32_t action : target.actions) {
        int source = 0;
        int destination = 0;
        soo::decode_action(action, source, destination);
        auto moved = occupied;
        moved.reset(source);
        moved.set(destination);
        target.progress.push_back(before - soo::vacancy_potential(moved, camp));
    }
    return target;
}

} // namespace diamond_pipeline
