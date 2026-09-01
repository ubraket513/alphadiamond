#pragma once
#include <cstdint>
#include <string_view>
namespace diamond_training {
enum class ParameterGroup {
    input_projection,
    residual_trunk,
    last_residual_block,
    output_norm,
    policy_source,
    policy_destination,
    value_hidden,
    value_output
};
ParameterGroup classify_parameter(std::string_view name, int64_t residual_blocks);
const char* parameter_group_name(ParameterGroup group);
struct GroupNorms {
    double parameter_l2 = 0.0;
    double gradient_l2 = 0.0;
    double update_l2 = 0.0;
    double relative_update = 0.0;
};
} // namespace diamond_training
