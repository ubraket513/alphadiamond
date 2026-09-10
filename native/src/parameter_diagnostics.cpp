#include "diamond_training/parameter_diagnostics.hpp"

#include <charconv>
#include <stdexcept>
#include <string>

namespace diamond_training {

ParameterGroup classify_parameter(std::string_view name, int64_t residual_blocks) {
    if (residual_blocks <= 0)
        throw std::invalid_argument("residual block count must be positive");
    if (name.starts_with("input_projection."))
        return ParameterGroup::input_projection;
    if (name.starts_with("output_norm."))
        return ParameterGroup::output_norm;
    if (name.starts_with("policy_source."))
        return ParameterGroup::policy_source;
    if (name.starts_with("policy_destination."))
        return ParameterGroup::policy_destination;
    if (name.starts_with("value_linear1."))
        return ParameterGroup::value_hidden;
    if (name.starts_with("value_linear2."))
        return ParameterGroup::value_output;
    if (name.starts_with("block_")) {
        const auto dot = name.find('.');
        if (dot != std::string_view::npos) {
            int64_t index = -1;
            const auto digits = name.substr(6, dot - 6);
            const auto parsed =
                std::from_chars(digits.data(), digits.data() + digits.size(), index);
            if (parsed.ec == std::errc{} && parsed.ptr == digits.data() + digits.size() &&
                index >= 0 && index < residual_blocks)
                return index == residual_blocks - 1 ? ParameterGroup::last_residual_block
                                                    : ParameterGroup::residual_trunk;
        }
    }
    throw std::invalid_argument("unknown model parameter: " + std::string(name));
}

const char* parameter_group_name(ParameterGroup group) {
    switch (group) {
    case ParameterGroup::input_projection:
        return "input_projection";
    case ParameterGroup::residual_trunk:
        return "residual_trunk";
    case ParameterGroup::last_residual_block:
        return "last_residual_block";
    case ParameterGroup::output_norm:
        return "output_norm";
    case ParameterGroup::policy_source:
        return "policy_source";
    case ParameterGroup::policy_destination:
        return "policy_destination";
    case ParameterGroup::value_hidden:
        return "value_hidden";
    case ParameterGroup::value_output:
        return "value_output";
    }
    throw std::invalid_argument("unknown parameter group");
}

} // namespace diamond_training
