#include <stdexcept>
#include <string>
#include "check.hpp"
#include "diamond_model/soo_model.hpp"
#include "diamond_training/parameter_diagnostics.hpp"
namespace {
template <typename Callable> bool throws_invalid_argument(Callable&& callable) {
    try {
        callable();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}
} // namespace
int main() {
    using diamond_training::classify_parameter;
    using diamond_training::ParameterGroup;
    auto model = diamond_model::DiamondModel(8, 2, 6, 3);
    for (const auto& parameter : model->named_parameters()) {
        const std::string name = parameter.key();
        ParameterGroup expected;
        if (name.rfind("input_projection.", 0) == 0)
            expected = ParameterGroup::input_projection;
        else if (name.rfind("block_0.", 0) == 0)
            expected = ParameterGroup::residual_trunk;
        else if (name.rfind("block_1.", 0) == 0)
            expected = ParameterGroup::last_residual_block;
        else if (name.rfind("output_norm.", 0) == 0)
            expected = ParameterGroup::output_norm;
        else if (name.rfind("policy_source.", 0) == 0)
            expected = ParameterGroup::policy_source;
        else if (name.rfind("policy_destination.", 0) == 0)
            expected = ParameterGroup::policy_destination;
        else if (name.rfind("value_linear1.", 0) == 0)
            expected = ParameterGroup::value_hidden;
        else if (name.rfind("value_linear2.", 0) == 0)
            expected = ParameterGroup::value_output;
        else
            REQUIRE(false, ("unexpected model parameter: " + name).c_str());
        CHECK_EQ(classify_parameter(name, 2), expected);
    }
    CHECK(throws_invalid_argument([] { (void)classify_parameter("mystery.weight", 2); }));
    return soo_test::report("parameter_diagnostics_test");
}
