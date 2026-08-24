#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace diamond_support {

struct JsonValue {
    using Array = std::vector<JsonValue>;
    using Object = std::map<std::string, JsonValue>;
    std::variant<std::nullptr_t, bool, int64_t, double, std::string, Array, Object> value;
};

// Strict JSON parser: duplicate keys and trailing data are errors. Integers
// remain int64_t, while decimal/exponent numbers are finite doubles.
JsonValue parse_json(std::string_view text);

}  // namespace diamond_support
