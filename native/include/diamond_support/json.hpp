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

// Canonical JSON matches the frozen Python v1 store: sorted object keys,
// compact separators, ASCII-only strings, and finite numeric values.
std::string canonical_json(const JsonValue& value);
std::string sha256(std::string_view bytes);

}  // namespace diamond_support
