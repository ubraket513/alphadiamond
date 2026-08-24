#include <fstream>
#include <iostream>
#include <string>

#include "diamond_support/json.hpp"

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << "usage: alphadiamond-pipeline <request-v2.json> <result-v2.json>\n";
        return 0;
    }
    if (argc != 3) { std::cerr << "expected request and result JSON paths\n"; return 2; }
    try {
        std::ifstream input(argv[1], std::ios::binary);
        std::string request((std::istreambuf_iterator<char>(input)), {});
        const auto parsed = diamond_support::parse_json(request);
        const auto* object = std::get_if<diamond_support::JsonValue::Object>(&parsed.value);
        if (!object || !object->contains("schema_version") || !object->contains("operation_id"))
            throw std::invalid_argument("pipeline request requires schema_version and operation_id");
        const auto* version = std::get_if<int64_t>(&object->at("schema_version").value);
        const auto* operation = std::get_if<std::string>(&object->at("operation_id").value);
        if (!version || *version != 2 || !operation || operation->empty())
            throw std::invalid_argument("pipeline request must be schema v2 with operation_id");
        diamond_support::JsonValue result{diamond_support::JsonValue::Object{
            {"operation_id", diamond_support::JsonValue{*operation}},
            {"schema_version", diamond_support::JsonValue{int64_t{2}}},
            {"status", diamond_support::JsonValue{std::string("accepted")}},
        }};
        std::ofstream output(argv[2], std::ios::binary | std::ios::trunc);
        output << diamond_support::canonical_json(result) << '\n';
        return output ? 0 : 5;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 2; }
}
