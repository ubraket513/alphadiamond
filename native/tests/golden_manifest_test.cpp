#include "diamond_support/json.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using diamond_support::JsonValue;

std::string read_bytes(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read " + path.string());
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

const JsonValue& field(const JsonValue::Object& object, std::string_view name) {
    const auto found = object.find(std::string(name));
    if (found == object.end()) throw std::runtime_error("missing field: " + std::string(name));
    return found->second;
}

const std::string& string_field(const JsonValue::Object& object, std::string_view name) {
    const auto* value = std::get_if<std::string>(&field(object, name).value);
    if (!value) throw std::runtime_error("field must be a string: " + std::string(name));
    return *value;
}

int64_t integer_field(const JsonValue::Object& object, std::string_view name) {
    const auto* value = std::get_if<int64_t>(&field(object, name).value);
    if (!value) throw std::runtime_error("field must be an integer: " + std::string(name));
    return *value;
}

bool is_lower_hex(std::string_view value, std::size_t length) {
    return value.size() == length && std::ranges::all_of(value, [](const unsigned char byte) {
        return std::isdigit(byte) != 0 || (byte >= 'a' && byte <= 'f');
    });
}

bool safe_relative_path(const std::string& value) {
    const std::filesystem::path path(value);
    if (value.empty() || value.contains('\\') || path.is_absolute()) return false;
    return std::ranges::none_of(path, [](const auto& component) { return component == ".."; });
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: golden_manifest_test <golden-dir>");
        const std::filesystem::path root(argv[1]);
        const auto parsed = diamond_support::parse_json(read_bytes(root / "MANIFEST.json"));
        const auto* manifest = std::get_if<JsonValue::Object>(&parsed.value);
        if (!manifest || manifest->size() != 6) throw std::runtime_error("unexpected manifest fields");
        if (integer_field(*manifest, "golden_format_version") != 1 ||
            string_field(*manifest, "game_contract_version") != "diamond73-v1") {
            throw std::runtime_error("golden contract version mismatch");
        }

        const auto& oracle = field(*manifest, "oracle_commit").value;
        if (!std::holds_alternative<std::nullptr_t>(oracle)) {
            const auto* commit = std::get_if<std::string>(&oracle);
            if (!commit || !is_lower_hex(*commit, 40)) throw std::runtime_error("invalid oracle_commit");
        }

        const auto* payload = std::get_if<JsonValue::Object>(&field(*manifest, "payload_sha256").value);
        if (!payload || payload->empty()) throw std::runtime_error("payload_sha256 must be non-empty");
        std::set<std::string> declared;
        std::string corpus_input;
        for (const auto& [relative, digest_value] : *payload) {
            const auto* digest = std::get_if<std::string>(&digest_value.value);
            if (!safe_relative_path(relative) || !digest || !is_lower_hex(*digest, 64)) {
                throw std::runtime_error("invalid payload descriptor: " + relative);
            }
            const auto bytes = read_bytes(root / relative);
            if (diamond_support::sha256(bytes) != *digest) {
                throw std::runtime_error("payload digest mismatch: " + relative);
            }
            declared.insert(relative);
            corpus_input.append(relative).push_back('\0');
            corpus_input.append(*digest).push_back('\0');
        }

        std::set<std::string> actual;
        for (const auto& entry : std::filesystem::directory_iterator(root)) {
            if (entry.is_regular_file() && entry.path().extension() == ".txt") {
                actual.insert(entry.path().filename().generic_string());
            }
        }
        for (const auto& entry : std::filesystem::recursive_directory_iterator(root / "topology")) {
            if (entry.is_regular_file()) {
                actual.insert(std::filesystem::relative(entry.path(), root).generic_string());
            }
        }
        if (actual != declared) throw std::runtime_error("manifest contract scope has unnamed files");

        const auto& corpus_digest = string_field(*manifest, "corpus_sha256");
        if (!is_lower_hex(corpus_digest, 64) || diamond_support::sha256(corpus_input) != corpus_digest) {
            throw std::runtime_error("corpus digest mismatch");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "golden manifest failed: " << error.what() << '\n';
        return 1;
    }
}
