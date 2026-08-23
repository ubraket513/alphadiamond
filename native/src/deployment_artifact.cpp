#include "diamond_model/deployment_artifact.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace diamond_model {
namespace {

struct JsonValue {
    using Array = std::vector<JsonValue>;
    using Object = std::map<std::string, JsonValue>;
    std::variant<std::nullptr_t, bool, int64_t, std::string, Array, Object> value;
};

class JsonParser {
  public:
    explicit JsonParser(std::string_view text) : text_(text) {}

    JsonValue parse() {
        JsonValue result = parse_value();
        whitespace();
        if (position_ != text_.size()) error("trailing data");
        return result;
    }

  private:
    [[noreturn]] void error(const char* message) const {
        throw std::runtime_error("invalid deployment metadata at byte " +
                                 std::to_string(position_) + ": " + message);
    }

    void whitespace() {
        while (position_ < text_.size() &&
               std::isspace(static_cast<unsigned char>(text_[position_]))) ++position_;
    }

    bool consume(char expected) {
        whitespace();
        if (position_ < text_.size() && text_[position_] == expected) {
            ++position_;
            return true;
        }
        return false;
    }

    JsonValue parse_value() {
        whitespace();
        if (position_ >= text_.size()) error("unexpected end of input");
        switch (text_[position_]) {
            case '{': return JsonValue{parse_object()};
            case '[': return JsonValue{parse_array()};
            case '"': return JsonValue{parse_string()};
            case 'n': literal("null"); return JsonValue{nullptr};
            case 't': literal("true"); return JsonValue{true};
            case 'f': literal("false"); return JsonValue{false};
            default:
                if (text_[position_] == '-' || std::isdigit(
                        static_cast<unsigned char>(text_[position_])))
                    return JsonValue{parse_integer()};
                error("unexpected token");
        }
    }

    JsonValue::Object parse_object() {
        if (!consume('{')) error("expected object");
        JsonValue::Object object;
        if (consume('}')) return object;
        for (;;) {
            whitespace();
            if (position_ >= text_.size() || text_[position_] != '"')
                error("expected object key");
            std::string key = parse_string();
            if (!consume(':')) error("expected ':'");
            if (!object.emplace(std::move(key), parse_value()).second)
                error("duplicate object key");
            if (consume('}')) return object;
            if (!consume(',')) error("expected ','");
        }
    }

    JsonValue::Array parse_array() {
        if (!consume('[')) error("expected array");
        JsonValue::Array array;
        if (consume(']')) return array;
        for (;;) {
            array.push_back(parse_value());
            if (consume(']')) return array;
            if (!consume(',')) error("expected ','");
        }
    }

    std::string parse_string() {
        if (!consume('"')) error("expected string");
        std::string result;
        while (position_ < text_.size()) {
            const char character = text_[position_++];
            if (character == '"') return result;
            if (static_cast<unsigned char>(character) < 0x20) error("control character in string");
            if (character != '\\') {
                result.push_back(character);
                continue;
            }
            if (position_ >= text_.size()) error("unterminated string escape");
            const char escaped = text_[position_++];
            switch (escaped) {
                case '"': result.push_back('"'); break;
                case '\\': result.push_back('\\'); break;
                case '/': result.push_back('/'); break;
                case 'b': result.push_back('\b'); break;
                case 'f': result.push_back('\f'); break;
                case 'n': result.push_back('\n'); break;
                case 'r': result.push_back('\r'); break;
                case 't': result.push_back('\t'); break;
                default: error("unsupported string escape");
            }
        }
        error("unterminated string");
    }

    int64_t parse_integer() {
        whitespace();
        const size_t begin = position_;
        if (text_[position_] == '-') ++position_;
        if (position_ >= text_.size() || !std::isdigit(
                static_cast<unsigned char>(text_[position_]))) error("invalid integer");
        while (position_ < text_.size() && std::isdigit(
                   static_cast<unsigned char>(text_[position_]))) ++position_;
        if (position_ < text_.size() &&
            (text_[position_] == '.' || text_[position_] == 'e' || text_[position_] == 'E'))
            error("metadata numbers must be integers");
        try {
            return std::stoll(std::string(text_.substr(begin, position_ - begin)));
        } catch (const std::exception&) {
            error("integer is out of range");
        }
    }

    void literal(std::string_view expected) {
        if (text_.substr(position_, expected.size()) != expected) error("invalid literal");
        position_ += expected.size();
    }

    std::string_view text_;
    size_t position_ = 0;
};

const JsonValue& field(const JsonValue::Object& object, const std::string& name) {
    const auto found = object.find(name);
    if (found == object.end()) throw std::runtime_error("metadata is missing field: " + name);
    return found->second;
}

int64_t integer_field(const JsonValue::Object& object, const std::string& name) {
    const auto* value = std::get_if<int64_t>(&field(object, name).value);
    if (!value) throw std::runtime_error("metadata field must be an integer: " + name);
    return *value;
}

std::string string_field(const JsonValue::Object& object, const std::string& name) {
    const auto* value = std::get_if<std::string>(&field(object, name).value);
    if (!value) throw std::runtime_error("metadata field must be a string: " + name);
    return *value;
}

void require_string(const JsonValue::Object& object, const std::string& name,
                    const std::string& expected) {
    const auto actual = string_field(object, name);
    if (actual != expected)
        throw std::runtime_error("metadata " + name + " mismatch: expected " + expected +
                                 ", got " + actual);
}

void require_integer(const JsonValue::Object& object, const std::string& name, int64_t expected) {
    const int64_t actual = integer_field(object, name);
    if (actual != expected)
        throw std::runtime_error("metadata " + name + " mismatch: expected " +
                                 std::to_string(expected) + ", got " + std::to_string(actual));
}

void require_shape(const JsonValue::Object& object, const std::string& name,
                   std::initializer_list<int64_t> expected) {
    const auto* array = std::get_if<JsonValue::Array>(&field(object, name).value);
    if (!array || array->size() != expected.size())
        throw std::runtime_error("metadata shape mismatch: " + name);
    size_t index = 0;
    for (int64_t dimension : expected) {
        const auto* actual = std::get_if<int64_t>(&array->at(index++).value);
        if (!actual || *actual != dimension)
            throw std::runtime_error("metadata shape mismatch: " + name);
    }
}

bool is_hex_digest(const std::string& value) {
    return value.size() == 64 && std::all_of(value.begin(), value.end(), [](char character) {
        return std::isxdigit(static_cast<unsigned char>(character)) != 0;
    });
}

std::string read_text(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open deployment file: " + path.string());
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

constexpr std::array<uint32_t, 64> kSha256Round = {
    0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
    0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
    0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
    0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
    0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
    0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
    0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
    0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U};

std::string sha256_bytes(std::vector<uint8_t> bytes) {
    const uint64_t bit_size = static_cast<uint64_t>(bytes.size()) * 8U;
    bytes.push_back(0x80U);
    while (bytes.size() % 64 != 56) bytes.push_back(0U);
    for (int shift = 56; shift >= 0; shift -= 8)
        bytes.push_back(static_cast<uint8_t>((bit_size >> shift) & 0xffU));

    std::array<uint32_t, 8> hash = {0x6a09e667U,0xbb67ae85U,0x3c6ef372U,0xa54ff53aU,
                                     0x510e527fU,0x9b05688cU,0x1f83d9abU,0x5be0cd19U};
    for (size_t offset = 0; offset < bytes.size(); offset += 64) {
        std::array<uint32_t, 64> words{};
        for (size_t index = 0; index < 16; ++index) {
            const size_t at = offset + index * 4;
            words[index] = (static_cast<uint32_t>(bytes[at]) << 24) |
                           (static_cast<uint32_t>(bytes[at + 1]) << 16) |
                           (static_cast<uint32_t>(bytes[at + 2]) << 8) |
                           static_cast<uint32_t>(bytes[at + 3]);
        }
        for (size_t index = 16; index < 64; ++index) {
            const uint32_t s0 = std::rotr(words[index - 15], 7) ^
                                std::rotr(words[index - 15], 18) ^ (words[index - 15] >> 3);
            const uint32_t s1 = std::rotr(words[index - 2], 17) ^
                                std::rotr(words[index - 2], 19) ^ (words[index - 2] >> 10);
            words[index] = words[index - 16] + s0 + words[index - 7] + s1;
        }
        auto [a,b,c,d,e,f,g,h] = hash;
        for (size_t index = 0; index < 64; ++index) {
            const uint32_t sum1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
            const uint32_t choice = (e & f) ^ ((~e) & g);
            const uint32_t temporary1 = h + sum1 + choice + kSha256Round[index] + words[index];
            const uint32_t sum0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
            const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temporary2 = sum0 + majority;
            h=g; g=f; f=e; e=d+temporary1; d=c; c=b; b=a; a=temporary1+temporary2;
        }
        hash[0]+=a; hash[1]+=b; hash[2]+=c; hash[3]+=d;
        hash[4]+=e; hash[5]+=f; hash[6]+=g; hash[7]+=h;
    }
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (uint32_t value : hash) output << std::setw(8) << value;
    return output.str();
}

std::vector<uint8_t> read_bytes(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open deployment file: " + path.string());
    std::vector<uint8_t> bytes;
    char byte = 0;
    while (file.get(byte)) bytes.push_back(static_cast<uint8_t>(byte));
    return bytes;
}

std::string sha256_file(const std::filesystem::path& path) {
    return sha256_bytes(read_bytes(path));
}

void add_tensor(std::map<std::string, uintmax_t>& files, const std::string& name,
                uintmax_t elements) {
    files.emplace(name + ".f32", elements * sizeof(float));
}

std::map<std::string, uintmax_t> expected_weights(int64_t width, int64_t blocks,
                                                  int64_t input_features, int64_t value_size) {
    std::map<std::string, uintmax_t> files;
    add_tensor(files, "trunk__input_projection__weight", width * input_features);
    add_tensor(files, "trunk__input_projection__bias", width);
    for (int64_t block = 0; block < blocks; ++block) {
        const std::string prefix = "trunk__blocks__" + std::to_string(block) + "__";
        add_tensor(files, prefix + "self_projection__weight", width * width);
        add_tensor(files, prefix + "self_projection__bias", width);
        for (int direction = 0; direction < 6; ++direction)
            add_tensor(files, prefix + "direction_projections__" + std::to_string(direction) +
                              "__weight", width * width);
        add_tensor(files, prefix + "norm__weight", width);
        add_tensor(files, prefix + "norm__bias", width);
    }
    add_tensor(files, "trunk__output_norm__weight", width);
    add_tensor(files, "trunk__output_norm__bias", width);
    for (const std::string head : {"policy_head__source", "policy_head__destination"}) {
        add_tensor(files, head + "__weight", width * width);
        add_tensor(files, head + "__bias", width);
    }
    add_tensor(files, "value_head__0__weight", width * width);
    add_tensor(files, "value_head__0__bias", width);
    add_tensor(files, "value_head__2__weight", width * value_size);
    add_tensor(files, "value_head__2__bias", value_size);
    add_tensor(files, "trunk__adjacency", 6 * 73 * 73);
    return files;
}

void validate_file_size(const std::filesystem::path& path, uintmax_t expected) {
    if (!std::filesystem::is_regular_file(path) || std::filesystem::file_size(path) != expected)
        throw std::runtime_error("deployment file size mismatch: " + path.string());
}

std::string runtime_sha256(const std::filesystem::path& root,
                           const std::map<std::string, uintmax_t>& weights) {
    std::vector<std::string> relative_paths = {
        "topology_neighbour.i8",
        "topology_camp_positions.i32",
        "topology_pairwise_distance.i32",
        "topology_physical_to_canonical.i32",
        "topology_canonical_to_physical.i32",
    };
    for (const auto& [name, unused] : weights) {
        (void)unused;
        relative_paths.push_back("weights/" + name);
    }
    std::sort(relative_paths.begin(), relative_paths.end());
    std::vector<uint8_t> aggregate;
    for (const std::string& relative : relative_paths) {
        aggregate.insert(aggregate.end(), relative.begin(), relative.end());
        aggregate.push_back(0);
        const auto contents = read_bytes(root / std::filesystem::path(relative));
        aggregate.insert(aggregate.end(), contents.begin(), contents.end());
    }
    return sha256_bytes(std::move(aggregate));
}


const JsonValue::Object& object_field(const JsonValue::Object& object, const std::string& name) {
    const auto* nested = std::get_if<JsonValue::Object>(&field(object, name).value);
    if (!nested) throw std::runtime_error("metadata field must be an object: " + name);
    return *nested;
}

void require_keys(const JsonValue::Object& object, const std::set<std::string>& expected,
                  const std::string& where) {
    std::set<std::string> actual;
    for (const auto& [key, unused] : object) {
        (void)unused;
        actual.insert(key);
    }
    if (actual != expected) throw std::runtime_error("deployment metadata fields mismatch: " + where);
}

// A batched shape: [batch, rest...]. The batch size is the corpus batch and is
// not fixed by the format, so only the trailing dimensions are required.
void require_batched_shape(const JsonValue::Object& object, const std::string& name,
                           std::initializer_list<int64_t> trailing) {
    const auto* array = std::get_if<JsonValue::Array>(&field(object, name).value);
    if (!array || array->size() != trailing.size() + 1)
        throw std::runtime_error("metadata shape mismatch: " + name);
    const auto* batch = std::get_if<int64_t>(&array->at(0).value);
    if (!batch || *batch <= 0) throw std::runtime_error("metadata shape mismatch: " + name);
    size_t index = 1;
    for (int64_t dimension : trailing) {
        const auto* actual = std::get_if<int64_t>(&array->at(index++).value);
        if (!actual || *actual != dimension)
            throw std::runtime_error("metadata shape mismatch: " + name);
    }
}

void require_nullable_digest(const JsonValue::Object& object, const std::string& name) {
    const JsonValue& value = field(object, name);
    if (std::holds_alternative<std::nullptr_t>(value.value)) return;
    const auto* digest = std::get_if<std::string>(&value.value);
    if (!digest || !is_hex_digest(*digest))
        throw std::runtime_error("metadata " + name + " is invalid");
}

}  // namespace

DeploymentArtifact validate_deployment_artifact(const std::filesystem::path& root) {
    static const std::set<std::string> expected_keys = {
        "architecture", "corpus_seed", "dtype", "format_version", "game_contract",
        "model_family", "model_sha256", "model_version", "runtime_sha256", "source",
        "tensor_shapes"};
    static const std::set<std::string> architecture_keys = {"residual_blocks", "type", "width"};
    static const std::set<std::string> contract_keys = {"action_space", "encoder", "topology"};
    static const std::set<std::string> shape_keys = {"input", "policy", "value"};
    static const std::set<std::string> source_keys = {
        "checkpoint_sha256", "training_commit", "training_step"};

    const JsonValue parsed = JsonParser(read_text(root / "metadata.json")).parse();
    const auto* object = std::get_if<JsonValue::Object>(&parsed.value);
    if (!object) throw std::runtime_error("deployment metadata must be a JSON object");
    require_keys(*object, expected_keys, "metadata");

    require_integer(*object, "format_version", 3);
    require_string(*object, "dtype", "float32");

    // The game contract is what this binary implements. An artifact declaring
    // another one is for a different game, not a different model.
    const JsonValue::Object& contract = object_field(*object, "game_contract");
    require_keys(contract, contract_keys, "game_contract");
    require_string(contract, "topology", "diamond73-v1");
    require_string(contract, "encoder", "diamond-camp-relative-v1");
    require_string(contract, "action_space", "diamond73-srcdst-v1");

    const std::string family = string_field(*object, "model_family");
    int64_t input_features = 0;
    int64_t value_size = 0;
    if (family == "soo") {
        input_features = 4;
        value_size = 1;
    } else if (family == "min") {
        input_features = 6;
        value_size = 3;
    } else {
        throw std::runtime_error("unknown model family: " + family);
    }

    const std::string model_version = string_field(*object, "model_version");
    if (model_version.empty()) throw std::runtime_error("metadata model_version is empty");

    const JsonValue::Object& architecture = object_field(*object, "architecture");
    require_keys(architecture, architecture_keys, "architecture");
    require_string(architecture, "type", "directional_residual");
    const int64_t width = integer_field(architecture, "width");
    const int64_t blocks = integer_field(architecture, "residual_blocks");
    if (width <= 0 || blocks <= 0)
        throw std::runtime_error("metadata architecture must be positive");

    const JsonValue::Object& shapes = object_field(*object, "tensor_shapes");
    require_keys(shapes, shape_keys, "tensor_shapes");
    require_batched_shape(shapes, "input", {73, input_features});
    require_batched_shape(shapes, "policy", {5329});
    require_batched_shape(shapes, "value", {value_size});

    const JsonValue::Object& provenance = object_field(*object, "source");
    require_keys(provenance, source_keys, "source");
    require_nullable_digest(provenance, "checkpoint_sha256");

    (void)integer_field(*object, "corpus_seed");

    const std::string model_hash = string_field(*object, "model_sha256");
    if (!is_hex_digest(model_hash)) throw std::runtime_error("metadata model_sha256 is invalid");
    if (sha256_file(root / "model.ts") != model_hash)
        throw std::runtime_error("deployment model SHA-256 mismatch");
    const std::string runtime_hash = string_field(*object, "runtime_sha256");
    if (!is_hex_digest(runtime_hash)) throw std::runtime_error("metadata runtime_sha256 is invalid");

    const auto weights = root / "weights";
    if (!std::filesystem::is_directory(weights))
        throw std::runtime_error("deployment weights directory is missing");
    // Checked against the *declared* architecture, not against constants: that
    // is the whole point of format 3.
    const auto expected = expected_weights(width, blocks, input_features, value_size);
    std::set<std::string> actual;
    for (const auto& entry : std::filesystem::directory_iterator(weights)) {
        if (!entry.is_regular_file())
            throw std::runtime_error("unexpected deployment weight entry: " + entry.path().string());
        actual.insert(entry.path().filename().string());
    }
    std::set<std::string> expected_names;
    for (const auto& [name, bytes] : expected) {
        expected_names.insert(name);
        validate_file_size(weights / name, bytes);
    }
    if (actual != expected_names) throw std::runtime_error("deployment weight tensor manifest mismatch");

    validate_file_size(root / "topology_neighbour.i8", 73 * 6);
    validate_file_size(root / "topology_camp_positions.i32", 6 * 10 * sizeof(int32_t));
    validate_file_size(root / "topology_pairwise_distance.i32", 73 * 73 * sizeof(int32_t));
    validate_file_size(root / "topology_physical_to_canonical.i32", 6 * 73 * sizeof(int32_t));
    validate_file_size(root / "topology_canonical_to_physical.i32", 6 * 73 * sizeof(int32_t));
    if (runtime_sha256(root, expected) != runtime_hash)
        throw std::runtime_error("deployment runtime SHA-256 mismatch");

    return DeploymentArtifact{root,   weights,        family,         model_version,
                              width,  blocks,         input_features, value_size};
}

DeploymentArtifact validate_deployment_artifact(const std::filesystem::path& root,
                                                const std::string& expected_family) {
    DeploymentArtifact artifact = validate_deployment_artifact(root);
    if (artifact.model_family != expected_family)
        throw std::runtime_error("deployment model family mismatch: expected " + expected_family +
                                 ", got " + artifact.model_family);
    return artifact;
}

}  // namespace diamond_model
