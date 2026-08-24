#include "diamond_support/json.hpp"

#include <cctype>
#include <cmath>
#include <stdexcept>

namespace diamond_support {
namespace {

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
        throw std::runtime_error("invalid JSON at byte " + std::to_string(position_) + ": " + message);
    }

    void whitespace() {
        while (position_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[position_])))
            ++position_;
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
                if (text_[position_] == '-' || std::isdigit(static_cast<unsigned char>(text_[position_])))
                    return parse_number();
                error("unexpected token");
        }
    }

    JsonValue::Object parse_object() {
        if (!consume('{')) error("expected object");
        JsonValue::Object object;
        if (consume('}')) return object;
        for (;;) {
            whitespace();
            if (position_ >= text_.size() || text_[position_] != '"') error("expected object key");
            std::string key = parse_string();
            if (!consume(':')) error("expected ':'");
            if (!object.emplace(std::move(key), parse_value()).second) error("duplicate object key");
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
            if (character != '\\') { result.push_back(character); continue; }
            if (position_ >= text_.size()) error("unterminated string escape");
            switch (const char escaped = text_[position_++]) {
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

    JsonValue parse_number() {
        whitespace();
        const size_t begin = position_;
        if (text_[position_] == '-') ++position_;
        if (position_ >= text_.size() || !std::isdigit(static_cast<unsigned char>(text_[position_])))
            error("invalid number");
        while (position_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[position_]))) ++position_;
        bool floating = false;
        if (position_ < text_.size() && text_[position_] == '.') {
            floating = true;
            ++position_;
            if (position_ >= text_.size() || !std::isdigit(static_cast<unsigned char>(text_[position_])))
                error("invalid number");
            while (position_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[position_]))) ++position_;
        }
        if (position_ < text_.size() && (text_[position_] == 'e' || text_[position_] == 'E')) {
            floating = true;
            ++position_;
            if (position_ < text_.size() && (text_[position_] == '+' || text_[position_] == '-')) ++position_;
            if (position_ >= text_.size() || !std::isdigit(static_cast<unsigned char>(text_[position_])))
                error("invalid number");
            while (position_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[position_]))) ++position_;
        }
        const std::string token(text_.substr(begin, position_ - begin));
        try {
            if (!floating) return JsonValue{std::stoll(token)};
            const double value = std::stod(token);
            if (!std::isfinite(value)) error("number must be finite");
            return JsonValue{value};
        } catch (const std::exception&) {
            error(floating ? "number is out of range" : "integer is out of range");
        }
    }

    void literal(std::string_view expected) {
        if (text_.substr(position_, expected.size()) != expected) error("invalid literal");
        position_ += expected.size();
    }

    std::string_view text_;
    size_t position_ = 0;
};

}  // namespace

JsonValue parse_json(std::string_view text) { return JsonParser(text).parse(); }

}  // namespace diamond_support
