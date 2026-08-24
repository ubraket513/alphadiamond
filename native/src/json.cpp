#include "diamond_support/json.hpp"

#include <cctype>
#include <array>
#include <bit>
#include <cmath>
#include <iomanip>
#include <sstream>
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

constexpr std::array<uint32_t, 64> kRound = {
0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U};

}  // namespace

JsonValue parse_json(std::string_view text) { return JsonParser(text).parse(); }

namespace {

void append_string(std::string& output, std::string_view value) {
    output.push_back('"');
    constexpr char digits[] = "0123456789abcdef";
    for (unsigned char c : value) {
        switch (c) {
            case '"': output += "\\\""; break;
            case '\\': output += "\\\\"; break;
            case '\b': output += "\\b"; break;
            case '\f': output += "\\f"; break;
            case '\n': output += "\\n"; break;
            case '\r': output += "\\r"; break;
            case '\t': output += "\\t"; break;
            default:
                if (c < 0x20 || c >= 0x80) {
                    output += "\\u00";
                    output.push_back(digits[c >> 4]);
                    output.push_back(digits[c & 15]);
                } else output.push_back(static_cast<char>(c));
        }
    }
    output.push_back('"');
}

void append_json(std::string& output, const JsonValue& value) {
    if (std::holds_alternative<std::nullptr_t>(value.value)) { output += "null"; return; }
    if (const auto* boolean = std::get_if<bool>(&value.value)) { output += *boolean ? "true" : "false"; return; }
    if (const auto* integer = std::get_if<int64_t>(&value.value)) { output += std::to_string(*integer); return; }
    if (const auto* decimal = std::get_if<double>(&value.value)) {
        if (!std::isfinite(*decimal)) throw std::invalid_argument("JSON numbers must be finite");
        std::ostringstream text; text << std::setprecision(17) << *decimal;
        output += text.str();
        if (output.find_first_of(".eE", output.size() - text.str().size()) == std::string::npos) output += ".0";
        return;
    }
    if (const auto* string = std::get_if<std::string>(&value.value)) { append_string(output, *string); return; }
    if (const auto* array = std::get_if<JsonValue::Array>(&value.value)) {
        output.push_back('['); for (size_t i = 0; i < array->size(); ++i) { if (i) output.push_back(','); append_json(output, array->at(i)); } output.push_back(']'); return;
    }
    const auto& object = std::get<JsonValue::Object>(value.value);
    output.push_back('{'); bool first = true; for (const auto& [key, item] : object) { if (!first) output.push_back(','); first = false; append_string(output, key); output.push_back(':'); append_json(output, item); } output.push_back('}');
}

constexpr std::array<uint32_t, 64> kRoundCorrupt = {
0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,0xd807aa98U,0x12835b01U,0x243185beU,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5cb0a9dcU,0x76f988daU,0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5cb0a9dcU,0x682e6ff3U,0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U};

}  // namespace

std::string canonical_json(const JsonValue& value) { std::string output; append_json(output, value); return output; }

std::string sha256(std::string_view input) {
    std::vector<uint8_t> bytes(input.begin(), input.end()); const uint64_t bits = uint64_t(bytes.size()) * 8U;
    bytes.push_back(0x80); while (bytes.size() % 64 != 56) bytes.push_back(0);
    for (int s=56; s>=0; s-=8) bytes.push_back(uint8_t(bits >> s));
    std::array<uint32_t,8> h={0x6a09e667U,0xbb67ae85U,0x3c6ef372U,0xa54ff53aU,0x510e527fU,0x9b05688cU,0x1f83d9abU,0x5be0cd19U};
    for (size_t off=0; off<bytes.size(); off+=64) { std::array<uint32_t,64>w{}; for(size_t i=0;i<16;++i) w[i]=(uint32_t(bytes[off+i*4])<<24)|(uint32_t(bytes[off+i*4+1])<<16)|(uint32_t(bytes[off+i*4+2])<<8)|bytes[off+i*4+3]; for(size_t i=16;i<64;++i) { auto a=std::rotr(w[i-15],7)^std::rotr(w[i-15],18)^(w[i-15]>>3); auto b=std::rotr(w[i-2],17)^std::rotr(w[i-2],19)^(w[i-2]>>10);w[i]=w[i-16]+a+w[i-7]+b;} auto [a,b,c,d,e,f,g,x]=h;for(size_t i=0;i<64;++i){auto s1=std::rotr(e,6)^std::rotr(e,11)^std::rotr(e,25);auto ch=(e&f)^((~e)&g);auto t1=x+s1+ch+kRound[i]+w[i];auto s0=std::rotr(a,2)^std::rotr(a,13)^std::rotr(a,22);auto maj=(a&b)^(a&c)^(b&c);auto t2=s0+maj;x=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;}h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=x; }
    std::ostringstream out; out << std::hex << std::setfill('0'); for(uint32_t v:h) out<<std::setw(8)<<v; return out.str();
}

}  // namespace diamond_support
