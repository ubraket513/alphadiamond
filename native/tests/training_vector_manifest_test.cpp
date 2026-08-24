#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "diamond_support/json.hpp"

namespace {
using diamond_support::JsonValue;

const JsonValue& field(const JsonValue::Object& object, const std::string& name) { const auto it = object.find(name); if (it == object.end()) throw std::runtime_error("manifest is missing field: " + name); return it->second; }
const std::string& string_field(const JsonValue::Object& object, const std::string& name) { const auto* value = std::get_if<std::string>(&field(object, name).value); if (!value) throw std::runtime_error("manifest field must be a string: " + name); return *value; }
int64_t integer_field(const JsonValue::Object& object, const std::string& name) { const auto* value = std::get_if<int64_t>(&field(object, name).value); if (!value) throw std::runtime_error("manifest field must be an integer: " + name); return *value; }
void require_keys(const JsonValue::Object& object, const std::set<std::string>& expected, const std::string& where) { std::set<std::string> actual; for (const auto& [key, ignored] : object) { (void)ignored; actual.insert(key); } if (actual != expected) throw std::runtime_error("manifest fields mismatch: " + where); }
std::vector<uint8_t> bytes(const std::filesystem::path& path) { std::ifstream file(path, std::ios::binary); if (!file) throw std::runtime_error("missing fixture payload: " + path.string()); return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()}; }

std::string sha256(std::vector<uint8_t> value) {
    static constexpr std::array<uint32_t, 64> k = {
        0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
        0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
        0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
        0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
        0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
        0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
        0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
        0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U};
    // This is deliberately test-local: the contract test independently checks each frozen payload.
    const uint64_t bits = static_cast<uint64_t>(value.size()) * 8U; value.push_back(0x80U); while (value.size() % 64 != 56) value.push_back(0); for (int shift = 56; shift >= 0; shift -= 8) value.push_back(static_cast<uint8_t>(bits >> shift));
    std::array<uint32_t, 8> hash = {0x6a09e667U,0xbb67ae85U,0x3c6ef372U,0xa54ff53aU,0x510e527fU,0x9b05688cU,0x1f83d9abU,0x5be0cd19U};
    for (size_t offset = 0; offset < value.size(); offset += 64) { std::array<uint32_t,64> words{}; for (size_t i=0;i<16;++i) words[i]=(uint32_t(value[offset+4*i])<<24)|(uint32_t(value[offset+4*i+1])<<16)|(uint32_t(value[offset+4*i+2])<<8)|value[offset+4*i+3]; for(size_t i=16;i<64;++i){const auto a=std::rotr(words[i-15],7)^std::rotr(words[i-15],18)^(words[i-15]>>3);const auto b=std::rotr(words[i-2],17)^std::rotr(words[i-2],19)^(words[i-2]>>10);words[i]=words[i-16]+a+words[i-7]+b;} auto [a,b,c,d,e,f,g,h]=hash; for(size_t i=0;i<64;++i){const auto s1=std::rotr(e,6)^std::rotr(e,11)^std::rotr(e,25);const auto t1=h+s1+((e&f)^((~e)&g))+k[i]+words[i];const auto s0=std::rotr(a,2)^std::rotr(a,13)^std::rotr(a,22);const auto t2=s0+((a&b)^(a&c)^(b&c));h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;} hash[0]+=a;hash[1]+=b;hash[2]+=c;hash[3]+=d;hash[4]+=e;hash[5]+=f;hash[6]+=g;hash[7]+=h; }
    std::ostringstream output; output << std::hex << std::setfill('0'); for (auto word : hash) output << std::setw(8) << word; return output.str();
}
}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: training_vector_manifest_test <fixture-dir>");
        const auto root = std::filesystem::path(argv[1]); const auto input = bytes(root / "manifest.json");
        const auto parsed = diamond_support::parse_json({reinterpret_cast<const char*>(input.data()), input.size()}); const auto* manifest = std::get_if<JsonValue::Object>(&parsed.value); if (!manifest) throw std::runtime_error("manifest root must be an object");
        require_keys(*manifest, {"device","dtype","families","files","fixture_version","game_contract"}, "root");
        if (integer_field(*manifest,"fixture_version") != 1 || string_field(*manifest,"game_contract") != "diamond-authoritative-rules-v1" || string_field(*manifest,"dtype") != "float32" || string_field(*manifest,"device") != "cpu") throw std::runtime_error("manifest contract mismatch");
        const auto* families=std::get_if<JsonValue::Array>(&field(*manifest,"families").value); if(!families||families->size()!=2||!std::holds_alternative<std::string>((*families)[0].value)||!std::holds_alternative<std::string>((*families)[1].value)||std::get<std::string>((*families)[0].value)!="soo"||std::get<std::string>((*families)[1].value)!="min") throw std::runtime_error("manifest families mismatch");
        const auto* files=std::get_if<JsonValue::Object>(&field(*manifest,"files").value); if(!files||files->empty()) throw std::runtime_error("manifest files must be non-empty");
        for(const auto& [relative, entry_value]:*files){const auto* entry=std::get_if<JsonValue::Object>(&entry_value.value);if(!entry||relative.empty()||relative.find("..")!=std::string::npos)throw std::runtime_error("invalid fixture descriptor: "+relative);require_keys(*entry,{"byte_count","element_type","sha256","shape"},relative);const auto payload=bytes(root/relative);if(integer_field(*entry,"byte_count")!=static_cast<int64_t>(payload.size())||sha256(payload)!=string_field(*entry,"sha256"))throw std::runtime_error("fixture digest mismatch: "+relative);}
        return 0;
    } catch (const std::exception& error) { std::cerr << "training vector manifest failed: " << error.what() << '\n'; return 1; }
}
