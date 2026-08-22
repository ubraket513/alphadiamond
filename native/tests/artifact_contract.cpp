#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

#include "diamond_model/deployment_artifact.hpp"

namespace {

std::string read_text(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot read " + path.string());
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

void write_text(const std::filesystem::path& path, const std::string& text) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) throw std::runtime_error("cannot write " + path.string());
    file.write(text.data(), static_cast<std::streamsize>(text.size()));
}

void replace_once(std::string& text, const std::string& before, const std::string& after) {
    const size_t position = text.find(before);
    if (position == std::string::npos) throw std::runtime_error("fixture text not found");
    text.replace(position, before.size(), after);
}

void erase_line_containing(std::string& text, const std::string& marker) {
    const size_t marker_at = text.find(marker);
    if (marker_at == std::string::npos) throw std::runtime_error("fixture line not found");
    const size_t line_begin = text.rfind('\n', marker_at);
    const size_t erase_begin = line_begin == std::string::npos ? 0 : line_begin + 1;
    const size_t line_end = text.find('\n', marker_at);
    text.erase(erase_begin, line_end == std::string::npos
        ? text.size() - erase_begin : line_end + 1 - erase_begin);
}

bool rejects(const std::filesystem::path& root) {
    try {
        (void)diamond_model::validate_soo_deployment_artifact(root);
        return false;
    } catch (const std::exception&) {
        return true;
    }
}

struct TempArtifact {
    std::filesystem::path path;
    ~TempArtifact() { std::error_code ignored; std::filesystem::remove_all(path, ignored); }
};

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    try {
        const std::filesystem::path source(argv[1]);
        TempArtifact temporary{std::filesystem::temp_directory_path() /
            ("alphadiamond-artifact-contract-" + std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()))};
        std::filesystem::copy(source, temporary.path,
            std::filesystem::copy_options::recursive);

        const auto metadata_path = temporary.path / "metadata.json";
        const std::string original_metadata = read_text(metadata_path);
        (void)diamond_model::validate_soo_deployment_artifact(temporary.path);

        std::string changed = original_metadata;
        erase_line_containing(changed, "\"model_name\"");
        write_text(metadata_path, changed);
        if (!rejects(temporary.path)) throw std::runtime_error("missing metadata field accepted");

        changed = original_metadata;
        replace_once(changed, "{", "{\n  \"unexpected\": 1,");
        write_text(metadata_path, changed);
        if (!rejects(temporary.path)) throw std::runtime_error("extra metadata field accepted");

        changed = original_metadata;
        replace_once(changed, "  \"width\": 128", "  \"width\": 64");
        write_text(metadata_path, changed);
        if (!rejects(temporary.path)) throw std::runtime_error("wrong model width accepted");

        changed = original_metadata;
        const std::string hash_prefix = "  \"model_sha256\": \"";
        const size_t hash_at = changed.find(hash_prefix);
        if (hash_at == std::string::npos) throw std::runtime_error("model hash fixture missing");
        changed.replace(hash_at + hash_prefix.size(), 64, std::string(64, '0'));
        write_text(metadata_path, changed);
        if (!rejects(temporary.path)) throw std::runtime_error("wrong model hash accepted");
        write_text(metadata_path, original_metadata);

        const auto weight = temporary.path / "weights" /
            "value_head__2__bias.f32";
        const auto moved = temporary.path / "weights" /
            "value_head__2__bias.f32.missing";
        std::filesystem::rename(weight, moved);
        if (!rejects(temporary.path)) throw std::runtime_error("missing tensor accepted");
        std::filesystem::rename(moved, weight);

        const auto extra = temporary.path / "weights" / "unexpected.f32";
        write_text(extra, "data");
        if (!rejects(temporary.path)) throw std::runtime_error("extra tensor accepted");
        std::filesystem::remove(extra);

        const std::string original_weight = read_text(weight);
        write_text(weight, "x");
        if (!rejects(temporary.path)) throw std::runtime_error("wrong tensor shape accepted");
        write_text(weight, original_weight);

        std::string corrupted_weight = original_weight;
        corrupted_weight[0] = static_cast<char>(corrupted_weight[0] ^ 0x01);
        write_text(weight, corrupted_weight);
        if (!rejects(temporary.path)) throw std::runtime_error("corrupt tensor content accepted");
        write_text(weight, original_weight);

        (void)diamond_model::validate_soo_deployment_artifact(temporary.path);
        std::cout << "Soo deployment artifact contract passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "artifact contract failed: " << error.what() << "\n";
        return 1;
    }
}
