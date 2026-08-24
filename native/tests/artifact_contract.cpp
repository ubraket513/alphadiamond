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
        (void)diamond_model::validate_deployment_artifact(root);
        return false;
    } catch (const std::exception&) {
        return true;
    }
}

bool rejects_family(const std::filesystem::path& root, const std::string& family) {
    try {
        (void)diamond_model::validate_deployment_artifact(root, family);
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
    if (argc < 2) return 2;
    try {
        const std::filesystem::path source(argv[1]);
        TempArtifact temporary{std::filesystem::temp_directory_path() /
            ("alphadiamond-artifact-contract-" + std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()))};
        std::filesystem::copy(source, temporary.path,
            std::filesystem::copy_options::recursive);

        const auto metadata_path = temporary.path / "metadata.json";
        const std::string original_metadata = read_text(metadata_path);
        const auto artifact = diamond_model::validate_deployment_artifact(temporary.path);
        if (artifact.model_family != "soo") throw std::runtime_error("expected a Soo artifact");
        if (artifact.input_features != 4 || artifact.value_size != 1)
            throw std::runtime_error("Soo artifact reported the wrong tensor shape");

        // A family the binary knows, declared on weights that belong to another
        // one: the shapes must catch it even though the name is valid.
        std::string changed = original_metadata;
        replace_once(changed, "\"model_family\": \"soo\"", "\"model_family\": \"min\"");
        write_text(metadata_path, changed);
        if (!rejects(temporary.path)) throw std::runtime_error("wrong model family accepted");

        changed = original_metadata;
        erase_line_containing(changed, "\"model_family\"");
        write_text(metadata_path, changed);
        if (!rejects(temporary.path)) throw std::runtime_error("missing metadata field accepted");

        changed = original_metadata;
        replace_once(changed, "{", "{\n  \"unexpected\": 1,");
        write_text(metadata_path, changed);
        if (!rejects(temporary.path)) throw std::runtime_error("extra metadata field accepted");

        // Format 3 validates the weights against the *declared* architecture,
        // so a declaration that no longer matches the tensors must be refused.
        changed = original_metadata;
        replace_once(changed, "\"width\": 128", "\"width\": 64");
        write_text(metadata_path, changed);
        if (!rejects(temporary.path)) throw std::runtime_error("wrong model width accepted");

        changed = original_metadata;
        replace_once(changed, "\"residual_blocks\": 6", "\"residual_blocks\": 8");
        write_text(metadata_path, changed);
        if (!rejects(temporary.path)) throw std::runtime_error("wrong block count accepted");

        // A different game contract is a different game, not a different model.
        changed = original_metadata;
        replace_once(changed, "\"topology\": \"diamond73-v1\"", "\"topology\": \"diamond73-v2\"");
        write_text(metadata_path, changed);
        if (!rejects(temporary.path)) throw std::runtime_error("foreign game contract accepted");

        changed = original_metadata;
        replace_once(changed, "\"format_version\": 3", "\"format_version\": 2");
        write_text(metadata_path, changed);
        if (!rejects(temporary.path)) throw std::runtime_error("older format version accepted");

        changed = original_metadata;
        const std::string hash_prefix = "\"model_sha256\": \"";
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

        (void)diamond_model::validate_deployment_artifact(temporary.path, "soo");
        if (!rejects_family(temporary.path, "min"))
            throw std::runtime_error("family-scoped validation accepted the wrong family");

        // A release package carries only what the runtime loads: metadata, the
        // topology tables and the weights. The TorchScript graph and the parity
        // corpus stay in the development artifact, because shipping model.ts
        // would put a second 3.1 MB copy of every model in the installer for
        // something nothing opens.
        //
        // So a package's integrity rests on runtime_sha256, which was defined
        // as exactly that set. model_sha256 names the absent graph and is inert
        // there -- which is worth asserting rather than assuming, in both
        // directions: the package must still validate, and corrupting a tensor
        // must still be caught.
        TempArtifact packaged{std::filesystem::temp_directory_path() /
            ("alphadiamond-artifact-package-" + std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()))};
        std::filesystem::create_directories(packaged.path);
        for (const char* name : {"metadata.json", "topology_neighbour.i8",
                                 "topology_camp_positions.i32",
                                 "topology_pairwise_distance.i32",
                                 "topology_physical_to_canonical.i32",
                                 "topology_canonical_to_physical.i32"}) {
            std::filesystem::copy_file(temporary.path / name, packaged.path / name);
        }
        std::filesystem::copy(temporary.path / "weights", packaged.path / "weights",
                              std::filesystem::copy_options::recursive);

        (void)diamond_model::validate_deployment_artifact(packaged.path, "soo");

        const auto packaged_weight = packaged.path / "weights" / "value_head__2__bias.f32";
        const std::string packaged_original = read_text(packaged_weight);
        std::string flipped = packaged_original;
        flipped[0] = static_cast<char>(flipped[0] ^ 0x01);
        write_text(packaged_weight, flipped);
        if (!rejects(packaged.path))
            throw std::runtime_error("a corrupt tensor was accepted in a runtime-only package");
        write_text(packaged_weight, packaged_original);

        std::cout << "runtime-only package validated\n";

        // The second artifact, when supplied, is a Min bundle: the same
        // validator must accept it without a single Soo constant relaxed.
        if (argc >= 3) {
            const auto min_artifact = diamond_model::validate_deployment_artifact(argv[2], "min");
            if (min_artifact.input_features != 6 || min_artifact.value_size != 3)
                throw std::runtime_error("Min artifact reported the wrong tensor shape");
            std::cout << "Min deployment artifact accepted\n";
        }

        std::cout << "deployment artifact contract passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "artifact contract failed: " << error.what() << "\n";
        return 1;
    }
}
