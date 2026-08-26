#include <cstddef>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "diamond_model/deployment_artifact.hpp"
#include "diamond_release/promotion.hpp"
#include "diamond_support/json.hpp"

namespace {

using Json = diamond_support::JsonValue;
using Object = Json::Object;

constexpr char kUsage[] = "usage:\n"
                          "  alphadiamond-release validate <deployment-artifact>\n"
                          "  alphadiamond-release init <native-checkpoint-root> --family soo|min\n"
                          "  alphadiamond-release promote <checkpoint> --to candidate|promoted"
                          " [--artifact <deployment-artifact>]\n"
                          "  alphadiamond-release stage <output> --artifact <deployment-artifact>"
                          " [--artifact <deployment-artifact> ...] [--default <family> ...]\n";

void print_json(Object object) {
    std::cout << diamond_support::canonical_json(Json{std::move(object)}) << '\n';
}

int validate(const std::filesystem::path& root) {
    const auto artifact = diamond_model::validate_deployment_artifact(root);
    print_json({
        {"command", Json{std::string{"validate"}}},
        {"model_family", Json{artifact.model_family}},
        {"model_sha256", Json{artifact.model_sha256}},
        {"model_version", Json{artifact.model_version}},
        {"runtime_sha256", Json{artifact.runtime_sha256}},
        {"status", Json{std::string{"ok"}}},
    });
    return 0;
}

int initialize(int argc, char** argv) {
    if (argc != 5 || std::string{argv[3]} != "--family") {
        throw diamond_release::ReleaseError(
            "init requires <native-checkpoint-root> --family soo|min");
    }
    const auto manifest = diamond_release::initialize_promotion(argv[2], argv[4]);
    print_json({
        {"checkpoint_generation", Json{manifest.checkpoint_generation}},
        {"checkpoint_sha256", Json{manifest.checkpoint_sha256}},
        {"command", Json{std::string{"init"}}},
        {"model_family", Json{manifest.model_family}},
        {"state", Json{diamond_release::promotion_state_name(manifest.state)}},
        {"status", Json{std::string{"ok"}}},
    });
    return 0;
}

int promote(int argc, char** argv) {
    if (argc < 5 || std::string{argv[3]} != "--to") {
        throw diamond_release::ReleaseError(
            "promote requires <checkpoint> --to candidate|promoted");
    }
    const auto target = diamond_release::parse_promotion_state(argv[4]);
    if (target == diamond_release::PromotionState::archival)
        throw diamond_release::ReleaseError("promote target must be candidate or promoted");
    std::optional<std::filesystem::path> artifact;
    for (int index = 5; index < argc; ++index) {
        if (std::string{argv[index]} != "--artifact" || ++index >= argc || artifact) {
            throw diamond_release::ReleaseError(
                "promote accepts one optional --artifact <deployment-artifact>");
        }
        artifact = std::filesystem::path{argv[index]};
    }
    const auto manifest = diamond_release::promote_checkpoint(argv[2], target, artifact);
    print_json({
        {"checkpoint_sha256", Json{manifest.checkpoint_sha256}},
        {"command", Json{std::string{"promote"}}},
        {"model_family", Json{manifest.model_family}},
        {"state", Json{diamond_release::promotion_state_name(manifest.state)}},
        {"status", Json{std::string{"ok"}}},
    });
    return 0;
}

int stage(int argc, char** argv) {
    std::vector<std::filesystem::path> artifacts;
    std::vector<std::string> defaults;
    for (int index = 3; index < argc; ++index) {
        const std::string option = argv[index];
        if (++index >= argc) {
            throw diamond_release::ReleaseError(option + " requires a value");
        }
        if (option == "--artifact") {
            artifacts.emplace_back(argv[index]);
        } else if (option == "--default") {
            defaults.emplace_back(argv[index]);
        } else {
            throw diamond_release::ReleaseError("unknown stage option: " + option);
        }
    }
    const auto index = diamond_release::stage_release_models(argv[2], artifacts, defaults);
    print_json({
        {"command", Json{std::string{"stage"}}},
        {"models", Json{static_cast<int64_t>(index.models.size())}},
        {"output", Json{std::filesystem::path{argv[2]}.generic_string()}},
        {"status", Json{std::string{"ok"}}},
    });
    return 0;
}

int run(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << kUsage;
        return 2;
    }
    const std::string command = argv[1];
    if (command == "validate") {
        if (argc != 3) throw diamond_release::ReleaseError("validate takes one artifact");
        return validate(argv[2]);
    }
    if (command == "init") return initialize(argc, argv);
    if (command == "promote") return promote(argc, argv);
    if (command == "stage") {
        if (argc < 5) throw diamond_release::ReleaseError("stage requires an output and artifact");
        return stage(argc, argv);
    }
    std::cerr << kUsage;
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const diamond_release::ReleaseError& error) {
        std::cerr << "release: " << error.what() << '\n';
        return 3;
    } catch (const std::exception& error) {
        std::cerr << "release: " << error.what() << '\n';
        return 3;
    }
}
