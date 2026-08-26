#include <filesystem>
#include <exception>
#include <iostream>
#include <string>

#include "diamond_training/checkpoint.hpp"

namespace {

constexpr char kUsage[] =
    "usage:\n"
    "  alphadiamond-checkpoint inspect <checkpoint-root>\n"
    "  alphadiamond-checkpoint validate <checkpoint-root>\n"
    "  alphadiamond-checkpoint migrate <checkpoint-v2-root> --out <new-v2-root>\n"
    "\n"
    "Only transactional native checkpoint-v2/v3 roots are accepted. Raw Python-v1\n"
    ".pt checkpoints must be pre-converted by their owning Python tooling.\n";

void print_info(const diamond_training::CheckpointInfo& info) {
    std::cout << "format_version=" << info.format_version << '\n'
              << "generation=" << info.generation.string() << '\n'
              << "training_step=" << info.training_step << '\n';
    if (!info.lineage)
        return;
    const auto& lineage = *info.lineage;
    std::cout << "run_id=" << lineage.run_id << '\n'
              << "iteration=" << lineage.iteration << '\n'
              << "model_step=" << lineage.model_step << '\n'
              << "optimizer_restored=" << (lineage.optimizer_restored ? "true" : "false") << '\n';
}

[[noreturn]] void unsupported_morph(const std::string& command,
                                    const std::filesystem::path& root) {
    // Resolve the root first so a raw Python-v1 .pt file consistently receives
    // the v1 rejection rather than a misleading feature error.
    (void)diamond_training::inspect_checkpoint_v2(root);
    throw diamond_training::CheckpointError(
        command + " is not supported for checkpoint-v2: v2 intentionally stores "
        "no model shape or recorded device. Use the Python-v1 tool before native "
        "conversion, then validate the resulting v2 checkpoint.");
}

int run(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << kUsage;
        return 2;
    }
    const std::string command = argv[1];
    const std::filesystem::path root = argv[2];
    if (command == "inspect") {
        if (argc != 3) throw diamond_training::CheckpointError("inspect takes one checkpoint root");
        print_info(diamond_training::inspect_checkpoint_v2(root));
        return 0;
    }
    if (command == "validate") {
        if (argc != 3) throw diamond_training::CheckpointError("validate takes one checkpoint root");
        print_info(diamond_training::validate_checkpoint_v2(root));
        std::cout << "valid=true\n";
        return 0;
    }
    if (command == "migrate") {
        if (argc != 5 || std::string(argv[3]) != "--out")
            throw diamond_training::CheckpointError("migrate requires --out <new-v2-root>");
        print_info(diamond_training::migrate_checkpoint_v2(root, argv[4]));
        std::cout << "migrated=true\n";
        return 0;
    }
    if (command == "device-remap" || command == "widen" || command == "deepen") {
        unsupported_morph(command, root);
    }
    std::cerr << kUsage;
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const diamond_training::CheckpointError& error) {
        std::cerr << "checkpoint: " << error.what() << '\n';
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "checkpoint: unexpected error: " << error.what() << '\n';
        return 1;
    }
}
