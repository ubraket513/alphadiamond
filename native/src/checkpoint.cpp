#include "diamond_training/checkpoint.hpp"

#include <atomic>
#include <cstdlib>
#include <fstream>
#include <string>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace diamond_training {
namespace {
constexpr char kPointer[] = "CURRENT";
constexpr char kState[] = "state.pt";
constexpr char kOptimizer[] = "optimizer.pt";

std::string read_pointer(const std::filesystem::path& root) {
    if (!std::filesystem::is_directory(root))
        throw CheckpointError("checkpoint v2 root must be a directory; v1 files are rejected");
    std::ifstream input(root / kPointer, std::ios::binary);
    std::string generation;
    std::getline(input, generation);
    if (!input || generation.empty() || generation.find_first_of("\\/.") != std::string::npos)
        throw CheckpointError("checkpoint v2 CURRENT pointer is invalid");
    return generation;
}

void atomic_write(const std::filesystem::path& path, const std::string& contents) {
    static std::atomic_uint64_t sequence{0};
    std::filesystem::create_directories(path.parent_path());
    const auto pid =
#ifdef _WIN32
        static_cast<unsigned long long>(GetCurrentProcessId());
#else
        static_cast<unsigned long long>(::getpid());
#endif
    const auto temporary = path.parent_path() / (".checkpoint-tmp-" + std::to_string(pid) + "-" + std::to_string(++sequence));
    { std::ofstream output(temporary, std::ios::binary | std::ios::trunc); if (!output) throw CheckpointError("cannot write checkpoint transaction"); output << contents; if (!output) throw CheckpointError("cannot write checkpoint transaction"); }
    if (std::getenv("DIAMOND_CHECKPOINT_FAIL_ACTIVATE")) { std::filesystem::remove(temporary); throw CheckpointError("injected checkpoint activation failure"); }
#ifdef _WIN32
    if (!MoveFileExW(temporary.wstring().c_str(), path.wstring().c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) { std::filesystem::remove(temporary); throw CheckpointError("cannot activate checkpoint transaction"); }
#else
    std::error_code error; std::filesystem::rename(temporary, path, error); if (error) { std::filesystem::remove(temporary); throw CheckpointError("cannot activate checkpoint transaction"); }
#endif
}

CheckpointInfo state_info(const std::filesystem::path& generation) {
    const auto state = generation / kState;
    const auto optimizer = generation / kOptimizer;
    if (!std::filesystem::is_regular_file(state) || !std::filesystem::is_regular_file(optimizer))
        throw CheckpointError("checkpoint generation is incomplete");
    std::ifstream input(generation / "training_step", std::ios::binary);
    uint64_t step = 0;
    if (!(input >> step)) throw CheckpointError("checkpoint training_step is invalid");
    return {generation, step};
}
}  // namespace

CheckpointInfo inspect_checkpoint_v2(const std::filesystem::path& root) {
    const auto generation = read_pointer(root);
    return state_info(root / "generations" / generation);
}

CheckpointInfo validate_checkpoint_v2(const std::filesystem::path& root) {
    const auto info = inspect_checkpoint_v2(root);
    try {
        torch::serialize::InputArchive state;
        state.load_from((info.generation / kState).string());
        torch::serialize::InputArchive optimizer;
        optimizer.load_from((info.generation / kOptimizer).string());
        return info;
    } catch (const c10::Error& error) {
        throw CheckpointError(std::string("checkpoint v2 archive is unreadable: ") + error.what());
    }
}

CheckpointInfo migrate_checkpoint_v2(const std::filesystem::path& source,
                                     const std::filesystem::path& destination) {
    const auto info = validate_checkpoint_v2(source);
    if (std::filesystem::exists(destination))
        throw CheckpointError("checkpoint migration destination already exists");

    const auto parent = destination.has_parent_path()
        ? destination.parent_path() : std::filesystem::current_path();
    const auto filename = destination.filename().string();
    if (filename.empty()) throw CheckpointError("checkpoint migration destination is invalid");
    static std::atomic_uint64_t sequence{0};
    const auto staged = parent / ("." + filename + ".checkpoint-migrate-" +
                                  std::to_string(++sequence));
    std::error_code ignored;
    if (std::filesystem::exists(staged))
        throw CheckpointError("checkpoint migration staging path already exists");
    try {
        std::filesystem::create_directories(staged / "generations");
        const auto generation = info.generation.filename();
        std::filesystem::copy(info.generation, staged / "generations" / generation,
                              std::filesystem::copy_options::recursive);
        atomic_write(staged / kPointer, generation.string() + "\n");
        std::filesystem::rename(staged, destination);
        return {destination / "generations" / generation, info.training_step};
    } catch (const std::filesystem::filesystem_error& error) {
        std::filesystem::remove_all(staged, ignored);
        throw CheckpointError(std::string("cannot migrate checkpoint v2: ") + error.what());
    } catch (...) {
        std::filesystem::remove_all(staged, ignored);
        throw;
    }
}

CheckpointInfo save_checkpoint_v2(const std::filesystem::path& root, Trainer& trainer) {
    trainer.compatibility().validate();
    std::filesystem::create_directories(root / "generations");
    static std::atomic_uint64_t sequence{0};
    const std::string generation = "generation-" + std::to_string(trainer.training_step()) + "-" + std::to_string(++sequence);
    const auto staged = root / "generations" / ("." + generation + ".staging");
    const auto committed = root / "generations" / generation;
    std::error_code ignored; std::filesystem::remove_all(staged, ignored); std::filesystem::create_directories(staged);
    try {
        // Native LibTorch archives make this generation self-contained and
        // deliberately distinct from the legacy Python checkpoint reader.
        torch::save(trainer.model(), (staged / kState).string());
        torch::save(trainer.optimizer(), (staged / kOptimizer).string());
        {
            std::ofstream step(staged / "training_step", std::ios::binary | std::ios::trunc);
            step << trainer.training_step();
            if (!step) throw CheckpointError("cannot write checkpoint training_step");
        }
        {
            std::ofstream manifest(staged / "manifest.json", std::ios::binary | std::ios::trunc);
            manifest << "{\"format_version\":2,\"generation\":\"" << generation << "\",\"training_step\":" << trainer.training_step() << "}";
            if (!manifest) throw CheckpointError("cannot write checkpoint manifest");
        }
#ifdef _WIN32
        if (!MoveFileExA(staged.string().c_str(), committed.string().c_str(), MOVEFILE_WRITE_THROUGH)) {
            const DWORD error = GetLastError();
            throw CheckpointError("cannot promote checkpoint generation " + staged.string() + " -> " + committed.string() + " (Win32=" + std::to_string(error) + ")");
        }
#else
        std::filesystem::rename(staged, committed);
#endif
        atomic_write(root / kPointer, generation + "\n");
        return {committed, trainer.training_step()};
    } catch (const CheckpointError&) { std::filesystem::remove_all(staged, ignored); throw; }
      catch (const c10::Error& error) { std::filesystem::remove_all(staged, ignored); throw CheckpointError(std::string("cannot save checkpoint v2: ") + error.what()); }
      catch (const std::filesystem::filesystem_error& error) { std::filesystem::remove_all(staged, ignored); throw CheckpointError(error.what()); }
}

CheckpointInfo load_checkpoint_v2_weights(const std::filesystem::path& root,
                                          diamond_model::DiamondModel model) {
    const auto info = inspect_checkpoint_v2(root);
    if (!model) throw CheckpointError("checkpoint model destination is empty");
    try {
        torch::load(model, (info.generation / kState).string());
        return info;
    } catch (const c10::Error& error) {
        throw CheckpointError(std::string("cannot load checkpoint model weights: ") + error.what());
    }
}

CheckpointInfo load_checkpoint_v2(const std::filesystem::path& root, Trainer& trainer) {
    const auto info = load_checkpoint_v2_weights(root, trainer.model());
    try {
        torch::load(trainer.optimizer(), (info.generation / kOptimizer).string());
        trainer.restore_checkpoint_state(trainer.config(), info.training_step);
        return info;
    } catch (const c10::Error& error) {
        throw CheckpointError(std::string("cannot load checkpoint optimizer state: ") + error.what());
    }
}

}  // namespace diamond_training
