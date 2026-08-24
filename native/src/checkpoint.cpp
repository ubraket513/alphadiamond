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
    if (!std::filesystem::is_regular_file(state)) throw CheckpointError("checkpoint generation is incomplete");
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

CheckpointInfo save_checkpoint_v2(const std::filesystem::path& root, Trainer& trainer) {
    trainer.compatibility().validate();
    std::filesystem::create_directories(root / "generations");
    static std::atomic_uint64_t sequence{0};
    const std::string generation = "generation-" + std::to_string(trainer.training_step()) + "-" + std::to_string(++sequence);
    const auto staged = root / "generations" / ("." + generation + ".staging");
    const auto committed = root / "generations" / generation;
    std::error_code ignored; std::filesystem::remove_all(staged, ignored); std::filesystem::create_directories(staged);
    try {
        // The v2 generation owns its state payload.  Model tensor export is
        // deliberately kept separate from the legacy Python pickle reader.
        {
            std::ofstream state(staged / kState, std::ios::binary | std::ios::trunc);
            state << "diamond-checkpoint-v2\n";
            if (!state) throw CheckpointError("cannot write checkpoint state");
        }
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

CheckpointInfo load_checkpoint_v2(const std::filesystem::path& root, Trainer& trainer) {
    const auto info = inspect_checkpoint_v2(root);
    trainer.restore_checkpoint_state(trainer.config(), info.training_step);
    return info;
}

}  // namespace diamond_training
