#include <filesystem>
#include <fstream>
#include <string>

#include "check.hpp"
#include "diamond_training/checkpoint.hpp"

namespace {

template <typename Operation>
bool rejects(Operation&& operation) {
    try {
        operation();
        return false;
    } catch (const diamond_training::CheckpointError&) {
        return true;
    }
}

void write_file(const std::filesystem::path& path, const std::string& contents) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << contents;
}

}  // namespace

int main(int argc, char** argv) {
    REQUIRE(argc == 2, "usage: checkpoint_v2_reject_test <scratch>");
    const auto scratch = std::filesystem::path(argv[1]);
    std::filesystem::remove_all(scratch);
    std::filesystem::create_directories(scratch);

    write_file(scratch / "legacy.pt", "v1");
    CHECK(rejects([&] {
        (void)diamond_training::inspect_checkpoint_v2(scratch / "legacy.pt");
    }));

    const auto invalid_pointer = scratch / "invalid-pointer";
    write_file(invalid_pointer / "CURRENT", "../escape\n");
    CHECK(rejects([&] {
        (void)diamond_training::inspect_checkpoint_v2(invalid_pointer);
    }));

    const auto incomplete = scratch / "incomplete";
    write_file(incomplete / "CURRENT", "generation-1\n");
    write_file(incomplete / "generations" / "generation-1" / "state.pt", "state");
    write_file(incomplete / "generations" / "generation-1" / "training_step", "1\n");
    CHECK(rejects([&] {
        (void)diamond_training::inspect_checkpoint_v2(incomplete);
    }));

    const auto invalid_step = scratch / "invalid-step";
    write_file(invalid_step / "CURRENT", "generation-1\n");
    write_file(invalid_step / "generations" / "generation-1" / "state.pt", "state");
    write_file(invalid_step / "generations" / "generation-1" / "optimizer.pt", "optimizer");
    write_file(invalid_step / "generations" / "generation-1" / "training_step", "not-a-step\n");
    CHECK(rejects([&] {
        (void)diamond_training::inspect_checkpoint_v2(invalid_step);
    }));

    const auto unreadable = scratch / "unreadable";
    write_file(unreadable / "CURRENT", "generation-1\n");
    write_file(unreadable / "generations" / "generation-1" / "state.pt", "state");
    write_file(unreadable / "generations" / "generation-1" / "optimizer.pt", "optimizer");
    write_file(unreadable / "generations" / "generation-1" / "training_step", "1\n");
    CHECK(rejects([&] {
        (void)diamond_training::validate_checkpoint_v2(unreadable);
    }));

    std::filesystem::remove_all(scratch);
    return soo_test::report("checkpoint_v2_reject_test");
}
