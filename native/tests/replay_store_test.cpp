#include <filesystem>
#include <string>

#include "check.hpp"
#include "diamond_pipeline/replay_store.hpp"
#include "diamond_support/json.hpp"

int main(int argc, char** argv) {
    REQUIRE(argc == 3, "usage: replay_store_test <fixture-dir> <scratch-dir>");
    const diamond_pipeline::Compatibility compatibility{"soo", "1.2.3"};
    const auto fixtures = std::filesystem::path(argv[1]);
    const auto scratch = std::filesystem::path(argv[2]);
    std::filesystem::remove_all(scratch);
    std::filesystem::create_directories(scratch);
    CHECK_EQ(diamond_support::sha256(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK_EQ(diamond_support::sha256("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    CHECK_EQ(diamond_support::sha256("game-completed"), "ef9b00a3d875a4e0a49ed5de9f4e52b7eb89bb1bc8e12dae7f3d965ed43e1afc");
    bool corrupt = false;
    std::filesystem::copy(fixtures / "corrupt-digest", scratch / "corrupt-digest", std::filesystem::copy_options::recursive);
    try { diamond_pipeline::ReplayStore store(scratch / "corrupt-digest", compatibility, 8, 3); (void)store.sample(1); }
    catch (const std::runtime_error&) { corrupt = true; }
    CHECK(corrupt);
    std::filesystem::copy(fixtures / "capacity-prune", scratch / "capacity-prune", std::filesystem::copy_options::recursive);
    diamond_pipeline::ReplayStore capacity(scratch / "capacity-prune", compatibility, 3, 3);
    const auto rows = capacity.sample(3);
    REQUIRE(rows.size() == 3, "capacity fixture retains only reachable rows");
    std::filesystem::copy(fixtures / "rollback", scratch / "rollback", std::filesystem::copy_options::recursive);
    diamond_pipeline::ReplayStore rollback(scratch / "rollback", compatibility, 8, 3);
    rollback.restore_manifest(scratch / "rollback" / "persistent-replay-v1" / "Soo" / "3f3372c174dba4b7bfa9288e2c7e0a33e284dfdc3313f1d073210de8e47df229" / "before.json");
    CHECK_EQ(rollback.sample(1).size(), 1U);
    std::filesystem::remove_all(scratch);
    return soo_test::report("replay_store_test");
}
