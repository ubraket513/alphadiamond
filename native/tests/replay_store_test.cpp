#include <filesystem>

#include "check.hpp"
#include "diamond_pipeline/replay_store.hpp"

int main(int argc, char** argv) {
    REQUIRE(argc == 2, "usage: replay_store_test <scratch-dir>");
    const diamond_pipeline::Compatibility compatibility{"soo", "1.2.3"};
    diamond_pipeline::ReplayStore store(std::filesystem::path(argv[1]), compatibility, 8, 3);
    CHECK(store.ingest({}) == 0U);
    return soo_test::report("replay_store_test");
}
