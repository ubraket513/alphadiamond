#include <chrono>
#include <filesystem>
#include <stop_token>
#include <string>

#include "check.hpp"
#include "diamond_model/soo_model.hpp"
#include "diamond_pipeline/model_pool.hpp"

namespace {

diamond_pipeline::ModelKey key(std::string digest) {
    return {"Soo", "1.0.0", std::move(digest)};
}

}  // namespace

int main(int argc, char** argv) {
    REQUIRE(argc == 2, "usage: inference_coordinator_test <scratch>");
    const auto scratch = std::filesystem::path(argv[1]);
    std::filesystem::remove_all(scratch);

    diamond_pipeline::ModelPool pool(2);
    const auto first = key(std::string(64, 'a'));
    const auto second = key(std::string(64, 'b'));
    pool.install(first, diamond_model::DiamondModel(8, 1, 4, 1));
    pool.install(second, diamond_model::DiamondModel(8, 1, 4, 1));
    CHECK_EQ(pool.resident_count(), std::size_t{2});
    pool.activate(first);
    CHECK(pool.active_key() == first);

    std::stop_source cancelled;
    cancelled.request_stop();
    bool saw_cancel = false;
    try {
        pool.require_ready(cancelled.get_token(), std::chrono::steady_clock::now() + std::chrono::seconds(1));
    } catch (const diamond_pipeline::CancelledError&) {
        saw_cancel = true;
    }
    CHECK(saw_cancel);

    bool saw_deadline = false;
    try {
        pool.require_ready({}, std::chrono::steady_clock::now() - std::chrono::seconds(1));
    } catch (const diamond_pipeline::DeadlineExceededError&) {
        saw_deadline = true;
    }
    CHECK(saw_deadline);

    bool saw_legacy = false;
    try {
        pool.install_checkpoint(key(std::string(64, 'c')), scratch / "legacy.pt",
                                diamond_model::DiamondModel(8, 1, 4, 1));
    } catch (const diamond_pipeline::IncompatibleCheckpointError&) {
        saw_legacy = true;
    }
    CHECK(saw_legacy);
    return soo_test::report("inference_coordinator_test");
}
