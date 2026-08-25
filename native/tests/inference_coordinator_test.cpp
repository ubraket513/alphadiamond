#include <chrono>
#include <filesystem>
#include <stop_token>
#include <string>

#include "check.hpp"
#include "diamond_model/soo_model.hpp"
#include "diamond_pipeline/model_pool.hpp"
#include "diamond_training/device.hpp"
#include "diamond_training/trainer.hpp"

namespace {

diamond_training::Compatibility compatibility() {
    return diamond_training::Compatibility::soo(
        "1.0.0", {.residual_blocks = 1, .width = 8});
}

}  // namespace

int main(int argc, char** argv) {
    REQUIRE(argc == 2, "usage: inference_coordinator_test <scratch>");
    const auto scratch = std::filesystem::path(argv[1]);
    std::filesystem::remove_all(scratch);

    const auto device = diamond_training::resolve_device("cpu");
    const auto resident_compatibility = compatibility();
    diamond_pipeline::ModelPool pool(2, device);
    auto source = diamond_model::DiamondModel(8, 1, 4, 1);
    {
        const auto source_holder_alias = source;
        CHECK(source_holder_alias->parameters().front().data_ptr() ==
              source->parameters().front().data_ptr());
    }
    const auto first = pool.install(resident_compatibility, source);
    const auto second = pool.install(resident_compatibility, diamond_model::DiamondModel(8, 1, 4, 1));
    CHECK_EQ(pool.resident_count(), std::size_t{2});
    CHECK(first != second);
    pool.activate(first);
    CHECK(pool.active_key() == first);

    const auto actor = pool.active_model();
    CHECK(first.model_name == resident_compatibility.model_name);
    CHECK(first.model_version == resident_compatibility.model_version);
    CHECK(first.checkpoint_sha256 == diamond_training::canonical_model_digest(actor));
    CHECK(actor->parameters().front().data_ptr() != source->parameters().front().data_ptr());
    CHECK(actor->adjacency.data_ptr() != source->adjacency.data_ptr());
    CHECK(actor->parameters().front().device() == device.torch_device);
    CHECK(actor->adjacency.device() == device.torch_device);
    CHECK(!actor->is_training());
    for (const auto& parameter : actor->parameters()) CHECK(!parameter.requires_grad());

    const auto actor_first_parameter = actor->parameters().front().detach().clone();
    const auto actor_adjacency = actor->adjacency.detach().clone();
    {
        torch::NoGradGuard no_grad;
        source->parameters().front().add_(1.0F);
        source->adjacency.add_(1.0F);
    }
    source = nullptr;
    CHECK(torch::equal(actor->parameters().front(), actor_first_parameter));
    CHECK(torch::equal(actor->adjacency, actor_adjacency));
    CHECK(pool.active_key() == first);
    CHECK(first.checkpoint_sha256 == diamond_training::canonical_model_digest(pool.active_model()));

    auto incompatible = resident_compatibility;
    incompatible.model_version = "2.0.0";
    bool saw_incompatible = false;
    try {
        pool.require_compatible(incompatible);
    } catch (const diamond_pipeline::IncompatibleCheckpointError&) {
        saw_incompatible = true;
    }
    CHECK(saw_incompatible);

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
        (void)pool.install_checkpoint(resident_compatibility, scratch / "legacy.pt",
                                      diamond_model::DiamondModel(8, 1, 4, 1));
    } catch (const diamond_pipeline::IncompatibleCheckpointError&) {
        saw_legacy = true;
    }
    CHECK(saw_legacy);
    return soo_test::report("inference_coordinator_test");
}
