#include "diamond_orchestration/rating_store.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}
} // namespace

int main() {
    const auto root =
        std::filesystem::temp_directory_path() / "alphadiamond-rating-store-transaction";
    try {
        using namespace diamond_orchestration;
        std::error_code ignored;
        std::filesystem::remove_all(root, ignored);

        EloConfig config;
        config.initial_rating = 1234.0;
        config.k_factor = 17.0;
        RatingRegistry registry{"sha256:rating-v2", config};
        const auto candidate =
            make_participant_identity(diamond_support::JsonValue{diamond_support::JsonValue::Object{
                                          {"checkpoint", diamond_support::JsonValue{"candidate"}}}},
                                      "Candidate");
        const auto baseline =
            make_participant_identity(diamond_support::JsonValue{diamond_support::JsonValue::Object{
                                          {"checkpoint", diamond_support::JsonValue{"baseline"}}}},
                                      "Baseline");
        registry.add_participant(candidate);
        registry.add_participant(baseline);
        const auto event = make_soo_rating_event(
            77, "sha256:rating-v2", {candidate.participant_id, baseline.participant_id}, {1, 2},
            {1, 2}, "opening", true, candidate.participant_id, baseline.participant_id,
            "stable-game");
        require(registry.record_event(event), "v2 event accepted");
        const auto registry_path = root / "registry.json";
        save_rating_registry(registry_path, registry);
        const auto restored = load_rating_registry(registry_path);
        require(restored.elo_config().initial_rating == 1234.0 &&
                    restored.elo_config().k_factor == 17.0,
                "schema v2 retains validated protocol config");
        require(restored.events().size() == 1, "schema v2 retains event");

        RatingEventOutbox outbox{root / "outbox"};
        require(outbox.publish(event), "first immutable publish succeeds");
        require(!outbox.publish(event), "byte-identical immutable publish is idempotent");
        require(outbox.pending_events().size() == 1, "event without receipt is pending");
        set_rating_store_failpoint_for_testing([](RatingStoreFailpoint point) {
            if (point == RatingStoreFailpoint::before_receipt_commit)
                throw RatingError("receipt failpoint");
        });
        bool receipt_failed = false;
        try {
            outbox.write_receipt(event.event_id,
                                 diamond_support::JsonValue{diamond_support::JsonValue::Object{}});
        } catch (const RatingError&) {
            receipt_failed = true;
        }
        require(receipt_failed && outbox.pending_events().size() == 1,
                "receipt failpoint preserves pending event");
        set_rating_store_failpoint_for_testing({});
        outbox.write_receipt(event.event_id,
                             diamond_support::JsonValue{diamond_support::JsonValue::Object{}});
        require(outbox.pending_events().empty(), "durable receipt clears pending event");

        const auto collision = root / "outbox" / "events" / (event.event_id.substr(7) + ".json");
        {
            std::ofstream output(collision, std::ios::binary | std::ios::trunc);
            output << "different bytes";
        }
        bool collision_rejected = false;
        try {
            outbox.publish(event);
        } catch (const RatingError&) {
            collision_rejected = true;
        }
        require(collision_rejected, "different immutable bytes with same event ID fail");

        const auto deferred = make_soo_rating_event(
            1, "sha256:rating-v2", {candidate.participant_id, baseline.participant_id}, {1, 2},
            {1, 2}, "opening-2", true, candidate.participant_id, baseline.participant_id,
            "stable-game-2");
        set_rating_store_failpoint_for_testing([](RatingStoreFailpoint point) {
            if (point == RatingStoreFailpoint::before_event_commit)
                throw RatingError("event failpoint");
        });
        bool event_failed = false;
        try {
            outbox.publish(deferred);
        } catch (const RatingError&) {
            event_failed = true;
        }
        set_rating_store_failpoint_for_testing({});
        require(event_failed && outbox.pending_events().empty(), "event failpoint commits nothing");
        std::filesystem::remove_all(root, ignored);
    } catch (const std::exception& error) {
        std::cerr << "rating_store_transaction_test: " << error.what() << '\n';
        std::error_code ignored;
        std::filesystem::remove_all(root, ignored);
        return 1;
    }
}
