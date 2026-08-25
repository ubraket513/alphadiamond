#pragma once

#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include "diamond_orchestration/rating.hpp"

namespace diamond_orchestration {

// Canonical, replayable registry persistence. Schema v2 retains full protocol
// configuration and participant identity; schema v1 remains readable.
void save_rating_registry(const std::filesystem::path& path, const RatingRegistry& registry);
RatingRegistry load_rating_registry(const std::filesystem::path& path);

enum class RatingStoreFailpoint { before_event_commit, before_receipt_commit };
void set_rating_store_failpoint_for_testing(
    std::function<void(RatingStoreFailpoint)> failpoint);

// Immutable local event/receipt outbox. Events publish once; receipts acknowledge
// durable delivery separately so pending work survives process interruption.
class RatingEventOutbox final {
  public:
    explicit RatingEventOutbox(std::filesystem::path root);

    bool publish(const SooRatingEvent& event);
    bool publish(const MinRatingEvent& event);
    void write_receipt(const std::string& event_id, diamond_support::JsonValue receipt);
    std::vector<diamond_support::JsonValue> pending_events() const;

  private:
    std::filesystem::path root_;
};

}  // namespace diamond_orchestration
