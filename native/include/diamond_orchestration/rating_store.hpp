#pragma once

#include <filesystem>

#include "diamond_orchestration/rating.hpp"

namespace diamond_orchestration {

// Canonical, replayable registry persistence. The file contains the protocol,
// registered participants and every canonical event; loading rebuilds ratings
// through the same registry APIs used for live arena events.
void save_rating_registry(const std::filesystem::path& path, const RatingRegistry& registry);
RatingRegistry load_rating_registry(const std::filesystem::path& path);

}  // namespace diamond_orchestration
