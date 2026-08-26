#pragma once

#include <string>

namespace diamond_support {

inline std::string build_provenance_json() {
#ifdef DIAMOND_SOURCE_GIT_COMMIT
    constexpr const char* source_commit = DIAMOND_SOURCE_GIT_COMMIT;
#else
    constexpr const char* source_commit = "unavailable";
#endif
#ifdef DIAMOND_SOURCE_DIRTY
    constexpr const char* dirty = DIAMOND_SOURCE_DIRTY != 0 ? "true" : "false";
#else
    constexpr const char* dirty = "\"unavailable\"";
#endif
    return std::string{"{\"source_commit\":\""} + source_commit + "\",\"dirty\":" + dirty + "}";
}

} // namespace diamond_support
