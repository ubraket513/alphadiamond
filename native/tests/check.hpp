// Test assertions that survive NDEBUG.  <cassert> compiles away in a release
// build, which would turn every native test into an expensive no-op.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>

namespace soo_test {

inline int& failures() {
    static int count = 0;
    return count;
}

inline void fail(const char* file, int line, const std::string& what) {
    std::fprintf(stderr, "%s:%d: FAIL %s\n", file, line, what.c_str());
    if (++failures() >= 20) {
        std::fprintf(stderr, "too many failures; stopping\n");
        std::exit(1);
    }
}

inline int report(const char* name) {
    if (failures() == 0) {
        std::fprintf(stderr, "%s: ok\n", name);
        return 0;
    }
    std::fprintf(stderr, "%s: %d failure(s)\n", name, failures());
    return 1;
}

}  // namespace soo_test

#define CHECK(condition)                                                      \
    do {                                                                      \
        if (!(condition)) soo_test::fail(__FILE__, __LINE__, #condition);     \
    } while (false)

#define CHECK_EQ(actual, expected)                                            \
    do {                                                                      \
        const auto _actual = (actual);                                        \
        const auto _expected = (expected);                                    \
        if (!(_actual == _expected)) {                                        \
            soo_test::fail(__FILE__, __LINE__,                                \
                           std::string(#actual) + " != " + #expected);        \
        }                                                                     \
    } while (false)

#define REQUIRE(condition, message)                                           \
    do {                                                                      \
        if (!(condition)) {                                                   \
            std::fprintf(stderr, "%s:%d: FATAL %s\n", __FILE__, __LINE__,     \
                         (message));                                          \
            std::exit(1);                                                     \
        }                                                                     \
    } while (false)
