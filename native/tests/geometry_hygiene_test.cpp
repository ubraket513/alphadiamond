// No seat geometry outside the factory.
//
// The Soo geometry bug was not a rules fault -- the rules engine was parity
// correct throughout -- it was orchestration duplication. Five call sites each
// wrote the seat triples out by hand, they drifted, and the trainer ended up
// playing a different game from the Qt application. Pinning the factory to the
// golden fixture (match_geometry_test) proves the factory is right; it says
// nothing about a call site that bypasses it.
//
// So this scans the sources for a hand-written PlayerSpec triple and fails on
// any outside the factory's own definition. It is a repository contract, not a
// unit test: the property it protects is "there is one authority", which no
// amount of testing the authority itself can establish.
#include <filesystem>
#include <fstream>
#include <regex>
#include <string>
#include <vector>

#include "check.hpp"

namespace {

// The file that is allowed to contain them: the factory itself.
constexpr std::string_view kFactory = "board.cpp";

// Sources that describe a *fixture* rather than a runtime match are exempt.
// The golden tests must be able to state the geometry independently, or they
// could not detect the factory drifting.
bool exempt(const std::filesystem::path& path) {
    const std::string name = path.filename().string();
    return name == kFactory || name == "match_geometry_test.cpp" ||
           name == "geometry_hygiene_test.cpp" || name == "golden.hpp" ||
           name == "rules_golden_test.cpp" || name == "mcts_golden_test.cpp" ||
           name == "mcts3p_golden_test.cpp" || name == "budget_test.cpp" ||
           name == "soo_mcts_probe.cpp";
}

}  // namespace

int main(int argc, char** argv) {
    REQUIRE(argc >= 2, "usage: geometry_hygiene_test <native-source-root>");
    const std::filesystem::path root = argv[1];
    REQUIRE(std::filesystem::exists(root), "native source root does not exist");

    // `players[N] = {id, camp, target}` or `PlayerSpec{id, camp, target}`.
    const std::regex triple(
        R"((players\s*\[\s*\d+\s*\]\s*=\s*\{\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\})"
        R"(|PlayerSpec\s*\{\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\}))");

    std::vector<std::string> offences;
    bool saw_factory = false;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root)) {
        if (!entry.is_regular_file()) continue;
        const auto& path = entry.path();
        const std::string extension = path.extension().string();
        if (extension != ".cpp" && extension != ".hpp") continue;
        if (path.filename().string() == kFactory) saw_factory = true;
        if (exempt(path)) continue;

        std::ifstream input(path);
        REQUIRE(static_cast<bool>(input), "cannot read a native source file");
        std::string line;
        for (int number = 1; std::getline(input, line); ++number) {
            if (!std::regex_search(line, triple)) continue;
            offences.push_back(path.filename().string() + ":" + std::to_string(number));
        }
    }

    // Guard against the scan silently covering nothing.
    REQUIRE(saw_factory, "the scan never reached board.cpp; the source root is wrong");

    for (const auto& offence : offences) {
        soo_test::fail(__FILE__, __LINE__,
                       offence + ": seat geometry written at a call site; use "
                                 "soo::standard_soo_match() or soo::standard_min_match()");
    }

    return soo_test::report("geometry_hygiene_test");
}
