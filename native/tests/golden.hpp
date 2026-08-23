// Reader for tests/golden/rules-v1.txt.  The format is deliberately trivial --
// whitespace-separated tokens, no JSON -- so that a build with no dependencies
// can consume the Python oracle's frozen answers.
#pragma once

#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "soo/state.hpp"

namespace soo_test {

struct GoldenCase {
    std::string tag;
    int player_count = 0;
    soo::State state;
    int action_count = 0;
    uint64_t physical_fnv = 0;
    uint64_t canonical_fnv = 0;
    uint64_t successor_fnv = 0;
    uint64_t encoded_fnv = 0;
    int prior_count = 0;
    double prior_max = 0.0;
    double prior_dot = 0.0;
};

struct Golden {
    std::vector<soo::Match> matches;  // indexed by player count
    std::vector<GoldenCase> cases;

    const soo::Match& match(int player_count) const { return matches.at(player_count); }
};

// FNV-1a 64, byte for byte the same as tools/build_golden.py.
class Fnv {
  public:
    void byte(uint8_t value) {
        hash_ ^= value;
        hash_ *= 1099511628211ULL;
    }
    void bytes(const void* data, std::size_t count) {
        const auto* first = static_cast<const uint8_t*>(data);
        for (std::size_t i = 0; i < count; ++i) byte(first[i]);
    }
    void i32(int32_t value) { bytes(&value, sizeof(value)); }
    void f32(float value) { bytes(&value, sizeof(value)); }
    uint64_t value() const { return hash_; }

  private:
    uint64_t hash_ = 14695981039346656037ULL;  // FNV-1a 64 offset basis
};

inline bool load_golden(const std::string& path, Golden& out, std::string& error) {
    std::ifstream file(path);
    if (!file) {
        error = "cannot open " + path;
        return false;
    }
    out.matches.assign(soo::kMaxPlayers + 1, soo::Match{});

    std::string line;
    GoldenCase pending;
    bool have_pending = false;
    while (std::getline(file, line)) {
        // A CRLF checkout would otherwise leave a carriage return on the
        // last token of every line.
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
            line.pop_back();
        }
        if (line.empty() || line[0] == '#') continue;
        std::istringstream tokens(line);
        std::string kind;
        tokens >> kind;

        if (kind == "match") {
            int player_count = 0;
            tokens >> player_count;
            soo::Match match;
            std::string seat;
            while (tokens >> seat) {
                std::istringstream fields(seat);
                std::string field;
                int values[3] = {0, 0, 0};
                int count = 0;
                while (count < 3 && std::getline(fields, field, ',')) {
                    values[count++] = std::stoi(field);
                }
                if (count != 3) {
                    error = "bad seat: " + seat;
                    return false;
                }
                match.players[match.count++] = {static_cast<uint8_t>(values[0]),
                                                static_cast<uint8_t>(values[1]),
                                                static_cast<uint8_t>(values[2])};
            }
            out.matches.at(player_count) = match;
            continue;
        }

        if (kind == "pos") {
            int current = 0;
            int turn = 0;
            int status = 0;
            std::string finish;
            std::string occupancy;
            pending = GoldenCase{};
            tokens >> pending.tag >> pending.player_count >> current >> turn >> status >> finish >>
                occupancy;
            if (occupancy.size() != static_cast<std::size_t>(soo::kBoardSize)) {
                error = "bad occupancy for " + pending.tag;
                return false;
            }
            for (int i = 0; i < soo::kBoardSize; ++i) {
                pending.state.occupancy[i] = static_cast<uint8_t>(occupancy[i] - '0');
            }
            pending.state.current_player = static_cast<uint8_t>(current);
            pending.state.turn_number = static_cast<uint16_t>(turn);
            pending.state.status = static_cast<uint8_t>(status);
            if (finish != "-") {
                std::istringstream ids(finish);
                std::string id;
                while (std::getline(ids, id, ',')) {
                    pending.state.finish_order[pending.state.finished_count++] =
                        static_cast<uint8_t>(std::stoi(id));
                }
            }
            have_pending = true;
            continue;
        }

        if (kind == "exp") {
            if (!have_pending) {
                error = "exp line without a preceding pos line";
                return false;
            }
            std::string physical;
            std::string canonical;
            std::string successor;
            std::string encoded;
            tokens >> pending.action_count >> physical >> canonical >> successor >> encoded >>
                pending.prior_count >> pending.prior_max >> pending.prior_dot;
            pending.physical_fnv = std::stoull(physical, nullptr, 16);
            pending.canonical_fnv = std::stoull(canonical, nullptr, 16);
            pending.successor_fnv = std::stoull(successor, nullptr, 16);
            pending.encoded_fnv = std::stoull(encoded, nullptr, 16);
            out.cases.push_back(pending);
            have_pending = false;
            continue;
        }

        error = "unknown record: " + kind;
        return false;
    }

    if (out.cases.empty()) {
        error = "golden file has no cases";
        return false;
    }
    return true;
}

}  // namespace soo_test
