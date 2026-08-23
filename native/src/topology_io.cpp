// Loading the exported topology tables from disk.  The tables themselves are
// generated from the Python board (src/diamond/alphazero/native/topology.py);
// this file only reads the exported bytes, so it is not a second authority.
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <vector>

#include "soo/board.hpp"

namespace soo {
namespace {

bool read_exact(const std::string& path, char* out, std::size_t bytes) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;
    file.read(out, static_cast<std::streamsize>(bytes));
    if (file.gcount() != static_cast<std::streamsize>(bytes)) return false;
    // A longer file means the exporter and this reader disagree about the
    // table's shape, which is exactly the drift the gates exist to catch.
    return file.peek() == std::ifstream::traits_type::eof();
}

bool read_i32(const std::string& path, std::vector<int32_t>& values) {
    return read_exact(path, reinterpret_cast<char*>(values.data()),
                      values.size() * sizeof(int32_t));
}

}  // namespace

bool load_topology_from_dir(const std::string& root) {
    Topology loaded;
    if (!read_exact(root + "/topology_neighbour.i8",
                    reinterpret_cast<char*>(loaded.neighbour.data()),
                    sizeof(loaded.neighbour))) {
        return false;
    }

    std::vector<int32_t> camps(kCamps * kCampSize);
    std::vector<int32_t> pairwise(kBoardSize * kBoardSize);
    std::vector<int32_t> physical(kCamps * kBoardSize);
    std::vector<int32_t> canonical(kCamps * kBoardSize);
    if (!read_i32(root + "/topology_camp_positions.i32", camps) ||
        !read_i32(root + "/topology_pairwise_distance.i32", pairwise) ||
        !read_i32(root + "/topology_physical_to_canonical.i32", physical) ||
        !read_i32(root + "/topology_canonical_to_physical.i32", canonical)) {
        return false;
    }

    for (int camp = 0; camp < kCamps; ++camp) {
        for (int index = 0; index < kCampSize; ++index) {
            loaded.camp_positions[camp][index] =
                static_cast<uint8_t>(camps[camp * kCampSize + index]);
        }
        for (int position = 0; position < kBoardSize; ++position) {
            loaded.physical_to_canonical[camp][position] =
                static_cast<uint8_t>(physical[camp * kBoardSize + position]);
            loaded.canonical_to_physical[camp][position] =
                static_cast<uint8_t>(canonical[camp * kBoardSize + position]);
        }
    }
    for (int row = 0; row < kBoardSize; ++row) {
        for (int column = 0; column < kBoardSize; ++column) {
            loaded.pairwise[row][column] =
                static_cast<uint8_t>(pairwise[row * kBoardSize + column]);
        }
    }

    loaded.configured = true;
    mutable_topology() = loaded;
    return true;
}

}  // namespace soo
