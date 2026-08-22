// pybind11 module ``diamond_native``.
//
// Phase 1 exposes only what Gate A needs: topology injection, state, rules,
// the canonical encoder and the vacancy bootstrap prior.  MCTS, the batcher
// and the lane runner arrive in Phase 2.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>
#include <string>
#include <vector>

#include "soo/action.hpp"
#include "soo/board.hpp"
#include "soo/encoder.hpp"
#include "soo/prior.hpp"
#include "soo/rules.hpp"
#include "soo/state.hpp"

namespace py = pybind11;
using namespace soo;

namespace {

template <typename T>
void fill_row(const py::handle& source, T* out, size_t expected, const char* what) {
    const py::sequence row = py::cast<py::sequence>(source);
    if (static_cast<size_t>(py::len(row)) != expected) {
        throw std::invalid_argument(std::string(what) + ": unexpected row length");
    }
    for (size_t i = 0; i < expected; ++i) {
        out[i] = py::cast<T>(row[i]);
    }
}

void configure(const py::dict& tables) {
    if (py::cast<int>(tables["board_size"]) != kBoardSize) {
        throw std::invalid_argument("native extension is built for a 73-hole board");
    }
    if (py::cast<int>(tables["directions"]) != kDirections) {
        throw std::invalid_argument("native extension is built for 6 lattice directions");
    }
    Topology& topo = mutable_topology();

    const py::sequence neighbour = py::cast<py::sequence>(tables["neighbour"]);
    for (int position = 0; position < kBoardSize; ++position) {
        fill_row<int8_t>(neighbour[position], topo.neighbour[position].data(), kDirections,
                         "neighbour");
    }
    const py::sequence camps = py::cast<py::sequence>(tables["camp_positions"]);
    for (int camp = 0; camp < kCamps; ++camp) {
        fill_row<uint8_t>(camps[camp], topo.camp_positions[camp].data(), kCampSize,
                          "camp_positions");
    }
    const py::sequence pairwise = py::cast<py::sequence>(tables["pairwise_distance"]);
    for (int position = 0; position < kBoardSize; ++position) {
        fill_row<uint8_t>(pairwise[position], topo.pairwise[position].data(), kBoardSize,
                          "pairwise_distance");
    }
    const py::sequence forward = py::cast<py::sequence>(tables["physical_to_canonical"]);
    const py::sequence inverse = py::cast<py::sequence>(tables["canonical_to_physical"]);
    for (int camp = 0; camp < kCamps; ++camp) {
        fill_row<uint8_t>(forward[camp], topo.physical_to_canonical[camp].data(), kBoardSize,
                          "physical_to_canonical");
        fill_row<uint8_t>(inverse[camp], topo.canonical_to_physical[camp].data(), kBoardSize,
                          "canonical_to_physical");
    }
    topo.configured = true;
}

py::dict export_tables() {
    const Topology& topo = topology();
    py::dict out;
    out["board_size"] = kBoardSize;
    out["directions"] = kDirections;
    py::list neighbour;
    for (const auto& row : topo.neighbour) {
        neighbour.append(py::cast(std::vector<int>(row.begin(), row.end())));
    }
    out["neighbour"] = neighbour;
    py::list camps;
    for (const auto& row : topo.camp_positions) {
        camps.append(py::cast(std::vector<int>(row.begin(), row.end())));
    }
    out["camp_positions"] = camps;
    py::list pairwise;
    for (const auto& row : topo.pairwise) {
        pairwise.append(py::cast(std::vector<int>(row.begin(), row.end())));
    }
    out["pairwise_distance"] = pairwise;
    py::list forward;
    py::list inverse;
    for (int camp = 0; camp < kCamps; ++camp) {
        forward.append(py::cast(std::vector<int>(topo.physical_to_canonical[camp].begin(),
                                                 topo.physical_to_canonical[camp].end())));
        inverse.append(py::cast(std::vector<int>(topo.canonical_to_physical[camp].begin(),
                                                 topo.canonical_to_physical[camp].end())));
    }
    out["physical_to_canonical"] = forward;
    out["canonical_to_physical"] = inverse;
    return out;
}

State make_state(const std::vector<int>& occupancy, int current_player, int turn_number,
                 int status, const std::vector<int>& finish_order) {
    if (occupancy.size() != static_cast<size_t>(kBoardSize)) {
        throw std::invalid_argument("occupancy must have 73 entries");
    }
    if (finish_order.size() > static_cast<size_t>(kMaxPlayers)) {
        throw std::invalid_argument("finish_order is longer than the seat count");
    }
    State state;
    for (int i = 0; i < kBoardSize; ++i) state.occupancy[i] = static_cast<uint8_t>(occupancy[i]);
    state.current_player = static_cast<uint8_t>(current_player);
    state.turn_number = static_cast<uint16_t>(turn_number);
    state.status = static_cast<uint8_t>(status);
    for (size_t i = 0; i < finish_order.size(); ++i) {
        state.finish_order[i] = static_cast<uint8_t>(finish_order[i]);
    }
    state.finished_count = static_cast<uint8_t>(finish_order.size());
    return state;
}

std::vector<int> finish_order_of(const State& state) {
    return std::vector<int>(state.finish_order.begin(),
                            state.finish_order.begin() + state.finished_count);
}

// A match plus every Gate A operation, so Python touches one handle.
class Game {
  public:
    explicit Game(const std::vector<std::array<int, 3>>& players) {
        if (players.size() < 2 || players.size() > static_cast<size_t>(kMaxPlayers)) {
            throw std::invalid_argument("a match needs 2 or 3 seats");
        }
        for (size_t seat = 0; seat < players.size(); ++seat) {
            match_.players[seat] = PlayerSpec{static_cast<uint8_t>(players[seat][0]),
                                              static_cast<uint8_t>(players[seat][1]),
                                              static_cast<uint8_t>(players[seat][2])};
        }
        match_.count = static_cast<uint8_t>(players.size());
    }

    std::vector<int32_t> legal_action_ids(const State& state) const {
        std::vector<int32_t> out;
        out.reserve(64);
        soo::legal_action_ids(state, out);
        return out;
    }

    std::vector<int32_t> canonical_legal_action_ids(const State& state) const {
        std::vector<int32_t> out;
        out.reserve(64);
        soo::canonical_legal_action_ids(state, match_, out);
        return out;
    }

    State apply_action(const State& state, int32_t action) const {
        return soo::apply_action(state, match_, action);
    }

    State apply_canonical_action(const State& state, int32_t action) const {
        return soo::apply_action(
            state, match_, soo::to_physical_action(action, match_, state.current_player));
    }

    bool is_terminal(const State& state) const { return state.status == kFinished; }

    int search_current_player(const State& state) const {
        return soo::search_current_player(state, match_);
    }

    py::tuple encode(const State& state) const {
        const Encoded encoded = soo::encode(state, match_);
        py::list rows;
        for (int position = 0; position < kBoardSize; ++position) {
            const float* row =
                encoded.node_features.data() + static_cast<size_t>(position) * encoded.feature_count;
            rows.append(py::cast(std::vector<float>(row, row + encoded.feature_count)));
        }
        return py::make_tuple(
            rows, py::cast(std::vector<int>(encoded.canonical_player_ids.begin(),
                                            encoded.canonical_player_ids.end())));
    }

    std::vector<double> vacancy_prior(const State& state) const {
        std::vector<int32_t> actions;
        soo::canonical_legal_action_ids(state, match_, actions);
        std::vector<double> out;
        soo::vacancy_prior(actions, canonical_self_occupancy(state, match_), out);
        return out;
    }

    int32_t to_canonical_action(int32_t action, const State& state) const {
        return soo::to_canonical_action(action, match_, state.current_player);
    }

    int32_t to_physical_action(int32_t action, const State& state) const {
        return soo::to_physical_action(action, match_, state.current_player);
    }

  private:
    Match match_;
};

}  // namespace

PYBIND11_MODULE(_diamond_native, m) {
    m.doc() = "Native Soo self-play primitives (Phase 1: rules, encoding, prior)";
    m.attr("BOARD_SIZE") = kBoardSize;
    m.attr("ACTION_SIZE") = kActionSize;
    m.attr("PHASE") = 1;

    m.def("configure", &configure, py::arg("tables"),
          "Install the authoritative board tables exported from Python.");
    m.def("is_configured", [] { return mutable_topology().configured; });
    m.def("export_tables", &export_tables,
          "Read the installed tables back, for the topology parity test.");

    py::class_<State>(m, "State")
        .def(py::init(&make_state), py::arg("occupancy"), py::arg("current_player"),
             py::arg("turn_number") = 1, py::arg("status") = 0,
             py::arg("finish_order") = std::vector<int>{})
        .def_property_readonly(
            "occupancy",
            [](const State& s) { return std::vector<int>(s.occupancy.begin(), s.occupancy.end()); })
        .def_property_readonly("current_player_id",
                               [](const State& s) { return static_cast<int>(s.current_player); })
        .def_property_readonly("turn_number",
                               [](const State& s) { return static_cast<int>(s.turn_number); })
        .def_property_readonly("status", [](const State& s) { return static_cast<int>(s.status); })
        .def_property_readonly("finish_order", &finish_order_of)
        .def("__eq__", [](const State& a, const State& b) { return a == b; })
        .def("__repr__", [](const State& s) {
            std::ostringstream out;
            out << "<native.State player=" << static_cast<int>(s.current_player)
                << " turn=" << s.turn_number << " status=" << static_cast<int>(s.status) << ">";
            return out.str();
        });

    py::class_<Game>(m, "Game")
        .def(py::init<const std::vector<std::array<int, 3>>&>(), py::arg("players"),
             "players: (seat id, camp index, target camp index) in turn order")
        .def("legal_action_ids", &Game::legal_action_ids, py::arg("state"))
        .def("canonical_legal_action_ids", &Game::canonical_legal_action_ids, py::arg("state"))
        .def("apply_action", &Game::apply_action, py::arg("state"), py::arg("action_id"))
        .def("apply_canonical_action", &Game::apply_canonical_action, py::arg("state"),
             py::arg("action_id"))
        .def("is_terminal", &Game::is_terminal, py::arg("state"))
        .def("search_current_player_id", &Game::search_current_player, py::arg("state"))
        .def("encode", &Game::encode, py::arg("state"))
        .def("vacancy_prior", &Game::vacancy_prior, py::arg("state"))
        .def("to_canonical_action", &Game::to_canonical_action, py::arg("action_id"),
             py::arg("state"))
        .def("to_physical_action", &Game::to_physical_action, py::arg("action_id"),
             py::arg("state"));
}
