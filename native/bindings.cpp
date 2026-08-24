// pybind11 module ``diamond_native``.
//
// Phase 1 exposes only what Gate A needs: topology injection, state, rules,
// the canonical encoder and the vacancy bootstrap prior.  MCTS, the batcher
// and the lane runner arrive in Phase 2.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "soo/action.hpp"
#include "soo/board.hpp"
#include "soo/encoder.hpp"
#include "soo/evaluator.hpp"
#include "soo/mcts.hpp"
#include "soo/prior.hpp"
#include "soo/profile.hpp"
#include "soo/random.hpp"
#include "soo/rules.hpp"
#include "soo/selfplay.hpp"
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

// Section 4's batch callback, Python side.
//
// GIL discipline (section 5): the outer entry point releases the GIL for the
// whole run, and this is the ONLY place that takes it back -- around the
// callback and nothing else. Lane threads never touch Python at all.
//
// ValueOnly is the B0 production path. The vacancy prior is native (Gate A
// proved it equals the Python oracle), so priors never cross the boundary and
// the entire policy tail disappears: no gather, no mask, no softmax, no
// [B, 5329] device-to-host copy, no priors dict, no response envelope.
class PythonBatchEvaluator : public BatchEvaluator {
  public:
    PythonBatchEvaluator(py::object callback, bool policy_value, const Match& match)
        : callback_(std::move(callback)), policy_value_(policy_value), match_(match) {}

    // ValueOnly's priors are a pure function of the request, so they are
    // computed on the search worker. Only values cross the boundary, and the
    // evaluator thread does nothing but marshal and call.
    void prepare(BatchItem& item) override {
        if (policy_value_) return;
        vacancy_prior(*item.actions, canonical_self_occupancy(*item.state, match_),
                      item.outcome->priors);
    }

    void evaluate(std::vector<BatchItem>& batch) override {
        const size_t rows = batch.size();
        if (rows == 0) return;
        const size_t features = static_cast<size_t>(batch[0].encoded->feature_count);

        // Staging buffer, reused across batches and owned by this object.
        staging_.resize(rows * kBoardSize * features);
        for (size_t i = 0; i < rows; ++i) {
            const std::vector<float>& source = batch[i].encoded->node_features;
            std::copy(source.begin(), source.end(),
                      staging_.begin() + static_cast<long>(i * kBoardSize * features));
        }

        // Ragged legal sets travel flat plus offsets rather than padded, so the
        // Python side can gather without materialising [B, max_legal].
        if (policy_value_) {
            offsets_.assign(1, 0);
            flat_actions_.clear();
            for (const BatchItem& item : batch) {
                flat_actions_.insert(flat_actions_.end(), item.actions->begin(),
                                     item.actions->end());
                offsets_.push_back(static_cast<int32_t>(flat_actions_.size()));
            }
        }

        {
            py::gil_scoped_acquire acquire;

            // A view onto the staging buffer, not a copy. The contract is that
            // the callback must not retain it past the call; it is reused on
            // the next batch.
            py::capsule borrowed(staging_.data(), [](void*) {});
            py::array_t<float> feature_view(
                {static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(kBoardSize),
                 static_cast<py::ssize_t>(features)},
                {static_cast<py::ssize_t>(kBoardSize * features * sizeof(float)),
                 static_cast<py::ssize_t>(features * sizeof(float)),
                 static_cast<py::ssize_t>(sizeof(float))},
                staging_.data(), borrowed);

            py::object answer;
            if (policy_value_) {
                py::capsule actions_borrowed(flat_actions_.data(), [](void*) {});
                py::capsule offsets_borrowed(offsets_.data(), [](void*) {});
                py::array_t<int32_t> action_view(
                    {static_cast<py::ssize_t>(flat_actions_.size())},
                    {static_cast<py::ssize_t>(sizeof(int32_t))}, flat_actions_.data(),
                    actions_borrowed);
                py::array_t<int32_t> offset_view(
                    {static_cast<py::ssize_t>(offsets_.size())},
                    {static_cast<py::ssize_t>(sizeof(int32_t))}, offsets_.data(),
                    offsets_borrowed);
                answer = callback_(feature_view, action_view, offset_view);
            } else {
                answer = callback_(feature_view);
            }
            unpack(answer, batch);
        }

    }

  private:
    void unpack(const py::object& answer, std::vector<BatchItem>& batch) {
        if (policy_value_) {
            auto pair = py::cast<py::tuple>(answer);
            if (pair.size() != 2) {
                throw std::runtime_error("PolicyValue callback must return (priors, values)");
            }
            const auto priors = py::cast<py::array_t<float>>(pair[0]);
            const auto values = py::cast<py::array_t<float>>(pair[1]);
            if (static_cast<size_t>(values.size()) != batch.size()) {
                throw std::runtime_error("callback returned the wrong number of values");
            }
            if (static_cast<size_t>(priors.size()) != flat_actions_.size()) {
                throw std::runtime_error("callback returned the wrong number of priors");
            }
            const float* prior_data = priors.data();
            const float* value_data = values.data();
            for (size_t i = 0; i < batch.size(); ++i) {
                const size_t begin = static_cast<size_t>(offsets_[i]);
                const size_t end = static_cast<size_t>(offsets_[i + 1]);
                batch[i].outcome->priors.assign(prior_data + begin, prior_data + end);
                batch[i].outcome->value = static_cast<double>(value_data[i]);
            }
            return;
        }
        const auto values = py::cast<py::array_t<float>>(answer);
        if (static_cast<size_t>(values.size()) != batch.size()) {
            throw std::runtime_error("callback returned the wrong number of values");
        }
        const float* value_data = values.data();
        for (size_t i = 0; i < batch.size(); ++i) {
            batch[i].outcome->value = static_cast<double>(value_data[i]);
        }
    }

    py::object callback_;
    bool policy_value_;
    const Match& match_;
    std::vector<float> staging_;
    std::vector<int32_t> flat_actions_;
    std::vector<int32_t> offsets_;
};

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

    static py::dict pack_search(const SearchResult& result) {
        py::dict out;
        out["selected_action"] = result.selected_action;
        out["root_actions"] = py::cast(result.root_actions);
        out["visit_counts"] = py::cast(result.visit_counts);
        out["q_values"] = py::cast(result.q_values);
        out["policy"] = py::cast(result.policy);
        out["root_priors"] = py::cast(result.root_priors);
        out["root_network_value"] = result.root_network_value;
        out["root_mean_value"] = result.root_mean_value;
        out["simulations_run"] = result.simulations_run;
        out["evaluator_calls"] = result.evaluator_calls;
        out["nodes_created"] = result.nodes_created;
        py::list trace_out;
        for (const EvalRecord& record : result.trace) {
            trace_out.append(py::make_tuple(record.request_hash, py::cast(record.legal_actions)));
        }
        out["trace"] = trace_out;
        return out;
    }

    // Gate B: one deterministic search, single-threaded, no Python callback.
    py::dict search(const State& state, const MCTSConfig& config, double temperature,
                    bool trace, const std::string& evaluator_name) const {
        DeterministicEvaluator hashed;
        UniformPriorEvaluator uniform;
        Evaluator& evaluator = pick(evaluator_name, hashed, uniform);
        MCTS2P search(match_, evaluator, config);
        return pack_search(search.run(state, temperature, trace));
    }

    // One search, answered by a Python callback: the arena's shape, where two
    // different networks alternate moves in the same game.
    //
    // Unlike schedule_with_callback this keeps the GIL. There is no evaluator
    // thread to deadlock against -- the session suspends on the calling thread,
    // the callback runs there too, and a batch is one position. That is the
    // right trade here: an arena move is latency-bound on a single forward
    // pass, and the alternative (a scheduler per move) would pay for threads it
    // cannot fill.
    py::dict search_with_callback(const State& state, const MCTSConfig& config, double temperature,
                                  bool trace, const py::object& callback,
                                  const std::string& mode) const {
        if (mode != "value_only" && mode != "policy_value") {
            throw std::invalid_argument("mode must be 'value_only' or 'policy_value'");
        }
        PythonBatchEvaluator evaluator(callback, mode == "policy_value", match_);
        SearchSession session(match_, config);
        session.begin(state, temperature, trace);
        while (session.advance() == SearchSession::Status::NeedsEvaluation) {
            EvalOutcome outcome;
            BatchItem item;
            item.state = &session.pending_state();
            item.encoded = &session.pending_features();
            item.actions = &session.pending_actions();
            item.outcome = &outcome;
            evaluator.prepare(item);
            std::vector<BatchItem> batch{item};
            evaluator.evaluate(batch);
            session.supply(outcome);
        }
        return pack_search(session.result());
    }

    // Gate D: the same scheduler, answered by a Python callback.
    //
    // The GIL is released for the entire run. The only thread that ever takes
    // it back is the single evaluator thread, around the callback alone. If
    // this entry point kept the GIL, the evaluator thread would block on it
    // forever and the run would hang immediately -- the one deadlock the
    // design must not have.
    py::dict schedule_with_callback(const State& opening, const SchedulerConfig& config,
                                    const py::object& callback,
                                    const std::string& mode) const {
        if (mode != "value_only" && mode != "policy_value") {
            throw std::invalid_argument("mode must be 'value_only' or 'policy_value'");
        }
        SchedulerMetrics metrics;
        PythonBatchEvaluator evaluator(callback, mode == "policy_value", match_);
        {
            py::gil_scoped_release release;
            metrics = run_scheduler(match_, opening, config, evaluator);
        }
        return pack(metrics);
    }

    // Native self-play: a fixed set of games played to completion, each move
    // recorded, over the Gate C scheduler and the Gate D callback.
    //
    // Features come back as one [moves, 73, F] float32 array per episode rather
    // than nested lists.  A 90-move game at 73x4 is 26k floats; as Python lists
    // that is 26k boxed objects per game, per iteration, on the parent process
    // -- which is precisely the serialized resource the whole port exists to
    // stop loading up.
    py::dict play_episodes(const std::vector<std::pair<State, uint64_t>>& jobs,
                           const EpisodeConfig& config, const py::object& callback,
                           const std::string& mode) const {
        if (mode != "value_only" && mode != "policy_value") {
            throw std::invalid_argument("mode must be 'value_only' or 'policy_value'");
        }
        std::vector<EpisodeJob> native_jobs;
        native_jobs.reserve(jobs.size());
        for (const auto& job : jobs) native_jobs.push_back(EpisodeJob{job.first, job.second});

        EpisodeMetrics metrics;
        std::vector<Episode> episodes;
        PythonBatchEvaluator evaluator(callback, mode == "policy_value", match_);
        {
            py::gil_scoped_release release;
            episodes = run_episodes(match_, native_jobs, config, evaluator, metrics);
        }

        py::list out;
        for (const Episode& episode : episodes) {
            py::list moves;
            for (const EpisodeMove& move : episode.moves) {
                const size_t features = static_cast<size_t>(move.features.feature_count);
                py::array_t<float> node_features(
                    std::vector<py::ssize_t>{static_cast<py::ssize_t>(kBoardSize),
                                             static_cast<py::ssize_t>(features)});
                std::memcpy(node_features.mutable_data(), move.features.node_features.data(),
                            move.features.node_features.size() * sizeof(float));
                py::dict entry;
                entry["node_features"] = node_features;
                entry["canonical_player_ids"] =
                    py::cast(std::vector<int>(move.features.canonical_player_ids.begin(),
                                              move.features.canonical_player_ids.end()));
                entry["root_actions"] = py::cast(move.root_actions);
                entry["visit_counts"] = py::cast(move.visit_counts);
                entry["selected_action"] = move.selected_action;
                moves.append(entry);
            }
            py::dict record;
            record["moves"] = moves;
            record["finish_order"] =
                py::cast(std::vector<int>(episode.finish_order.begin(), episode.finish_order.end()));
            record["move_count"] = episode.move_count;
            record["completed"] = episode.completed;
            record["move_limit_exceeded"] = episode.move_limit_exceeded;
            out.append(record);
        }

        py::dict result;
        result["episodes"] = out;
        result["evaluations"] = metrics.evaluations;
        result["batches"] = metrics.batches;
        result["moves"] = metrics.moves;
        result["boosted_moves"] = metrics.boosted_moves;
        result["wall_seconds"] = metrics.wall_seconds;
        result["evaluator_seconds"] = metrics.evaluator_seconds;
        result["worker_busy_seconds"] = metrics.worker_busy_seconds;
        result["batch_sizes"] = py::cast(metrics.batch_sizes);
        return result;
    }

    // Gate C.1: per-stage and whole-search native cost, no threads, no Python.
    py::dict profile(const std::vector<State>& states, const MCTSConfig& config, int repeats,
                     bool searches) const {
        StageTiming timing;
        {
            py::gil_scoped_release release;
            timing = searches ? profile_searches(states, match_, config, repeats)
                              : profile_stages(states, match_, repeats);
        }
        py::dict out;
        out["legal_ns"] = timing.legal_ns;
        out["prior_ns"] = timing.prior_ns;
        out["encode_ns"] = timing.encode_ns;
        out["apply_ns"] = timing.apply_ns;
        out["search_ns"] = timing.search_ns;
        out["evaluations"] = timing.evaluations;
        out["searches"] = timing.searches;
        out["nodes"] = timing.nodes;
        return out;
    }

    // Gate C.2/C.3: many logical games over a fixed worker pool, one global
    // batcher, dummy evaluator. The GIL is released for the whole run -- no
    // Python is touched between here and the return.
    py::dict schedule(const State& opening, const SchedulerConfig& config) const {
        SchedulerMetrics metrics;
        {
            DummyBatchEvaluator evaluator(config.eval_latency_ms);
            py::gil_scoped_release release;
            metrics = run_scheduler(match_, opening, config, evaluator);
        }
        return pack(metrics);
    }

    static py::dict pack(const SchedulerMetrics& metrics) {
        py::dict out;
        out["evaluations"] = metrics.evaluations;
        out["batches"] = metrics.batches;
        out["moves"] = metrics.moves;
        out["games_finished"] = metrics.games_finished;
        out["batcher_wakeups"] = metrics.batcher_wakeups;
        out["wall_seconds"] = metrics.wall_seconds;
        out["worker_busy_seconds"] = metrics.worker_busy_seconds;
        out["evaluator_seconds"] = metrics.evaluator_seconds;
        out["batch_sizes"] = py::cast(metrics.batch_sizes);
        out["ready_depth"] = py::cast(metrics.ready_depth);
        out["waiting"] = py::cast(metrics.waiting);
        out["wait_ns"] = py::cast(metrics.wait_ns);
        py::list lane_moves;
        for (const auto& moves : metrics.lane_moves) lane_moves.append(py::cast(moves));
        out["lane_moves"] = lane_moves;
        return out;
    }

    // The reference evaluator, exposed on its own so evaluator parity can be
    // tested apart from search parity.
    py::tuple evaluate(const State& state, const std::string& evaluator_name) const {
        const Encoded encoded = soo::encode(state, match_);
        std::vector<int32_t> legal;
        soo::canonical_legal_action_ids(state, match_, legal);
        DeterministicEvaluator hashed;
        UniformPriorEvaluator uniform;
        const EvalOutcome outcome = pick(evaluator_name, hashed, uniform).evaluate(encoded, legal);
        return py::make_tuple(py::cast(legal), py::cast(outcome.priors), outcome.value,
                              request_hash(encoded, legal));
    }

  private:
    static Evaluator& pick(const std::string& name, DeterministicEvaluator& hashed,
                           UniformPriorEvaluator& uniform) {
        if (name == "hash") return hashed;
        if (name == "uniform") return uniform;
        throw std::invalid_argument("unknown reference evaluator: " + name);
    }

    Match match_;
};

}  // namespace

PYBIND11_MODULE(_diamond_native, m) {
    m.doc() = "Native Soo self-play primitives (Phase 1: rules, encoding, prior)";
    m.attr("BOARD_SIZE") = kBoardSize;
    m.attr("ACTION_SIZE") = kActionSize;
    m.attr("PHASE") = 2;

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

    // The samplers, exposed so their *distributions* can be gated against
    // Python's directly.  Section 9 does not require a matching draw sequence,
    // which means a wrong sampler cannot be caught by comparing streams -- only
    // by comparing distributions, and that needs the sampler callable in
    // isolation rather than buried three layers inside a search.
    m.def(
        "sample_gamma",
        [](double alpha, size_t count, uint64_t seed) {
            soo::Rng rng(seed);
            std::vector<double> out;
            out.reserve(count);
            for (size_t i = 0; i < count; ++i) out.push_back(rng.gammavariate(alpha));
            return out;
        },
        py::arg("alpha"), py::arg("count"), py::arg("seed") = 0,
        "Gamma(alpha, 1) draws -- the distribution behind add_dirichlet_noise");
    m.def(
        "sample_weighted",
        [](const std::vector<double>& weights, size_t count, uint64_t seed) {
            soo::Rng rng(seed);
            std::vector<size_t> out;
            out.reserve(count);
            for (size_t i = 0; i < count; ++i) out.push_back(rng.weighted_index(weights));
            return out;
        },
        py::arg("weights"), py::arg("count"), py::arg("seed") = 0,
        "Indices drawn like random.choices(population, weights, k=1)");

    py::class_<MCTSConfig>(m, "MCTSConfig")
        .def(py::init([](int simulations, double c_puct, double dirichlet_alpha,
                         double dirichlet_epsilon, uint64_t seed) {
                 return MCTSConfig{simulations, c_puct, dirichlet_alpha, dirichlet_epsilon, seed};
             }),
             py::arg("simulations") = 200, py::arg("c_puct") = 1.5,
             py::arg("dirichlet_alpha") = 0.3, py::arg("dirichlet_epsilon") = 0.0,
             py::arg("seed") = 0)
        .def_readwrite("simulations", &MCTSConfig::simulations)
        .def_readwrite("c_puct", &MCTSConfig::c_puct)
        .def_readwrite("dirichlet_alpha", &MCTSConfig::dirichlet_alpha)
        .def_readwrite("dirichlet_epsilon", &MCTSConfig::dirichlet_epsilon)
        .def_readwrite("seed", &MCTSConfig::seed);

    py::class_<EpisodeConfig>(m, "EpisodeConfig")
        .def(py::init([](int lanes, int threads, int max_batch, int max_wait_us, int simulations,
                         int max_moves, double temperature, int temperature_moves,
                         double dirichlet_alpha, double dirichlet_epsilon,
                         int simulations_late, int late_move_threshold,
                         int repeat_window) {
                 return EpisodeConfig{lanes,            threads,
                                      max_batch,        max_wait_us,
                                      simulations,      max_moves,
                                      temperature,      temperature_moves,
                                      dirichlet_alpha,  dirichlet_epsilon,
                                      simulations_late, late_move_threshold,
                                      repeat_window};
             }),
             py::arg("lanes") = 0,
             py::arg("threads") = 4, py::arg("max_batch") = 32, py::arg("max_wait_us") = 2000,
             py::arg("simulations") = 64, py::arg("max_moves") = 2000,
             py::arg("temperature") = 1.0, py::arg("temperature_moves") = 20,
             py::arg("dirichlet_alpha") = 0.3, py::arg("dirichlet_epsilon") = 0.25,
             py::arg("simulations_late") = 0, py::arg("late_move_threshold") = 0,
             py::arg("repeat_window") = 0);

    py::class_<SchedulerConfig>(m, "SchedulerConfig")
        .def(py::init([](int games, int threads, int max_batch, int max_wait_us, int simulations,
                         int max_moves, double eval_latency_ms, double seconds,
                         bool trace_moves, int stop_after_moves, double temperature,
                         int temperature_moves, double dirichlet_alpha,
                         double dirichlet_epsilon, uint64_t seed) {
                 return SchedulerConfig{games,
                                        threads,
                                        max_batch,
                                        max_wait_us,
                                        simulations,
                                        max_moves,
                                        eval_latency_ms,
                                        seconds,
                                        trace_moves,
                                        stop_after_moves,
                                        temperature,
                                        temperature_moves,
                                        dirichlet_alpha,
                                        dirichlet_epsilon,
                                        seed};
             }),
             py::arg("games") = 64, py::arg("threads") = 4, py::arg("max_batch") = 32,
             py::arg("max_wait_us") = 2000, py::arg("simulations") = 64,
             py::arg("max_moves") = 400, py::arg("eval_latency_ms") = 0.0,
             py::arg("seconds") = 5.0, py::arg("trace_moves") = false,
             py::arg("stop_after_moves") = 0, py::arg("temperature") = 0.0,
             py::arg("temperature_moves") = 0, py::arg("dirichlet_alpha") = 0.3,
             py::arg("dirichlet_epsilon") = 0.0, py::arg("seed") = 0);

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
             py::arg("state"))
        .def("search", &Game::search, py::arg("state"), py::arg("config"),
             py::arg("temperature") = 0.0, py::arg("trace") = false,
             py::arg("evaluator") = "hash")
        .def("search_with_callback", &Game::search_with_callback, py::arg("state"),
             py::arg("config"), py::arg("temperature") = 0.0, py::arg("trace") = false,
             py::arg("callback") = py::none(), py::arg("mode") = "value_only")
        .def("schedule", &Game::schedule, py::arg("opening"), py::arg("config"))
        .def("play_episodes", &Game::play_episodes, py::arg("jobs"), py::arg("config"),
             py::arg("callback"), py::arg("mode") = "value_only")
        .def("schedule_with_callback", &Game::schedule_with_callback, py::arg("opening"),
             py::arg("config"), py::arg("callback"), py::arg("mode") = "value_only")
        .def("profile", &Game::profile, py::arg("states"), py::arg("config"),
             py::arg("repeats") = 1, py::arg("searches") = false)
        .def("reference_evaluate", &Game::evaluate, py::arg("state"),
             py::arg("evaluator") = "hash",
             "(legal actions, priors, value, request hash) from the Gate B evaluator");
}
