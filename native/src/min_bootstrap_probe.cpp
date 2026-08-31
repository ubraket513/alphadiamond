// Which prior should carry Min out of a cold start?
//
// A from-scratch Min network cannot finish a game, and a game that does not
// finish contributes nothing, so the first iterations are driven by whatever
// prior is put in front of the search. This probe measures the candidates
// against each other under one fixed condition: authoritative three-player
// rules, MCTS3P, the same openings and the same per-game seeds, and a leaf
// value of zero on every arm.
//
// Zero values are the point, not an omission. The question is whether a prior
// produces better trajectories, and a value signal -- Soo's scalar mapped onto
// three seats, or anything else -- would answer a different question with a
// heuristic of its own. With values pinned at zero, every arm differs in
// exactly one term.
//
//   soo-policy   the shipped Soo v2.0.0 policy, read through a 3P -> 2P input
//                adapter (see fold_to_soo_features)
//   vacancy      canonical-target-vacancy-distance, the current bootstrap
//   uniform      the negative control: no information at all
//
// What comes out is not a win rate. Three copies of one agent cannot rank
// themselves; what they can show is whether games end, whether they end from
// every seat, and whether the search escapes the short-cycle attractor that the
// aborted tail is made of.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "diamond_model/deployment_artifact.hpp"
#include "diamond_model/soo_model.hpp"
#include "diamond_support/json.hpp"
#include "soo/board.hpp"
#include "soo/encoder.hpp"
#include "soo/prior.hpp"
#include "soo/rules.hpp"
#include "soo/selfplay.hpp"

namespace {

using Json = diamond_support::JsonValue;
using JsonArray = Json::Array;
using JsonObject = Json::Object;

constexpr int kBoardNodes = 73;
constexpr int kActionSpace = kBoardNodes * kBoardNodes;
constexpr int kMinFeatures = 6;
constexpr int kSooFeatures = 4;

// Min's six channels are [occ(self), occ(next), occ(last), fin(self),
// fin(next), fin(last)] in canonical seat order; Soo's four are [occ(self),
// occ(other), fin(self), fin(other)]. Folding the two opponents together with
// OR is lossless about occupancy -- a hole holds at most one piece, so the
// union is exactly the set of blocked holes -- and lossy about identity: Soo
// cannot see *which* opponent is standing in a camp. That loss is the thing
// this probe exists to measure, not a defect to hide.
void fold_to_soo_features(const soo::Encoded& min_state, std::vector<float>& out) {
    // A two-player encoding is already what Soo wants; folding is the 3P case.
    if (min_state.feature_count == kSooFeatures) {
        out.assign(min_state.node_features.begin(), min_state.node_features.end());
        return;
    }
    if (min_state.feature_count != kMinFeatures ||
        min_state.node_features.size() != static_cast<std::size_t>(kBoardNodes) * kMinFeatures) {
        throw std::runtime_error("the adapter requires a three-player [73,6] encoding");
    }
    out.resize(static_cast<std::size_t>(kBoardNodes) * kSooFeatures);
    for (int node = 0; node < kBoardNodes; ++node) {
        const float* in = min_state.node_features.data() + static_cast<std::size_t>(node) * kMinFeatures;
        float* row = out.data() + static_cast<std::size_t>(node) * kSooFeatures;
        row[0] = in[0];
        row[1] = std::max(in[1], in[2]);
        row[2] = in[3];
        row[3] = std::max(in[4], in[5]);
    }
}

// Every arm returns a zero value vector; only the prior differs.
void fill_zero_values(soo::BatchItem& item) {
    item.outcome->value = 0.0;
    item.outcome->values = {0.0, 0.0, 0.0};
}

// The control this probe needs to be believable: Soo, on Soo's own game, under
// the same zero-value search. If that does not finish games either, the 3P
// result says nothing about the adapter -- it says the harness is wrong.

void require_finite_distribution(const std::vector<double>& priors) {
    double total = 0.0;
    for (const double prior : priors) {
        if (!std::isfinite(prior) || prior < 0.0)
            throw std::runtime_error("a prior must be finite and non-negative");
        total += prior;
    }
    if (!(total > 0.0)) throw std::runtime_error("a prior must not sum to zero");
}

// The vacancy heuristic, computed on the search worker: it is ~7.5 us per
// evaluation, and the evaluator thread is the serial resource.
class VacancyArm final : public soo::BatchEvaluator {
  public:
    void prepare(soo::BatchItem& item) override {
        soo::vacancy_prior(*item.actions,
                           soo::canonical_self_occupancy(*item.state, match_),
                           item.outcome->priors);
        require_finite_distribution(item.outcome->priors);
    }
    void evaluate(std::vector<soo::BatchItem>& batch) override {
        for (auto& item : batch) fill_zero_values(item);
    }
    explicit VacancyArm(const soo::Match& match) : match_(match) {}

  private:
    soo::Match match_;
};

class UniformArm final : public soo::BatchEvaluator {
  public:
    void prepare(soo::BatchItem& item) override {
        const double weight = 1.0 / static_cast<double>(item.actions->size());
        item.outcome->priors.assign(item.actions->size(), weight);
    }
    void evaluate(std::vector<soo::BatchItem>& batch) override {
        for (auto& item : batch) fill_zero_values(item);
    }
};

// Soo's policy, asked about three-player positions.
//
// The legal set is never Soo's to decide: the actions handed to the model come
// from the authoritative 3P rules, and the softmax runs over exactly those. So
// the adapter can only reweight moves that are genuinely available -- it cannot
// invent a two-player move.
class SooPolicyArm final : public soo::BatchEvaluator {
  public:
    // The model must already be on `device`, with its weights loaded there:
    // the trunk's neighbour gather tables are derived from `adjacency` and
    // cached against its version, so a plain to(device) after loading leaves
    // them behind on the host. Loading into a model that is already on the
    // device writes adjacency through copy_, which is what rebuilds them.
    SooPolicyArm(diamond_model::SooModel model, torch::Device device)
        : model_(std::move(model)), device_(device) {
        model_->eval();
    }

    void evaluate(std::vector<soo::BatchItem>& batch) override {
        if (batch.empty()) return;
        torch::NoGradGuard no_grad;
        const auto rows = static_cast<int64_t>(batch.size());

        std::size_t widest = 0;
        for (const auto& item : batch) widest = std::max(widest, item.actions->size());
        const auto columns = static_cast<int64_t>(std::max<std::size_t>(widest, 1));

        auto features = torch::empty({rows, kBoardNodes, kSooFeatures}, torch::kFloat32);
        auto indices = torch::zeros({rows, columns}, torch::kLong);
        auto valid = torch::zeros({rows, columns}, torch::kBool);
        float* feature_out = features.data_ptr<float>();
        int64_t* index_out = indices.data_ptr<int64_t>();
        bool* valid_out = valid.data_ptr<bool>();

        for (std::size_t row = 0; row < batch.size(); ++row) {
            const auto& item = batch[row];
            if (!item.encoded || !item.actions || !item.outcome)
                throw std::runtime_error("malformed probe batch item");
            if (item.actions->empty())
                throw std::runtime_error("a search node must have at least one legal action");
            fold_to_soo_features(*item.encoded, folded_);
            std::copy(folded_.begin(), folded_.end(),
                      feature_out + row * kBoardNodes * kSooFeatures);
            for (std::size_t column = 0; column < item.actions->size(); ++column) {
                const int32_t action = (*item.actions)[column];
                if (action < 0 || action >= kActionSpace)
                    throw std::runtime_error("a legal action is outside the policy space");
                index_out[row * static_cast<std::size_t>(columns) + column] = action;
                valid_out[row * static_cast<std::size_t>(columns) + column] = true;
            }
        }

        const auto [policy, value] = model_->forward(features.to(device_));
        (void)value;   // Soo's scalar is deliberately discarded; see the header.
        auto logits = policy.gather(1, indices.to(device_));
        // Padding columns must not take probability mass from the real ones.
        logits = logits.masked_fill(~valid.to(device_), -std::numeric_limits<float>::infinity());
        const auto probabilities = torch::softmax(logits, 1).to(torch::kCPU).contiguous();
        const float* read = probabilities.data_ptr<float>();

        for (std::size_t row = 0; row < batch.size(); ++row) {
            auto& item = batch[row];
            const float* source = read + row * static_cast<std::size_t>(columns);
            item.outcome->priors.assign(source, source + item.actions->size());
            require_finite_distribution(item.outcome->priors);
            fill_zero_values(item);
        }
    }

  private:
    diamond_model::SooModel model_;
    torch::Device device_;
    std::vector<float> folded_;
};

soo::State opening(const soo::Match& match) {
    soo::State state;
    state.occupancy.fill(soo::kEmpty);
    for (std::size_t seat = 0; seat < match.count; ++seat)
        for (uint8_t position : soo::topology().camp_positions[match.players[seat].camp])
            state.occupancy[position] = match.players[seat].id;
    state.current_player = match.players[0].id;
    return state;
}

// The shortest period the tail is consistent with, or 0 if it is not periodic.
//
// `max_revisits` counts how often one position recurred; it cannot say whether
// the game is locked in an A-B-A oscillation or returning slowly to a hub. The
// period is what distinguishes them, and it is a property of the tail by
// nature: a game that cycles is cycling when it is cut off.
int dominant_cycle_period(const std::vector<uint64_t>& tail, int longest = 32) {
    const auto size = static_cast<int>(tail.size());
    if (size < 4) return 0;
    for (int period = 1; period <= longest && period * 3 <= size; ++period) {
        // Require three full repeats, so a coincidental match is not a cycle.
        bool periodic = true;
        for (int back = 0; back < period * 3 && periodic; ++back) {
            const int here = size - 1 - back;
            const int previous = here - period;
            if (previous < 0 || tail[static_cast<std::size_t>(here)] !=
                                    tail[static_cast<std::size_t>(previous)])
                periodic = false;
        }
        if (periodic) return period;
    }
    return 0;
}

double percentile(std::vector<int> values, double fraction) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const auto index = static_cast<std::size_t>(fraction * static_cast<double>(values.size() - 1));
    return values[index];
}

struct Options {
    std::string arm = "vacancy";
    std::filesystem::path artifact = "models/soo/2.0.0";
    std::filesystem::path out;
    int games = 36;
    int simulations = 64;
    int max_moves = 2000;
    int lanes = 32;
    int threads = 8;
    int max_batch = 64;
    int max_wait_us = 50;
    double temperature = 1.0;
    int temperature_moves = 20;
    double dirichlet_epsilon = 0.25;
    uint64_t seed = 7;
    std::string device = "cuda";
    std::string match = "min";   // "soo" runs the two-player control
};

Options parse(int argc, char** argv) {
    Options options;
    const auto next = [&](int& index) -> std::string {
        if (index + 1 >= argc) throw std::runtime_error(std::string(argv[index]) + " needs a value");
        return argv[++index];
    };
    for (int index = 1; index < argc; ++index) {
        const std::string flag = argv[index];
        if (flag == "--arm") options.arm = next(index);
        else if (flag == "--artifact") options.artifact = next(index);
        else if (flag == "--out") options.out = next(index);
        else if (flag == "--games") options.games = std::stoi(next(index));
        else if (flag == "--simulations") options.simulations = std::stoi(next(index));
        else if (flag == "--max-moves") options.max_moves = std::stoi(next(index));
        else if (flag == "--lanes") options.lanes = std::stoi(next(index));
        else if (flag == "--threads") options.threads = std::stoi(next(index));
        else if (flag == "--batch") options.max_batch = std::stoi(next(index));
        else if (flag == "--max-wait-us") options.max_wait_us = std::stoi(next(index));
        else if (flag == "--temperature") options.temperature = std::stod(next(index));
        else if (flag == "--temperature-moves") options.temperature_moves = std::stoi(next(index));
        else if (flag == "--dirichlet-epsilon") options.dirichlet_epsilon = std::stod(next(index));
        else if (flag == "--seed") options.seed = std::stoull(next(index));
        else if (flag == "--device") options.device = next(index);
        else if (flag == "--match") options.match = next(index);
        else throw std::runtime_error("unknown option: " + flag);
    }
    if (options.arm != "soo-policy" && options.arm != "vacancy" && options.arm != "uniform")
        throw std::runtime_error("--arm must be soo-policy, vacancy or uniform");
    if (options.match != "min" && options.match != "soo")
        throw std::runtime_error("--match must be min or soo");
    if (options.games < 1) throw std::runtime_error("--games must be positive");
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse(argc, argv);
        soo::ensure_topology_configured();
        const soo::Match match =
            options.match == "soo" ? soo::standard_soo_match() : soo::standard_min_match();
        const soo::State start = opening(match);

        // The same per-game seeds on every arm: a game is the same experiment
        // with one term changed, not a different game.
        std::vector<soo::EpisodeJob> jobs;
        jobs.reserve(static_cast<std::size_t>(options.games));
        for (int game = 0; game < options.games; ++game) {
            uint64_t seed = options.seed ^ (static_cast<uint64_t>(game) * 0x9e3779b97f4a7c15ULL);
            seed ^= seed >> 30;
            seed *= 0xbf58476d1ce4e5b9ULL;
            seed ^= seed >> 27;
            jobs.push_back({start, seed});
        }

        soo::EpisodeConfig config;
        config.lanes = options.lanes;
        config.threads = options.threads;
        config.max_batch = options.max_batch;
        config.max_wait_us = options.max_wait_us;
        config.simulations = options.simulations;
        config.max_moves = options.max_moves;
        config.temperature = options.temperature;
        config.temperature_moves = options.temperature_moves;
        config.dirichlet_epsilon = options.dirichlet_epsilon;
        // The arms supply their own priors through the evaluator, so the
        // in-pipeline bootstrap switch stays off: one prior source per run.
        config.bootstrap_prior = false;

        std::unique_ptr<soo::BatchEvaluator> evaluator;
        std::string model_identity = "none";
        if (options.arm == "vacancy") {
            evaluator = std::make_unique<VacancyArm>(match);
        } else if (options.arm == "uniform") {
            evaluator = std::make_unique<UniformArm>();
        } else {
            const auto artifact = diamond_model::validate_deployment_artifact(
                options.artifact, "soo");
            if (artifact.input_features != kSooFeatures || artifact.value_size != 1)
                throw std::runtime_error("the teacher artifact is not a two-player Soo model");
            const torch::Device device(options.device == "cuda" && torch::cuda::is_available()
                                           ? torch::kCUDA
                                           : torch::kCPU);
            diamond_model::SooModel model(artifact.width, artifact.residual_blocks);
            model->to(device);
            model->load_weights(artifact.weights);
            evaluator = std::make_unique<SooPolicyArm>(std::move(model), device);
            model_identity = artifact.model_family + ":" + artifact.model_version + ":" +
                             artifact.model_sha256;
        }

        soo::EpisodeMetrics metrics;
        const auto started = std::chrono::steady_clock::now();
        const auto episodes = soo::run_episodes(match, jobs, config, *evaluator, metrics);
        const double wall =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

        int completed = 0;
        std::vector<int> completed_moves;
        std::vector<int> aborted_moves;
        std::array<int, 4> wins_by_seat{};        // indexed by player id
        double revisit_fraction_total = 0.0;
        double repeat_within_8_total = 0.0;
        double revisit_total = 0.0;
        int cycling_games = 0;
        std::vector<int> cycle_periods;
        double foreign_blockers_total = 0.0;
        double stale_camp_total = 0.0;
        JsonArray games;

        for (const auto& episode : episodes) {
            const auto& diagnostics = episode.diagnostics;
            // The opening is observed when the lane is seated and every move
            // adds one more, so the denominator is move_count + 1. Dividing by
            // move_count reports 1 + 1/move_count for a game that repeated
            // nothing, which reads like an error and hides the headroom.
            const double observations = diagnostics.observations > 0
                                            ? static_cast<double>(diagnostics.observations)
                                            : 1.0;
            const double revisit_fraction =
                1.0 - static_cast<double>(diagnostics.unique_positions) / observations;
            const double repeat_within_8 =
                static_cast<double>(diagnostics.repeat_within_8) / observations;
            revisit_fraction_total += revisit_fraction;
            repeat_within_8_total += repeat_within_8;
            revisit_total += diagnostics.max_revisits;
            const int period = dominant_cycle_period(diagnostics.recent_keys);
            if (period > 0) {
                ++cycling_games;
                cycle_periods.push_back(period);
            }
            int foreign = 0;
            uint32_t stale = 0;
            for (const auto& camp : diagnostics.camps) {
                foreign += camp.foreign_in_target;
                stale = std::max(stale, camp.plies_since_camp_changed);
            }
            foreign_blockers_total += foreign;
            stale_camp_total += stale;
            if (episode.completed) {
                ++completed;
                completed_moves.push_back(episode.move_count);
                if (!episode.finish_order.empty() &&
                    episode.finish_order.front() < wins_by_seat.size())
                    ++wins_by_seat[episode.finish_order.front()];
            } else {
                aborted_moves.push_back(episode.move_count);
            }

            JsonArray order;
            for (const uint8_t seat : episode.finish_order) order.push_back(Json{int64_t(seat)});
            games.push_back(Json{JsonObject{
                {"completed", Json{episode.completed}},
                {"moves", Json{static_cast<int64_t>(episode.move_count)}},
                {"finish_order", Json{order}},
                {"unique_positions", Json{static_cast<int64_t>(diagnostics.unique_positions)}},
                {"observations", Json{static_cast<int64_t>(diagnostics.observations)}},
                {"max_revisits", Json{static_cast<int64_t>(diagnostics.max_revisits)}},
                {"revisit_fraction", Json{revisit_fraction}},
                {"repeat_within_8_fraction", Json{repeat_within_8}},
                {"dominant_cycle_period", Json{static_cast<int64_t>(period)}},
                {"foreign_in_target", Json{static_cast<int64_t>(foreign)}},
                {"plies_since_camp_changed", Json{static_cast<int64_t>(stale)}},
            }});
        }

        const auto count = static_cast<double>(episodes.size());
        JsonObject seat_wins;
        for (std::size_t id = 1; id < wins_by_seat.size(); ++id)
            seat_wins.emplace("player_" + std::to_string(id), Json{int64_t(wins_by_seat[id])});

        const Json report{JsonObject{
            {"schema_version", Json{int64_t{1}}},
            {"arm", Json{options.arm}},
            {"match", Json{options.match}},
            {"teacher", Json{model_identity}},
            {"games", Json{static_cast<int64_t>(episodes.size())}},
            {"simulations", Json{static_cast<int64_t>(options.simulations)}},
            {"max_moves", Json{static_cast<int64_t>(options.max_moves)}},
            {"seed", Json{static_cast<int64_t>(options.seed)}},
            {"completed", Json{int64_t(completed)}},
            {"completion_rate", Json{count > 0 ? completed / count : 0.0}},
            {"completed_moves_p50", Json{percentile(completed_moves, 0.50)}},
            {"completed_moves_p90", Json{percentile(completed_moves, 0.90)}},
            {"completed_moves_p99", Json{percentile(completed_moves, 0.99)}},
            {"aborted_moves_p50", Json{percentile(aborted_moves, 0.50)}},
            {"mean_revisit_fraction", Json{count > 0 ? revisit_fraction_total / count : 0.0}},
            {"mean_repeat_within_8_fraction",
             Json{count > 0 ? repeat_within_8_total / count : 0.0}},
            {"mean_max_revisits", Json{count > 0 ? revisit_total / count : 0.0}},
            {"cycling_games", Json{int64_t(cycling_games)}},
            {"cycling_fraction", Json{count > 0 ? cycling_games / count : 0.0}},
            {"median_cycle_period", Json{percentile(cycle_periods, 0.50)}},
            {"mean_foreign_in_target", Json{count > 0 ? foreign_blockers_total / count : 0.0}},
            {"mean_plies_since_camp_changed", Json{count > 0 ? stale_camp_total / count : 0.0}},
            {"first_finisher_by_player", Json{seat_wins}},
            {"wall_seconds", Json{wall}},
            {"moves", Json{static_cast<int64_t>(metrics.moves)}},
            {"evaluations", Json{static_cast<int64_t>(metrics.evaluations)}},
            {"evaluator_seconds", Json{metrics.evaluator_seconds}},
            {"per_game", Json{games}},
        }};

        const auto text = diamond_support::canonical_json(report);
        if (!options.out.empty()) {
            std::ofstream file(options.out);
            if (!file) throw std::runtime_error("cannot write " + options.out.string());
            file << text << '\n';
        }
        std::cout << text << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "min_bootstrap_probe: " << error.what() << '\n';
        return 1;
    }
}
