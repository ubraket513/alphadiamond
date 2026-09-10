#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#include "diamond_model/soo_model.hpp"
#include "diamond_orchestration/config.hpp"
#include "diamond_training/checkpoint.hpp"
#include "soo/board.hpp"
#include "soo/encoder.hpp"
#include "soo/prior.hpp"
#include "golden.hpp"

#ifdef CHECK
#undef CHECK
#endif
#include "check.hpp"

namespace {
std::string quote(const std::filesystem::path& path) {
    return "\"" + path.string() + "\"";
}
std::string read(const std::filesystem::path& path) {
    std::ifstream file(path);
    return {std::istreambuf_iterator<char>(file), {}};
}
} // namespace

int main(int argc, char** argv) {
    REQUIRE(argc == 4, "usage: production_topology_test <scratch> <train-cli> <golden>");
    torch::set_num_threads(1);
    const auto root = std::filesystem::path(argv[1]);
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    diamond_orchestration::ProductionConfig config;
    config.model_name = "Min";
    config.network.width = 8;
    config.network.residual_blocks = 1;
    config.runtime.device = "cpu";
    config.workers.games_per_iteration = 2;
    config.workers.logical_lanes = 1;
    config.workers.search_threads = 1;
    config.inference.max_batch_size = 1;
    config.inference.max_wait_us = 1;
    config.mcts.simulations = 1;
    config.self_play.bootstrap_prior = "canonical-target-vacancy-distance-v2";
    config.self_play.bootstrap_prior_weight = 1.0;
    config.self_play.max_moves = 500;
    config.training.batch_size = 1;
    config.training.train_steps_per_iteration = 1;
    config.training.policy_loss_domain = "legal";
    config.run_budget.max_iterations = 1;
    config.arena.enabled = false;
    config.arena.games = 36;
    const auto config_path = root / "config.json";
    std::ofstream(config_path) << diamond_support::canonical_json(config.to_json());
    const auto run = root / "min" / "topology-regression";
    const std::string train = quote(argv[2]) + " train --run-dir " + quote(run) + " --config " +
                              quote(config_path) + " --scratch > " + quote(root / "train.log") +
                              " 2>&1";
    CHECK_EQ(std::system(train.c_str()), 0);
    const auto device = diamond_training::resolve_device("cpu");
    auto model = diamond_model::DiamondModel(8, 1, 6, 3);
    diamond_training::load_checkpoint_v2_weights(run / "initial-champion-checkpoint", model,
                                                 device);
    soo::ensure_topology_configured();
    auto expected = torch::zeros({6, 73, 73});
    for (int node = 0; node < 73; ++node)
        for (int direction = 0; direction < 6; ++direction) {
            const int next = soo::topology().neighbour[node][direction];
            if (next >= 0)
                expected[direction][node][next] = 1.0F;
        }
    CHECK(torch::equal(model->adjacency, expected));

    // Real encoder output, not a synthetic fixture with per-node finish flags.
    soo_test::Golden golden;
    std::string error;
    REQUIRE(soo_test::load_golden(std::string(argv[3]) + "/rules-v1.txt", golden, error),
            error.c_str());
    const auto& match = golden.match(3);
    const soo::State* opening = nullptr;
    for (const auto& entry : golden.cases)
        if (entry.tag == "opening" && entry.player_count == 3)
            opening = &entry.state;
    REQUIRE(opening != nullptr, "missing Min opening");
    auto encoded = soo::encode(*opening, match);
    std::vector<int32_t> actions;
    soo::canonical_legal_action_ids(*opening, match, actions);
    auto features = torch::from_blob(encoded.node_features.data(), {1, 73, 6}).clone();
    auto ids = torch::tensor(actions, torch::kInt32).to(torch::kLong);
    auto logits = std::get<0>(model->forward(features)).index_select(1, ids);
    CHECK((logits.max() - logits.min()).item<double>() > 1e-6);
    std::vector<double> teacher;
    soo::vacancy_prior(actions, soo::canonical_self_occupancy(*opening, match), teacher);
    auto target = torch::tensor(teacher).to(torch::kFloat32);
    (-(target * logits.log_softmax(1)).sum()).backward();
    double gradient = 0;
    for (const auto& p : model->named_parameters())
        if (p.key().find("direction_projection") != std::string::npos)
            gradient += p.value().grad().square().sum().item<double>();
    CHECK(gradient > 1e-12);

    // Save a checksum-valid but semantically invalid checkpoint, then exercise resume.
    const auto candidate = run / "iterations" / "0" / "candidate-checkpoint";
    diamond_training::Trainer trainer(diamond_model::DiamondModel(8, 1, 6, 3),
                                      diamond_training::Compatibility::min(
                                          config.model_version, {.residual_blocks = 1, .width = 8}),
                                      {config.training.learning_rate, config.training.weight_decay,
                                       diamond_training::PolicyLossDomain::legal},
                                      device);
    const auto saved = diamond_training::load_checkpoint_v3(
        candidate, trainer, device, diamond_training::CheckpointLoadIntent::exact_resume);
    trainer.model()->set_adjacency(torch::zeros({6, 73, 73}));
    diamond_training::save_checkpoint_v3(candidate, trainer, *saved.lineage, *saved.provenance);
    const std::string resume = quote(argv[2]) + " resume --run-dir " + quote(run) +
                               " --max-additional-iterations 1 > " + quote(root / "resume.log") +
                               " 2>&1";
    CHECK(std::system(resume.c_str()) != 0);
    CHECK(read(root / "resume.log").find("board adjacency does not match authoritative topology") !=
          std::string::npos);
    return soo_test::report("production_topology_test");
}
