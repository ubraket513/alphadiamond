#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "diamond_model/soo_model.hpp"
#include "diamond_orchestration/config.hpp"
#include "diamond_pipeline/learning_diagnostic.hpp"
#include "diamond_support/build_provenance.hpp"
#include "diamond_support/json.hpp"
#include "diamond_training/checkpoint.hpp"
#include "diamond_training/device.hpp"

namespace {
using Json = diamond_support::JsonValue;
using Object = Json::Object;
using Array = Json::Array;
struct Options {
    std::filesystem::path checkpoint, config, replay, out;
    std::string device;
    uint64_t iteration = 0, seed = 0;
    std::size_t steps = 0, batch = 0, eval_samples = 0, eval_batch = 0, log_every = 0;
};
uint64_t count(std::string_view value, std::string_view name) {
    std::size_t used = 0;
    const auto parsed = std::stoull(std::string(value), &used);
    if (used != value.size())
        throw std::invalid_argument(std::string(name) + " must be an integer");
    return parsed;
}
Options parse(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << "usage: min_learning_diagnostic --checkpoint DIR --config FILE --replay DIR "
                     "--device cpu|cuda|cuda:N --iteration N --steps N --batch-size N "
                     "--eval-samples N --eval-batch N --log-every N --seed N --out FILE\n";
        std::exit(0);
    }
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (++i == argc)
            throw std::invalid_argument(key + " requires a value");
        const std::string value = argv[i];
        if (key == "--checkpoint")
            o.checkpoint = value;
        else if (key == "--config")
            o.config = value;
        else if (key == "--replay")
            o.replay = value;
        else if (key == "--device")
            o.device = value;
        else if (key == "--iteration")
            o.iteration = count(value, key);
        else if (key == "--steps")
            o.steps = count(value, key);
        else if (key == "--batch-size")
            o.batch = count(value, key);
        else if (key == "--eval-samples")
            o.eval_samples = count(value, key);
        else if (key == "--eval-batch")
            o.eval_batch = count(value, key);
        else if (key == "--log-every")
            o.log_every = count(value, key);
        else if (key == "--seed")
            o.seed = count(value, key);
        else if (key == "--out")
            o.out = value;
        else
            throw std::invalid_argument("unknown argument: " + key);
    }
    if (o.checkpoint.empty() || o.config.empty() || o.replay.empty() || o.device.empty() ||
        o.out.empty())
        throw std::invalid_argument("checkpoint, config, replay, device, and out are required");
    return o;
}
diamond_orchestration::ProductionConfig read_config(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("cannot open config");
    const std::string text{std::istreambuf_iterator<char>(in), {}};
    return diamond_orchestration::ProductionConfig::from_json(diamond_support::parse_json(text));
}
std::filesystem::path replay_root(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path / "manifest.json")) return path;
    const auto family = path.parent_path();
    const auto schema = family.parent_path();
    if (schema.filename() != "persistent-replay-v2")
        throw std::invalid_argument("replay namespace is not persistent-replay-v2");
    return schema.parent_path();
}
Object metrics(const diamond_pipeline::HeldOutMetrics& m) {
    return {{"target_entropy", Json{m.target_entropy}},
            {"full_cross_entropy", Json{m.full_cross_entropy}},
            {"full_kl", Json{m.full_kl}},
            {"top1_agreement", Json{m.top1_agreement}},
            {"value_mse", Json{m.value_mse}}};
}
} // namespace
int main(int argc, char** argv) {
    try {
        const auto o = parse(argc, argv);
        const auto config = read_config(o.config);
        if (config.model_name != "Min")
            throw std::invalid_argument("config must be Min");
        const auto device = diamond_training::resolve_device(o.device);
        const auto compatibility = diamond_training::Compatibility::min(
            config.model_version,
            {.residual_blocks = config.network.residual_blocks, .width = config.network.width});
        auto model =
            diamond_model::DiamondModel(config.network.width, config.network.residual_blocks, 6, 3);
        diamond_training::Trainer trainer(model, compatibility,
                                          {.learning_rate = config.training.learning_rate,
                                           .weight_decay = config.training.weight_decay},
                                          device);
        const auto checkpoint = diamond_training::load_checkpoint_v3(
            o.checkpoint, trainer, device, diamond_training::CheckpointLoadIntent::exact_resume);
        const diamond_pipeline::ReplayStore replay(replay_root(o.replay), compatibility,
                                                   config.replay.capacity,
                                                   config.replay.seed,
                                                   diamond_pipeline::ReplayContents::full,
                                                   diamond_pipeline::ReplayOpenMode::must_exist);
        const auto result =
            diamond_pipeline::run_min_learning_diagnostic(trainer, replay,
                                                          {.iteration = o.iteration,
                                                           .steps = o.steps,
                                                           .batch_size = o.batch,
                                                           .evaluation_samples = o.eval_samples,
                                                           .evaluation_batch = o.eval_batch,
                                                           .log_every = o.log_every,
                                                           .seed = o.seed});
        Array steps;
        for (const auto& row : result.steps) {
            Object groups;
            for (const auto& [group, n] : row.groups)
                groups.emplace(diamond_training::parameter_group_name(group),
                               Json{Object{{"parameter_l2", Json{n.parameter_l2}},
                                           {"gradient_l2", Json{n.gradient_l2}},
                                           {"update_l2", Json{n.update_l2}},
                                           {"relative_update", Json{n.relative_update}}}});
            steps.emplace_back(
                Json{Object{{"local_step", Json{static_cast<int64_t>(row.local_step)}},
                            {"training_step", Json{static_cast<int64_t>(row.losses.training_step)}},
                            {"policy_loss", Json{row.losses.policy_loss}},
                            {"value_loss", Json{row.losses.value_loss}},
                            {"groups", Json{std::move(groups)}}}});
        }
        const Json report{Object{
            {"schema_version", Json{int64_t{1}}},
            {"checkpoint_training_step", Json{static_cast<int64_t>(checkpoint.training_step)}},
            {"model_sha256", Json{checkpoint.model_digest}},
            {"replay_size", Json{static_cast<int64_t>(replay.size())}},
            {"replay_manifest_sha256", Json{replay.manifest_digest()}},
            {"config_path", Json{o.config.string()}},
            {"device", Json{device.canonical_name}},
            {"iteration", Json{static_cast<int64_t>(o.iteration)}},
            {"steps_requested", Json{static_cast<int64_t>(o.steps)}},
            {"batch_size", Json{static_cast<int64_t>(o.batch)}},
            {"eval_samples", Json{static_cast<int64_t>(o.eval_samples)}},
            {"eval_batch", Json{static_cast<int64_t>(o.eval_batch)}},
            {"log_every", Json{static_cast<int64_t>(o.log_every)}},
            {"seed", Json{static_cast<int64_t>(o.seed)}},
            {"provenance", diamond_support::parse_json(diamond_support::build_provenance_json())},
            {"initial", Json{metrics(result.initial)}},
            {"final", Json{metrics(result.final)}},
            {"drift",
             Json{Object{{"policy_kl", Json{result.drift.policy_kl}},
                         {"logit_rms_delta", Json{result.drift.logit_rms_delta}},
                         {"value_rms_delta", Json{result.drift.value_rms_delta}},
                         {"top1_agreement_delta", Json{result.drift.top1_agreement_delta}},
                         {"full_cross_entropy_delta", Json{result.drift.full_cross_entropy_delta}},
                         {"value_mse_delta", Json{result.drift.value_mse_delta}}}}},
            {"step_diagnostics", Json{std::move(steps)}}}};
        std::ofstream out(o.out, std::ios::binary | std::ios::trunc);
        if (!out)
            throw std::runtime_error("cannot open output");
        out << diamond_support::canonical_json(report) << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "min_learning_diagnostic: " << e.what() << '\n';
        return 2;
    }
}
