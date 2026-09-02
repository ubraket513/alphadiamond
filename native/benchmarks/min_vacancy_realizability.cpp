#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "diamond_model/soo_model.hpp"
#include "diamond_orchestration/config.hpp"
#include "diamond_pipeline/replay_store.hpp"
#include "diamond_pipeline/vacancy_distillation.hpp"
#include "diamond_support/build_provenance.hpp"
#include "diamond_support/json.hpp"
#include "diamond_training/checkpoint.hpp"
#include "diamond_training/device.hpp"
#include "soo/board.hpp"

namespace {

using Json = diamond_support::JsonValue;
using Object = Json::Object;

struct Options {
    std::filesystem::path checkpoint;
    std::filesystem::path config;
    std::filesystem::path replay;
    std::filesystem::path out;
    std::filesystem::path checkpoint_out;
    std::string arm;
    std::string device;
    std::string expected_source_commit;
    std::string expected_config_sha256;
    std::string expected_model_sha256;
    std::size_t steps = 0;
    std::size_t batch_size = 0;
    std::size_t eval_samples = 0;
    std::size_t eval_batch = 0;
    uint64_t seed = 0;
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
        std::cout
            << "usage: min_vacancy_realizability --checkpoint DIR --config FILE --replay DIR "
               "--arm head|full --device cpu|cuda|cuda:N --steps N --batch-size N "
               "--eval-samples N --eval-batch N --seed N --expected-source-commit SHA "
               "--expected-config-sha256 SHA --expected-model-sha256 SHA --out FILE "
               "[--checkpoint-out DIR]\n";
        std::exit(0);
    }
    Options result;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        if (++index == argc)
            throw std::invalid_argument(key + " requires a value");
        const std::string value = argv[index];
        if (key == "--checkpoint") result.checkpoint = value;
        else if (key == "--config") result.config = value;
        else if (key == "--replay") result.replay = value;
        else if (key == "--arm") result.arm = value;
        else if (key == "--device") result.device = value;
        else if (key == "--steps") result.steps = count(value, key);
        else if (key == "--batch-size") result.batch_size = count(value, key);
        else if (key == "--eval-samples") result.eval_samples = count(value, key);
        else if (key == "--eval-batch") result.eval_batch = count(value, key);
        else if (key == "--seed") result.seed = count(value, key);
        else if (key == "--expected-source-commit") result.expected_source_commit = value;
        else if (key == "--expected-config-sha256") result.expected_config_sha256 = value;
        else if (key == "--expected-model-sha256") result.expected_model_sha256 = value;
        else if (key == "--out") result.out = value;
        else if (key == "--checkpoint-out") result.checkpoint_out = value;
        else throw std::invalid_argument("unknown argument: " + key);
    }
    if (result.arm != "head" && result.arm != "full")
        throw std::invalid_argument("arm must be head or full");
    if (result.checkpoint.empty() || result.config.empty() || result.replay.empty() ||
        result.device.empty() || result.out.empty() || result.expected_source_commit.empty() ||
        result.expected_config_sha256.empty() || result.expected_model_sha256.empty())
        throw std::invalid_argument("all required arguments must be provided");
    if (!result.steps || !result.batch_size || !result.eval_samples || !result.eval_batch)
        throw std::invalid_argument("all counts must be non-zero");
    return result;
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("cannot open " + path.string());
    return {std::istreambuf_iterator<char>(input), {}};
}

std::filesystem::path replay_root(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path / "manifest.json"))
        return path;
    const auto schema = path.parent_path().parent_path();
    if (schema.filename() != "persistent-replay-v2")
        throw std::invalid_argument("replay namespace is not persistent-replay-v2");
    return schema.parent_path();
}

Object metrics(const diamond_pipeline::VacancyMetrics& value) {
    return {{"legal_kl", Json{value.legal_kl}},
            {"expected_progress", Json{value.expected_progress}},
            {"teacher_expected_progress", Json{value.teacher_expected_progress}},
            {"expected_progress_ratio", Json{value.expected_progress_ratio}},
            {"top1_agreement", Json{value.top1_agreement}},
            {"top3_teacher_mass", Json{value.top3_teacher_mass}}};
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        const auto provenance = diamond_support::parse_json(diamond_support::build_provenance_json());
        const auto& provenance_object = std::get<Object>(provenance.value);
        const auto source_commit = std::get<std::string>(provenance_object.at("source_commit").value);
        if (source_commit != options.expected_source_commit)
            throw std::runtime_error("source commit mismatch");
        const auto* dirty = std::get_if<bool>(&provenance_object.at("dirty").value);
        if (!dirty || *dirty)
            throw std::runtime_error("source build is dirty");

        const auto config_json = diamond_support::parse_json(read_file(options.config));
        const auto canonical_config = diamond_support::canonical_json(config_json);
        if (diamond_support::sha256(canonical_config) != options.expected_config_sha256)
            throw std::runtime_error("config digest mismatch");
        const auto config = diamond_orchestration::ProductionConfig::from_json(config_json);
        if (config.model_name != "Min")
            throw std::invalid_argument("config must be Min");

        soo::ensure_topology_configured();
        const auto device = diamond_training::resolve_device(options.device);
        const auto compatibility = diamond_training::Compatibility::min(
            config.model_version,
            {.residual_blocks = config.network.residual_blocks, .width = config.network.width});
        auto model = diamond_model::DiamondModel(config.network.width,
                                                 config.network.residual_blocks, 6, 3);
        diamond_training::Trainer trainer(
            model, compatibility,
            {.learning_rate = config.training.learning_rate,
             .weight_decay = config.training.weight_decay,
             .policy_loss_domain = diamond_training::PolicyLossDomain::legal},
            device);
        const auto checkpoint = diamond_training::load_checkpoint_v3(
            options.checkpoint, trainer, device, diamond_training::CheckpointLoadIntent::exact_resume);
        if (checkpoint.model_digest != options.expected_model_sha256)
            throw std::runtime_error("checkpoint model digest mismatch");

        const diamond_pipeline::ReplayStore replay(
            replay_root(options.replay), compatibility, config.replay.capacity, config.replay.seed,
            diamond_pipeline::ReplayContents::full, diamond_pipeline::ReplayOpenMode::must_exist);
        const auto training_samples = replay.sample(options.batch_size * options.steps, options.seed);
        const auto held_out = replay.sample(options.eval_samples, options.seed ^ 0xd1a69057ULL);
        const auto result = diamond_pipeline::run_vacancy_distillation(
            trainer, training_samples, held_out,
            {.arm = options.arm == "head" ? diamond_pipeline::DistillationArm::policy_head
                                           : diamond_pipeline::DistillationArm::trunk_policy,
             .steps = options.steps,
             .batch_size = options.batch_size,
             .evaluation_batch = options.eval_batch});

        std::optional<diamond_training::CheckpointInfo> saved;
        if (!options.checkpoint_out.empty()) {
            saved = diamond_training::save_checkpoint_v3(
                options.checkpoint_out, trainer,
                {.initialization_mode = diamond_training::CheckpointInitializationMode::resume,
                 .run_id = "vacancy-realizability-" + options.arm,
                 .iteration = 0,
                 .model_step = trainer.training_step(),
                 .parent_digest = checkpoint.model_digest,
                 .source_digest = checkpoint.model_digest,
                 .source_training_step = checkpoint.training_step,
                 .optimizer_restored = true},
                {.source_git_commit = source_commit,
                 .resolved_config_bytes = canonical_config,
                 .replay_manifest_sha256 = replay.manifest_digest(),
                 .protocol_ids_json = "{\"diagnostic\":\"vacancy-realizability-v1\"}"});
        }

        Object report{{"schema_version", Json{int64_t{1}}},
                      {"arm", Json{options.arm}},
                      {"device", Json{device.canonical_name}},
                      {"seed", Json{static_cast<int64_t>(options.seed)}},
                      {"steps", Json{static_cast<int64_t>(options.steps)}},
                      {"batch_size", Json{static_cast<int64_t>(options.batch_size)}},
                      {"eval_samples", Json{static_cast<int64_t>(options.eval_samples)}},
                      {"eval_batch", Json{static_cast<int64_t>(options.eval_batch)}},
                      {"source_commit", Json{source_commit}},
                      {"source_dirty", Json{false}},
                      {"config_sha256", Json{options.expected_config_sha256}},
                      {"input_model_sha256", Json{checkpoint.model_digest}},
                      {"input_training_step", Json{static_cast<int64_t>(checkpoint.training_step)}},
                      {"output_training_step", Json{static_cast<int64_t>(trainer.training_step())}},
                      {"replay_manifest_sha256", Json{replay.manifest_digest()}},
                      {"initial", Json{metrics(result.initial)}},
                      {"final", Json{metrics(result.final)}},
                      {"policy_kl", Json{result.policy_kl}},
                      {"output_model_sha256", saved ? Json{saved->model_digest} : Json{nullptr}}};
        std::ofstream output(options.out, std::ios::binary | std::ios::trunc);
        if (!output)
            throw std::runtime_error("cannot open output report");
        output << diamond_support::canonical_json(Json{std::move(report)}) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "min_vacancy_realizability: " << error.what() << '\n';
        return 2;
    }
}
