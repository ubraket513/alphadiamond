#include "diamond_training/checkpoint.hpp"

#include "diamond_support/json.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace diamond_training {
namespace {
constexpr char kPointer[] = "CURRENT";
constexpr char kState[] = "state.pt";
constexpr char kOptimizer[] = "optimizer.pt";
constexpr char kManifest[] = "manifest.json";

using NamedTensors = std::map<std::string, torch::Tensor>;
using Json = diamond_support::JsonValue;
using Object = Json::Object;

static_assert(std::is_nothrow_move_assignable_v<diamond_model::DiamondModel>,
              "transactional weight restore requires a noexcept model commit");
static_assert(std::is_nothrow_move_assignable_v<Trainer>,
              "transactional trainer restore requires a noexcept trainer commit");

void require_target(const ResolvedDevice& target) {
    if (target.canonical_name != target.torch_device.str())
        throw CheckpointError("checkpoint target device is not canonical");
    if (target.torch_device.is_cuda()) {
        if (!target.cuda_index || target.torch_device.index() != *target.cuda_index)
            throw CheckpointError("checkpoint CUDA target index is inconsistent");
    } else if (!target.torch_device.is_cpu() || target.cuda_index) {
        throw CheckpointError("checkpoint target must be CPU or CUDA");
    }
}

NamedTensors collect_named_tensors(const diamond_model::DiamondModel& model, bool parameters) {
    if (!model) throw CheckpointError("checkpoint model destination is empty");
    NamedTensors result;
    const auto named = parameters ? model->named_parameters() : model->named_buffers();
    for (const auto& entry : named) {
        if (!entry.value().defined())
            throw CheckpointError("checkpoint model contains an undefined tensor: " + entry.key());
        if (!result.emplace(entry.key(), entry.value()).second)
            throw CheckpointError("checkpoint model contains a duplicate tensor: " + entry.key());
    }
    return result;
}

void require_finite_target_tensor(const torch::Tensor& tensor, const torch::Device& target,
                                  const std::string& name) {
    if (!tensor.defined()) throw CheckpointError(name + " is undefined");
    if (tensor.device() != target)
        throw CheckpointError(name + " is on " + tensor.device().str() +
                              ", expected " + target.str());
    if (tensor.scalar_type() != torch::kFloat32)
        throw CheckpointError(name + " must be float32");
    if (!torch::isfinite(tensor).all().item<bool>())
        throw CheckpointError(name + " must be finite");
}

void require_model_matches(const diamond_model::DiamondModel& model,
                           const diamond_model::DiamondModel& prototype,
                           const torch::Device& target) {
    if (!model || !prototype) throw CheckpointError("checkpoint model destination is empty");
    if (model->width() != prototype->width() ||
        model->residual_blocks() != prototype->residual_blocks() ||
        model->input_features() != prototype->input_features() ||
        model->value_size() != prototype->value_size()) {
        throw CheckpointError("checkpoint model architecture changed during restore");
    }
    if (model->is_training() != prototype->is_training())
        throw CheckpointError("checkpoint model runtime role changed during restore");

    const auto expected_parameters = collect_named_tensors(prototype, true);
    const auto actual_parameters = collect_named_tensors(model, true);
    const auto expected_buffers = collect_named_tensors(prototype, false);
    const auto actual_buffers = collect_named_tensors(model, false);
    if (expected_parameters.size() != actual_parameters.size() ||
        expected_buffers.size() != actual_buffers.size()) {
        throw CheckpointError("checkpoint model tensor inventory is incompatible");
    }

    for (const auto& [name, expected] : expected_parameters) {
        const auto actual = actual_parameters.find(name);
        if (actual == actual_parameters.end() || actual->second.sizes() != expected.sizes() ||
            actual->second.scalar_type() != expected.scalar_type() ||
            actual->second.requires_grad() != expected.requires_grad()) {
            throw CheckpointError("checkpoint model parameter is incompatible: " + name);
        }
        require_finite_target_tensor(actual->second, target,
                                     "checkpoint model parameter " + name);
    }
    for (const auto& [name, expected] : expected_buffers) {
        const auto actual = actual_buffers.find(name);
        if (actual == actual_buffers.end() || actual->second.sizes() != expected.sizes() ||
            actual->second.scalar_type() != expected.scalar_type()) {
            throw CheckpointError("checkpoint model buffer is incompatible: " + name);
        }
        require_finite_target_tensor(actual->second, target,
                                     "checkpoint model buffer " + name);
    }
}

diamond_model::DiamondModel fresh_model_like(const diamond_model::DiamondModel& prototype,
                                             const torch::Device& target) {
    if (!prototype) throw CheckpointError("checkpoint model destination is empty");
    auto staged = diamond_model::DiamondModel(
        prototype->width(), prototype->residual_blocks(),
        prototype->input_features(), prototype->value_size());
    staged->to(target);
    if (prototype->is_training()) staged->train();
    else staged->eval();

    const auto expected_parameters = collect_named_tensors(prototype, true);
    auto staged_parameters = collect_named_tensors(staged, true);
    if (expected_parameters.size() != staged_parameters.size())
        throw CheckpointError("checkpoint model parameter inventory is incompatible");
    for (const auto& [name, expected] : expected_parameters) {
        const auto actual = staged_parameters.find(name);
        if (actual == staged_parameters.end())
            throw CheckpointError("checkpoint model parameter is missing: " + name);
        actual->second.set_requires_grad(expected.requires_grad());
    }
    return staged;
}

void load_model_archive(const std::filesystem::path& path,
                        diamond_model::DiamondModel& model,
                        const torch::Device& target) {
    torch::serialize::InputArchive archive;
    archive.load_from(path.string(), target);
    model->load(archive);
}

void load_optimizer_archive(const std::filesystem::path& path,
                            torch::optim::AdamW& optimizer,
                            const torch::Device& target) {
    torch::serialize::InputArchive archive;
    archive.load_from(path.string(), target);
    optimizer.load(archive);
}

void require_adamw_state_tensor(const torch::Tensor& tensor,
                                const torch::Tensor& parameter,
                                const torch::Device& target,
                                const std::string& name) {
    require_finite_target_tensor(tensor, target, name);
    if (tensor.sizes() != parameter.sizes() ||
        tensor.scalar_type() != parameter.scalar_type()) {
        throw CheckpointError(name + " is incompatible with its parameter");
    }
}

void validate_staged_trainer(Trainer& trainer, const ResolvedDevice& target,
                             uint64_t training_step,
                             const diamond_model::DiamondModel& prototype) {
    trainer.compatibility().validate();
    if (trainer.device().torch_device != target.torch_device ||
        trainer.device().canonical_name != target.canonical_name ||
        trainer.device().cuda_index != target.cuda_index) {
        throw CheckpointError("staged trainer device does not match checkpoint target");
    }
    if (trainer.training_step() != training_step)
        throw CheckpointError("staged trainer training_step is inconsistent");
    require_model_matches(trainer.model(), prototype, target.torch_device);

    auto& optimizer = trainer.optimizer();
    auto& groups = optimizer.param_groups();
    const auto parameters = trainer.model()->parameters();
    if (groups.size() != 1 || groups.front().params().size() != parameters.size())
        throw CheckpointError("checkpoint AdamW parameter groups are incompatible");

    const auto* options =
        dynamic_cast<const torch::optim::AdamWOptions*>(&groups.front().options());
    if (!options) throw CheckpointError("checkpoint optimizer is not AdamW");
    const auto [beta1, beta2] = options->betas();
    if (!std::isfinite(options->lr()) || options->lr() != trainer.config().learning_rate ||
        !std::isfinite(options->weight_decay()) ||
        options->weight_decay() != trainer.config().weight_decay ||
        !std::isfinite(options->eps()) || options->eps() <= 0.0 ||
        !std::isfinite(beta1) || !std::isfinite(beta2) ||
        beta1 < 0.0 || beta1 >= 1.0 || beta2 < 0.0 || beta2 >= 1.0) {
        throw CheckpointError("checkpoint AdamW options are incompatible");
    }

    for (std::size_t index = 0; index < parameters.size(); ++index) {
        if (groups.front().params()[index].unsafeGetTensorImpl() !=
            parameters[index].unsafeGetTensorImpl()) {
            throw CheckpointError("checkpoint AdamW is not bound to the staged learner");
        }
    }

    const auto& states = optimizer.state();
    if (training_step == 0) {
        if (!states.empty())
            throw CheckpointError("step-zero checkpoint contains AdamW state");
        return;
    }
    if (training_step > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
        states.size() != parameters.size()) {
        throw CheckpointError("checkpoint AdamW state inventory is incompatible");
    }

    for (const auto& parameter : parameters) {
        const auto state_entry = states.find(parameter.unsafeGetTensorImpl());
        if (state_entry == states.end())
            throw CheckpointError("checkpoint AdamW state is missing a parameter");
        const auto* state =
            dynamic_cast<const torch::optim::AdamWParamState*>(state_entry->second.get());
        if (!state || state->step() != static_cast<int64_t>(training_step))
            throw CheckpointError("checkpoint AdamW step is inconsistent");
        require_adamw_state_tensor(state->exp_avg(), parameter, target.torch_device,
                                   "checkpoint AdamW exp_avg");
        require_adamw_state_tensor(state->exp_avg_sq(), parameter, target.torch_device,
                                   "checkpoint AdamW exp_avg_sq");
        if (options->amsgrad()) {
            require_adamw_state_tensor(state->max_exp_avg_sq(), parameter,
                                       target.torch_device,
                                       "checkpoint AdamW max_exp_avg_sq");
        } else if (state->max_exp_avg_sq().defined()) {
            throw CheckpointError("non-AMSGrad checkpoint contains max_exp_avg_sq");
        }
    }
}

std::string read_pointer(const std::filesystem::path& root) {
    if (!std::filesystem::is_directory(root))
        throw CheckpointError("checkpoint v2 root must be a directory; v1 files are rejected");
    std::ifstream input(root / kPointer, std::ios::binary);
    std::string generation;
    std::getline(input, generation);
    if (!input || generation.empty() || generation.find_first_of("\\/.") != std::string::npos)
        throw CheckpointError("checkpoint v2 CURRENT pointer is invalid");
    return generation;
}

void atomic_write(const std::filesystem::path& path, const std::string& contents) {
    static std::atomic_uint64_t sequence{0};
    std::filesystem::create_directories(path.parent_path());
    const auto pid =
#ifdef _WIN32
        static_cast<unsigned long long>(GetCurrentProcessId());
#else
        static_cast<unsigned long long>(::getpid());
#endif
    const auto temporary = path.parent_path() / (".checkpoint-tmp-" + std::to_string(pid) + "-" + std::to_string(++sequence));
    { std::ofstream output(temporary, std::ios::binary | std::ios::trunc); if (!output) throw CheckpointError("cannot write checkpoint transaction"); output << contents; if (!output) throw CheckpointError("cannot write checkpoint transaction"); }
    if (std::getenv("DIAMOND_CHECKPOINT_FAIL_ACTIVATE")) { std::filesystem::remove(temporary); throw CheckpointError("injected checkpoint activation failure"); }
#ifdef _WIN32
    if (!MoveFileExW(temporary.wstring().c_str(), path.wstring().c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) { std::filesystem::remove(temporary); throw CheckpointError("cannot activate checkpoint transaction"); }
#else
    std::error_code error; std::filesystem::rename(temporary, path, error); if (error) { std::filesystem::remove(temporary); throw CheckpointError("cannot activate checkpoint transaction"); }
#endif
}

std::string read_text(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw CheckpointError("cannot read checkpoint manifest");
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

bool is_digest(const std::string& value) {
    return value.size() == 64 && std::all_of(value.begin(), value.end(), [](char character) {
               return (character >= '0' && character <= '9') ||
                      (character >= 'a' && character <= 'f');
           });
}

std::string file_digest(const std::filesystem::path& path) {
    return diamond_support::sha256(read_text(path));
}

const Json& field(const Object& object, const char* name) {
    const auto found = object.find(name);
    if (found == object.end())
        throw CheckpointError("checkpoint v3 manifest is missing " + std::string(name));
    return found->second;
}

const Object& object_field(const Object& object, const char* name) {
    const auto* value = std::get_if<Object>(&field(object, name).value);
    if (!value)
        throw CheckpointError("checkpoint v3 manifest field is not an object: " +
                              std::string(name));
    return *value;
}

std::string string_field(const Object& object, const char* name) {
    const auto* value = std::get_if<std::string>(&field(object, name).value);
    if (!value)
        throw CheckpointError("checkpoint v3 manifest field is not a string: " + std::string(name));
    return *value;
}

uint64_t unsigned_field(const Object& object, const char* name) {
    const auto* value = std::get_if<int64_t>(&field(object, name).value);
    if (!value || *value < 0)
        throw CheckpointError("checkpoint v3 manifest field is not a non-negative integer: " +
                              std::string(name));
    return static_cast<uint64_t>(*value);
}

bool bool_field(const Object& object, const char* name) {
    const auto* value = std::get_if<bool>(&field(object, name).value);
    if (!value)
        throw CheckpointError("checkpoint v3 manifest field is not a boolean: " +
                              std::string(name));
    return *value;
}

std::optional<std::string> nullable_digest_field(const Object& object, const char* name) {
    const Json& value = field(object, name);
    if (std::holds_alternative<std::nullptr_t>(value.value))
        return std::nullopt;
    const auto* text = std::get_if<std::string>(&value.value);
    if (!text || !is_digest(*text))
        throw CheckpointError("checkpoint v3 manifest digest is invalid: " + std::string(name));
    return *text;
}

std::optional<uint64_t> nullable_step_field(const Object& object, const char* name) {
    const Json& value = field(object, name);
    if (std::holds_alternative<std::nullptr_t>(value.value))
        return std::nullopt;
    const auto* step = std::get_if<int64_t>(&value.value);
    if (!step || *step < 0)
        throw CheckpointError("checkpoint v3 manifest step is invalid: " + std::string(name));
    return static_cast<uint64_t>(*step);
}

std::optional<std::string> nullable_string_field(const Object& object, const char* name) {
    const Json& value = field(object, name);
    if (std::holds_alternative<std::nullptr_t>(value.value))
        return std::nullopt;
    const auto* text = std::get_if<std::string>(&value.value);
    if (!text || text->empty())
        throw CheckpointError("checkpoint v3 manifest string is invalid: " + std::string(name));
    return *text;
}

void exact_fields(const Object& object, std::initializer_list<const char*> names) {
    std::set<std::string> expected;
    for (const char* name : names)
        expected.emplace(name);
    if (object.size() != expected.size())
        throw CheckpointError("checkpoint v3 manifest has unknown fields");
    for (const auto& [name, value] : object) {
        (void)value;
        if (!expected.contains(name))
            throw CheckpointError("checkpoint v3 manifest has unknown field: " + name);
    }
}

const char* mode_name(CheckpointInitializationMode mode) {
    switch (mode) {
    case CheckpointInitializationMode::scratch:
        return "scratch";
    case CheckpointInitializationMode::warm_start:
        return "warm_start";
    case CheckpointInitializationMode::resume:
        return "resume";
    case CheckpointInitializationMode::audited_legacy_import:
        return "audited_legacy_import";
    }
    throw CheckpointError("checkpoint v3 initialization mode is invalid");
}

CheckpointInitializationMode parse_mode(const std::string& value) {
    if (value == "scratch")
        return CheckpointInitializationMode::scratch;
    if (value == "warm_start")
        return CheckpointInitializationMode::warm_start;
    if (value == "resume")
        return CheckpointInitializationMode::resume;
    if (value == "audited_legacy_import")
        return CheckpointInitializationMode::audited_legacy_import;
    throw CheckpointError("checkpoint v3 initialization mode is invalid");
}

const char* load_intent_name(CheckpointInitializationMode mode) {
    switch (mode) {
    case CheckpointInitializationMode::scratch:
        return "new_run";
    case CheckpointInitializationMode::warm_start:
        return "warm_start";
    case CheckpointInitializationMode::resume:
        return "exact_resume";
    case CheckpointInitializationMode::audited_legacy_import:
        return "audited_legacy_import";
    }
    throw CheckpointError("checkpoint v3 load intent is invalid");
}

void validate_lineage(const CheckpointLineage& lineage, uint64_t training_step) {
    if (lineage.run_id.empty())
        throw CheckpointError("checkpoint v3 run_id is empty");
    if (lineage.iteration > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
        throw CheckpointError("checkpoint v3 iteration exceeds JSON integer range");
    if (lineage.model_step != training_step)
        throw CheckpointError("checkpoint v3 model_step must equal training_step");
    const bool has_source = lineage.source_digest.has_value();
    if (lineage.source_training_step && !lineage.source_digest)
        throw CheckpointError("checkpoint v3 source training_step requires a source digest");
    if (lineage.parent_digest && !is_digest(*lineage.parent_digest))
        throw CheckpointError("checkpoint v3 parent_digest is invalid");
    if (lineage.source_digest && !is_digest(*lineage.source_digest))
        throw CheckpointError("checkpoint v3 source_digest is invalid");
    if (lineage.optimizer_reset_reason && lineage.optimizer_reset_reason->empty())
        throw CheckpointError("checkpoint v3 optimizer_reset_reason is empty");

    switch (lineage.initialization_mode) {
    case CheckpointInitializationMode::scratch:
        if (lineage.parent_digest || has_source || lineage.source_training_step ||
            lineage.optimizer_reset || lineage.optimizer_restored || lineage.optimizer_reset_reason)
            throw CheckpointError("checkpoint v3 scratch lineage is inconsistent");
        break;
    case CheckpointInitializationMode::warm_start:
        if (!has_source || !lineage.optimizer_reset || lineage.optimizer_restored ||
            !lineage.optimizer_reset_reason)
            throw CheckpointError("checkpoint v3 warm_start lineage is inconsistent");
        break;
    case CheckpointInitializationMode::resume:
        if (!lineage.parent_digest || !has_source || !lineage.source_training_step ||
            !lineage.optimizer_restored || lineage.optimizer_reset ||
            lineage.optimizer_reset_reason)
            throw CheckpointError("checkpoint v3 resume lineage is inconsistent");
        break;
    case CheckpointInitializationMode::audited_legacy_import:
        if (!has_source || !lineage.source_training_step || !lineage.optimizer_reset ||
            lineage.optimizer_restored || !lineage.optimizer_reset_reason)
            throw CheckpointError("checkpoint v3 audited_legacy_import lineage is inconsistent");
        break;
    }
}

Json nullable_json(const std::optional<std::string>& value) {
    return value ? Json{*value} : Json{nullptr};
}

Json nullable_json(const std::optional<uint64_t>& value) {
    if (!value)
        return Json{nullptr};
    if (*value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
        throw CheckpointError("checkpoint v3 step exceeds JSON integer range");
    return Json{static_cast<int64_t>(*value)};
}

Json compatibility_json(const Compatibility& value) {
    value.validate();
    return Json{Object{{"action_space_version", Json{value.action_space_version}},
                       {"board_topology_version", Json{value.board_topology_version}},
                       {"encoder_version", Json{value.encoder_version}},
                       {"model_name", Json{value.model_name}},
                       {"model_version", Json{value.model_version}},
                       {"network_config",
                        Json{Object{{"residual_blocks", Json{value.network_config.residual_blocks}},
                                    {"width", Json{value.network_config.width}}}}},
                       {"player_count", Json{value.player_count}},
                       {"ruleset_fingerprint", Json{value.ruleset_fingerprint}},
                       {"ruleset_version", Json{value.ruleset_version}},
                       {"seat_layout_version", Json{value.seat_layout_version}},
                       {"value_semantics_version", Json{value.value_semantics_version}}}};
}

Json canonical_object_json(const std::string& bytes, const char* name) {
    try {
        const auto parsed = diamond_support::parse_json(bytes);
        if (!std::holds_alternative<Object>(parsed.value) ||
            diamond_support::canonical_json(parsed) != bytes)
            throw CheckpointError(std::string("checkpoint v3 ") + name +
                                  " is not canonical object JSON");
        return parsed;
    } catch (const CheckpointError&) {
        throw;
    } catch (const std::exception& error) {
        throw CheckpointError(std::string("checkpoint v3 ") + name +
                              " is invalid: " + error.what());
    }
}

void validate_provenance(const CheckpointProvenance& value) {
    if (value.source_git_commit.empty() || value.creation_timestamp.empty() ||
        !is_digest(value.replay_manifest_sha256) ||
        value.rng_state_status != "gap_no_stable_libtorch_cpp_api" || value.rng_state_version != 0)
        throw CheckpointError("checkpoint v3 provenance is invalid");
    (void)canonical_object_json(value.resolved_config_bytes, "resolved_config_bytes");
    (void)canonical_object_json(value.protocol_ids_json, "protocol_ids_json");
    (void)canonical_object_json(value.environment_json, "environment_json");
}

Json provenance_json(const CheckpointProvenance& value, const Trainer& trainer) {
    validate_provenance(value);
    const auto& model = trainer.model();
    return Json{Object{
        {"architecture", Json{Object{{"input_features", Json{model->input_features()}},
                                     {"residual_blocks", Json{model->residual_blocks()}},
                                     {"value_size", Json{model->value_size()}},
                                     {"width", Json{model->width()}}}}},
        {"compatibility", compatibility_json(trainer.compatibility())},
        {"creation_timestamp", Json{value.creation_timestamp}},
        {"environment", canonical_object_json(value.environment_json, "environment_json")},
        {"precision", Json{"float32"}},
        {"protocol_ids", canonical_object_json(value.protocol_ids_json, "protocol_ids_json")},
        {"replay_manifest_sha256", Json{value.replay_manifest_sha256}},
        {"resolved_config_bytes", Json{value.resolved_config_bytes}},
        {"resolved_config_sha256", Json{diamond_support::sha256(value.resolved_config_bytes)}},
        {"resolved_device", Json{trainer.device().canonical_name}},
        {"rng_state",
         Json{Object{{"status", Json{value.rng_state_status}},
                     {"version", Json{static_cast<int64_t>(value.rng_state_version)}}}}},
        {"source_git_commit", Json{value.source_git_commit}}}};
}

CheckpointProvenance parse_provenance(const Object& value) {
    exact_fields(value,
                 {"architecture", "compatibility", "creation_timestamp", "environment", "precision",
                  "protocol_ids", "replay_manifest_sha256", "resolved_config_bytes",
                  "resolved_config_sha256", "resolved_device", "rng_state", "source_git_commit"});
    const auto& architecture = object_field(value, "architecture");
    exact_fields(architecture, {"input_features", "residual_blocks", "value_size", "width"});
    (void)unsigned_field(architecture, "input_features");
    (void)unsigned_field(architecture, "residual_blocks");
    (void)unsigned_field(architecture, "value_size");
    (void)unsigned_field(architecture, "width");
    const auto compatibility = compatibility_json(Compatibility{
        .model_name = string_field(object_field(value, "compatibility"), "model_name"),
        .model_version = string_field(object_field(value, "compatibility"), "model_version"),
        .player_count = static_cast<int64_t>(
            unsigned_field(object_field(value, "compatibility"), "player_count")),
        .ruleset_version = string_field(object_field(value, "compatibility"), "ruleset_version"),
        .board_topology_version =
            string_field(object_field(value, "compatibility"), "board_topology_version"),
        .ruleset_fingerprint =
            string_field(object_field(value, "compatibility"), "ruleset_fingerprint"),
        .encoder_version = string_field(object_field(value, "compatibility"), "encoder_version"),
        .action_space_version =
            string_field(object_field(value, "compatibility"), "action_space_version"),
        .seat_layout_version =
            string_field(object_field(value, "compatibility"), "seat_layout_version"),
        .value_semantics_version =
            string_field(object_field(value, "compatibility"), "value_semantics_version"),
        .network_config = {
            .residual_blocks = static_cast<int64_t>(
                unsigned_field(object_field(object_field(value, "compatibility"), "network_config"),
                               "residual_blocks")),
            .width = static_cast<int64_t>(unsigned_field(
                object_field(object_field(value, "compatibility"), "network_config"), "width"))}});
    if (diamond_support::canonical_json(field(value, "compatibility")) !=
        diamond_support::canonical_json(compatibility))
        throw CheckpointError("checkpoint v3 provenance compatibility is invalid");
    CheckpointProvenance result{
        .source_git_commit = string_field(value, "source_git_commit"),
        .resolved_config_bytes = string_field(value, "resolved_config_bytes"),
        .replay_manifest_sha256 = string_field(value, "replay_manifest_sha256"),
        .protocol_ids_json = diamond_support::canonical_json(field(value, "protocol_ids")),
        .creation_timestamp = string_field(value, "creation_timestamp"),
        .environment_json = diamond_support::canonical_json(field(value, "environment")),
        .rng_state_status = string_field(object_field(value, "rng_state"), "status"),
        .rng_state_version = unsigned_field(object_field(value, "rng_state"), "version")};
    validate_provenance(result);
    if (string_field(value, "precision") != "float32" ||
        string_field(value, "resolved_device").empty() ||
        string_field(value, "resolved_config_sha256") !=
            diamond_support::sha256(result.resolved_config_bytes))
        throw CheckpointError("checkpoint v3 provenance is inconsistent");
    return result;
}

Json lineage_json(const CheckpointLineage& lineage) {
    validate_lineage(lineage, lineage.model_step);
    return Json{
        Object{{"initialization_mode", Json{std::string(mode_name(lineage.initialization_mode))}},
               {"iteration", Json{static_cast<int64_t>(lineage.iteration)}},
               {"load_intent", Json{std::string(load_intent_name(lineage.initialization_mode))}},
               {"model_step", Json{static_cast<int64_t>(lineage.model_step)}},
               {"optimizer_reset", Json{lineage.optimizer_reset}},
               {"optimizer_reset_reason", nullable_json(lineage.optimizer_reset_reason)},
               {"optimizer_restored", Json{lineage.optimizer_restored}},
               {"parent_digest", nullable_json(lineage.parent_digest)},
               {"run_id", Json{lineage.run_id}},
               {"source_digest", nullable_json(lineage.source_digest)},
               {"source_training_step", nullable_json(lineage.source_training_step)}}};
}

CheckpointLineage parse_lineage(const Object& object, uint64_t training_step) {
    exact_fields(object, {"initialization_mode", "iteration", "load_intent", "model_step",
                          "optimizer_reset", "optimizer_reset_reason", "optimizer_restored",
                          "parent_digest", "run_id", "source_digest", "source_training_step"});
    CheckpointLineage lineage{
        .initialization_mode = parse_mode(string_field(object, "initialization_mode")),
        .run_id = string_field(object, "run_id"),
        .iteration = unsigned_field(object, "iteration"),
        .model_step = unsigned_field(object, "model_step"),
        .parent_digest = nullable_digest_field(object, "parent_digest"),
        .source_digest = nullable_digest_field(object, "source_digest"),
        .source_training_step = nullable_step_field(object, "source_training_step"),
        .optimizer_restored = bool_field(object, "optimizer_restored"),
        .optimizer_reset = bool_field(object, "optimizer_reset"),
        .optimizer_reset_reason = nullable_string_field(object, "optimizer_reset_reason"),
    };
    if (string_field(object, "load_intent") != load_intent_name(lineage.initialization_mode))
        throw CheckpointError("checkpoint v3 load_intent is inconsistent");
    validate_lineage(lineage, training_step);
    return lineage;
}

CheckpointInfo state_info(const std::filesystem::path& generation) {
    const auto state = generation / kState;
    const auto optimizer = generation / kOptimizer;
    if (!std::filesystem::is_regular_file(state) || !std::filesystem::is_regular_file(optimizer))
        throw CheckpointError("checkpoint generation is incomplete");
    std::ifstream input(generation / "training_step", std::ios::binary);
    uint64_t step = 0;
    if (!(input >> step) || !(input >> std::ws).eof())
        throw CheckpointError("checkpoint training_step is invalid");
    CheckpointInfo result{.generation = generation, .training_step = step};
    const auto manifest_path = generation / kManifest;
    if (!std::filesystem::exists(manifest_path))
        return result;
    try {
        const auto parsed = diamond_support::parse_json(read_text(manifest_path));
        const auto* manifest = std::get_if<Object>(&parsed.value);
        if (!manifest)
            throw CheckpointError("checkpoint manifest is not an object");
        const auto version = unsigned_field(*manifest, "format_version");
        if (version == 2)
            return result;
        if (version != 3)
            throw CheckpointError("checkpoint manifest format_version is unsupported");
        exact_fields(*manifest, {"format_version", "generation", "lineage", "model_digest",
                                 "model_step", "optimizer_digest", "provenance", "training_step"});
        if (string_field(*manifest, "generation") != generation.filename().string() ||
            unsigned_field(*manifest, "training_step") != step ||
            unsigned_field(*manifest, "model_step") != step)
            throw CheckpointError("checkpoint v3 manifest progress is inconsistent");
        result.format_version = 3;
        result.lineage = parse_lineage(object_field(*manifest, "lineage"), step);
        result.provenance = parse_provenance(object_field(*manifest, "provenance"));
        result.model_digest = string_field(*manifest, "model_digest");
        result.optimizer_digest = string_field(*manifest, "optimizer_digest");
        if (!is_digest(result.model_digest) || !is_digest(result.optimizer_digest) ||
            result.model_digest != file_digest(state) ||
            result.optimizer_digest != file_digest(optimizer))
            throw CheckpointError("checkpoint v3 manifest digest mismatch");
        return result;
    } catch (const CheckpointError&) {
        throw;
    } catch (const std::exception& error) {
        throw CheckpointError(std::string("checkpoint manifest is invalid: ") + error.what());
    }
}

void validate_provenance_for_trainer(const CheckpointInfo& info, const Trainer& trainer,
                                     const ResolvedDevice& target) {
    if (info.format_version != 3)
        return;
    (void)target; // Save-device provenance is retained; checkpoints are device-portable.
    const auto parsed = diamond_support::parse_json(read_text(info.generation / kManifest));
    const auto& provenance = object_field(*std::get_if<Object>(&parsed.value), "provenance");
    const auto& architecture = object_field(provenance, "architecture");
    const auto& model = trainer.model();
    if (unsigned_field(architecture, "width") != static_cast<uint64_t>(model->width()) ||
        unsigned_field(architecture, "residual_blocks") !=
            static_cast<uint64_t>(model->residual_blocks()) ||
        unsigned_field(architecture, "input_features") !=
            static_cast<uint64_t>(model->input_features()) ||
        unsigned_field(architecture, "value_size") != static_cast<uint64_t>(model->value_size()) ||
        diamond_support::canonical_json(field(provenance, "compatibility")) !=
            diamond_support::canonical_json(compatibility_json(trainer.compatibility())))
        throw CheckpointError("checkpoint v3 provenance is incompatible with destination trainer");
}
}  // namespace

CheckpointInfo inspect_checkpoint_v2(const std::filesystem::path& root) {
    const auto generation = read_pointer(root);
    return state_info(root / "generations" / generation);
}

CheckpointInfo validate_checkpoint_v2(const std::filesystem::path& root) {
    const auto info = inspect_checkpoint_v2(root);
    try {
        torch::serialize::InputArchive state;
        state.load_from((info.generation / kState).string(), torch::Device(torch::kCPU));
        torch::serialize::InputArchive optimizer;
        optimizer.load_from((info.generation / kOptimizer).string(),
                            torch::Device(torch::kCPU));
        return info;
    } catch (const c10::Error& error) {
        throw CheckpointError(std::string("checkpoint v2 archive is unreadable: ") + error.what());
    } catch (const std::exception& error) {
        throw CheckpointError(std::string("checkpoint v2 archive is unreadable: ") + error.what());
    }
}

CheckpointInfo migrate_checkpoint_v2(const std::filesystem::path& source,
                                     const std::filesystem::path& destination) {
    const auto info = validate_checkpoint_v2(source);
    if (std::filesystem::exists(destination))
        throw CheckpointError("checkpoint migration destination already exists");

    const auto parent = destination.has_parent_path()
        ? destination.parent_path() : std::filesystem::current_path();
    const auto filename = destination.filename().string();
    if (filename.empty()) throw CheckpointError("checkpoint migration destination is invalid");
    static std::atomic_uint64_t sequence{0};
    const auto staged = parent / ("." + filename + ".checkpoint-migrate-" +
                                  std::to_string(++sequence));
    std::error_code ignored;
    if (std::filesystem::exists(staged))
        throw CheckpointError("checkpoint migration staging path already exists");
    try {
        std::filesystem::create_directories(staged / "generations");
        const auto generation = info.generation.filename();
        std::filesystem::copy(info.generation, staged / "generations" / generation,
                              std::filesystem::copy_options::recursive);
        atomic_write(staged / kPointer, generation.string() + "\n");
        std::filesystem::rename(staged, destination);
        return {destination / "generations" / generation, info.training_step};
    } catch (const std::filesystem::filesystem_error& error) {
        std::filesystem::remove_all(staged, ignored);
        throw CheckpointError(std::string("cannot migrate checkpoint v2: ") + error.what());
    } catch (...) {
        std::filesystem::remove_all(staged, ignored);
        throw;
    }
}

CheckpointInfo save_checkpoint_v2(const std::filesystem::path& root, Trainer& trainer) {
    trainer.compatibility().validate();
    std::filesystem::create_directories(root / "generations");
    static std::atomic_uint64_t sequence{0};
    const std::string generation = "generation-" + std::to_string(trainer.training_step()) + "-" + std::to_string(++sequence);
    const auto staged = root / "generations" / ("." + generation + ".staging");
    const auto committed = root / "generations" / generation;
    std::error_code ignored; std::filesystem::remove_all(staged, ignored); std::filesystem::create_directories(staged);
    try {
        // Native LibTorch archives make this generation self-contained and
        // deliberately distinct from the legacy Python checkpoint reader.
        torch::save(trainer.model(), (staged / kState).string());
        torch::save(trainer.optimizer(), (staged / kOptimizer).string());
        {
            std::ofstream step(staged / "training_step", std::ios::binary | std::ios::trunc);
            step << trainer.training_step();
            if (!step) throw CheckpointError("cannot write checkpoint training_step");
        }
        {
            std::ofstream manifest(staged / "manifest.json", std::ios::binary | std::ios::trunc);
            manifest << "{\"format_version\":2,\"generation\":\"" << generation << "\",\"training_step\":" << trainer.training_step() << "}";
            if (!manifest) throw CheckpointError("cannot write checkpoint manifest");
        }
#ifdef _WIN32
        if (!MoveFileExW(staged.wstring().c_str(), committed.wstring().c_str(),
                         MOVEFILE_WRITE_THROUGH)) {
            const DWORD error = GetLastError();
            throw CheckpointError("cannot promote checkpoint generation " + staged.string() + " -> " + committed.string() + " (Win32=" + std::to_string(error) + ")");
        }
#else
        std::filesystem::rename(staged, committed);
#endif
        atomic_write(root / kPointer, generation + "\n");
        return {committed, trainer.training_step()};
    } catch (const CheckpointError&) { std::filesystem::remove_all(staged, ignored); throw; }
      catch (const c10::Error& error) { std::filesystem::remove_all(staged, ignored); throw CheckpointError(std::string("cannot save checkpoint v2: ") + error.what()); }
      catch (const std::filesystem::filesystem_error& error) { std::filesystem::remove_all(staged, ignored); throw CheckpointError(error.what()); }
}

CheckpointInfo save_checkpoint_v3(const std::filesystem::path& root, Trainer& trainer,
                                  const CheckpointLineage& lineage,
                                  const CheckpointProvenance& provenance) {
    trainer.compatibility().validate();
    if (trainer.training_step() > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
        throw CheckpointError("checkpoint v3 training_step exceeds JSON integer range");
    validate_lineage(lineage, trainer.training_step());
    if (lineage.initialization_mode == CheckpointInitializationMode::audited_legacy_import)
        throw CheckpointError(
            "audited_legacy_import is gated pending optimizer decode and parity evidence");
    std::filesystem::create_directories(root / "generations");
    static std::atomic_uint64_t sequence{0};
    const std::string generation = "generation-v3-" + std::to_string(trainer.training_step()) +
                                   "-" + std::to_string(++sequence);
    const auto staged = root / "generations" / ("." + generation + ".staging");
    const auto committed = root / "generations" / generation;
    std::error_code ignored;
    std::filesystem::remove_all(staged, ignored);
    std::filesystem::create_directories(staged);
    try {
        torch::save(trainer.model(), (staged / kState).string());
        torch::save(trainer.optimizer(), (staged / kOptimizer).string());
        const auto model_digest = file_digest(staged / kState);
        const auto optimizer_digest = file_digest(staged / kOptimizer);
        {
            std::ofstream step(staged / "training_step", std::ios::binary | std::ios::trunc);
            step << trainer.training_step();
            if (!step)
                throw CheckpointError("cannot write checkpoint training_step");
        }
        {
            const Json manifest{Object{
                {"format_version", Json{int64_t{3}}},
                {"generation", Json{generation}},
                {"lineage", lineage_json(lineage)},
                {"model_digest", Json{model_digest}},
                {"model_step", Json{static_cast<int64_t>(trainer.training_step())}},
                {"optimizer_digest", Json{optimizer_digest}},
                {"provenance", provenance_json(provenance, trainer)},
                {"training_step", Json{static_cast<int64_t>(trainer.training_step())}},
            }};
            std::ofstream manifest_file(staged / kManifest, std::ios::binary | std::ios::trunc);
            manifest_file << diamond_support::canonical_json(manifest);
            if (!manifest_file)
                throw CheckpointError("cannot write checkpoint v3 manifest");
        }
#ifdef _WIN32
        if (!MoveFileExW(staged.wstring().c_str(), committed.wstring().c_str(),
                         MOVEFILE_WRITE_THROUGH)) {
            const DWORD error = GetLastError();
            throw CheckpointError("cannot promote checkpoint generation " + staged.string() +
                                  " -> " + committed.string() + " (Win32=" + std::to_string(error) +
                                  ")");
        }
#else
        std::filesystem::rename(staged, committed);
#endif
        atomic_write(root / kPointer, generation + "\n");
        return {.generation = committed,
                .training_step = trainer.training_step(),
                .format_version = 3,
                .lineage = lineage,
                .model_digest = model_digest,
                .optimizer_digest = optimizer_digest,
                .provenance = provenance};
    } catch (const CheckpointError&) {
        std::filesystem::remove_all(staged, ignored);
        throw;
    } catch (const c10::Error& error) {
        std::filesystem::remove_all(staged, ignored);
        throw CheckpointError(std::string("cannot save checkpoint v3: ") + error.what());
    } catch (const std::filesystem::filesystem_error& error) {
        std::filesystem::remove_all(staged, ignored);
        throw CheckpointError(error.what());
    }
}

CheckpointInfo load_checkpoint_v2_weights(const std::filesystem::path& root,
                                          diamond_model::DiamondModel& model,
                                          const ResolvedDevice& target) {
    if (!model) throw CheckpointError("checkpoint model destination is empty");
    require_target(target);
    const auto info = inspect_checkpoint_v2(root);
    try {
        // The caller-visible holder is replaced only after the entire archive
        // has loaded and passed target-residency/structure validation.
        auto staged = fresh_model_like(model, target.torch_device);
        load_model_archive(info.generation / kState, staged, target.torch_device);
        require_model_matches(staged, model, target.torch_device);
        model = std::move(staged);
        return info;
    } catch (const CheckpointError&) {
        throw;
    } catch (const c10::Error& error) {
        throw CheckpointError(std::string("cannot load checkpoint model weights: ") + error.what());
    } catch (const std::exception& error) {
        throw CheckpointError(std::string("cannot load checkpoint model weights: ") + error.what());
    }
}

CheckpointInfo load_checkpoint_v2(const std::filesystem::path& root, Trainer& trainer,
                                  const ResolvedDevice& target) {
    require_target(target);
    const auto info = inspect_checkpoint_v2(root);
    if (info.lineage &&
        info.lineage->initialization_mode == CheckpointInitializationMode::audited_legacy_import) {
        throw CheckpointError("audited legacy imports cannot be used for exact resume");
    }
    validate_provenance_for_trainer(info, trainer, target);
    try {
        // AdamW must be constructed against the final target-device parameter
        // identities before its pointer-keyed state archive is restored.
        const auto compatibility = trainer.compatibility();
        const auto config = trainer.config();
        auto model = fresh_model_like(trainer.model(), target.torch_device);
        Trainer staged(std::move(model), compatibility, config, target);
        load_model_archive(info.generation / kState, staged.model(), target.torch_device);
        load_optimizer_archive(info.generation / kOptimizer, staged.optimizer(),
                               target.torch_device);
        staged.restore_checkpoint_state(config, info.training_step);
        validate_staged_trainer(staged, target, info.training_step, trainer.model());
        trainer = std::move(staged);
        return info;
    } catch (const CheckpointError&) {
        throw;
    } catch (const c10::Error& error) {
        throw CheckpointError(std::string("cannot load checkpoint v2: ") + error.what());
    } catch (const std::exception& error) {
        throw CheckpointError(std::string("cannot load checkpoint v2: ") + error.what());
    }
}

CheckpointInfo load_checkpoint_v3(const std::filesystem::path& root, Trainer& trainer,
                                  const ResolvedDevice& target, CheckpointLoadIntent intent) {
    if (intent == CheckpointLoadIntent::audited_legacy_import)
        throw CheckpointError(
            "audited_legacy_import is gated pending optimizer decode and parity evidence");
    if (intent == CheckpointLoadIntent::exact_resume) {
        if (inspect_checkpoint_v2(root).format_version != 3)
            throw CheckpointError("exact_resume requires a checkpoint v3 lineage manifest");
        return load_checkpoint_v2(root, trainer, target);
    }

    require_target(target);
    const auto info = inspect_checkpoint_v2(root);
    try {
        // A warm start creates a new step-zero trainer. The source optimizer
        // archive is deliberately never read, so it cannot leak state.
        const auto compatibility = trainer.compatibility();
        const auto config = trainer.config();
        auto model = fresh_model_like(trainer.model(), target.torch_device);
        Trainer staged(std::move(model), compatibility, config, target);
        load_model_archive(info.generation / kState, staged.model(), target.torch_device);
        staged.restore_checkpoint_state(config, 0);
        validate_staged_trainer(staged, target, 0, trainer.model());
        trainer = std::move(staged);
        return info;
    } catch (const CheckpointError&) {
        throw;
    } catch (const c10::Error& error) {
        throw CheckpointError(std::string("cannot warm_start checkpoint v3: ") + error.what());
    } catch (const std::exception& error) {
        throw CheckpointError(std::string("cannot warm_start checkpoint v3: ") + error.what());
    }
}

}  // namespace diamond_training
