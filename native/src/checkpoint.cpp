#include "diamond_training/checkpoint.hpp"

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
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

using NamedTensors = std::map<std::string, torch::Tensor>;

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

CheckpointInfo state_info(const std::filesystem::path& generation) {
    const auto state = generation / kState;
    const auto optimizer = generation / kOptimizer;
    if (!std::filesystem::is_regular_file(state) || !std::filesystem::is_regular_file(optimizer))
        throw CheckpointError("checkpoint generation is incomplete");
    std::ifstream input(generation / "training_step", std::ios::binary);
    uint64_t step = 0;
    if (!(input >> step)) throw CheckpointError("checkpoint training_step is invalid");
    return {generation, step};
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
    try {
        // AdamW must be constructed against the final target-device parameter
        // identities before its pointer-keyed state archive is restored.
        const auto compatibility = trainer.compatibility();
        const auto config = trainer.config();
        Trainer staged(trainer.model(), compatibility, config, target);
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

}  // namespace diamond_training
