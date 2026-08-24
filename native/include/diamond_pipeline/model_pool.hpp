#pragma once

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <map>
#include <optional>
#include <stdexcept>
#include <stop_token>

#include "diamond_model/soo_model.hpp"
#include "diamond_pipeline/replay.hpp"
#include "soo/selfplay.hpp"

namespace diamond_pipeline {

class PipelineError : public std::runtime_error { public: using std::runtime_error::runtime_error; };
class CancelledError final : public PipelineError { public: using PipelineError::PipelineError; };
class DeadlineExceededError final : public PipelineError { public: using PipelineError::PipelineError; };
class IncompatibleCheckpointError final : public PipelineError { public: using PipelineError::PipelineError; };

class ModelPool final : public soo::BatchEvaluator {
  public:
    explicit ModelPool(std::size_t capacity);

    void install(ModelKey key, diamond_model::DiamondModel model);
    void install_checkpoint(ModelKey key, const std::filesystem::path& checkpoint_root,
                            diamond_model::DiamondModel model);
    void activate(const ModelKey& key);
    const ModelKey& active_key() const;
    std::size_t resident_count() const;
    void require_ready(std::stop_token stop, std::chrono::steady_clock::time_point deadline) const;
    void evaluate(std::vector<soo::BatchItem>& batch) override;

  private:
    std::size_t capacity_;
    std::map<ModelKey, diamond_model::DiamondModel> models_;
    std::optional<ModelKey> active_;
};

}  // namespace diamond_pipeline
