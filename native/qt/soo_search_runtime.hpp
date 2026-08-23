#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "search_telemetry.hpp"
#include "soo/state.hpp"

class SooSearchRuntime final {
  public:
    SooSearchRuntime();
    ~SooSearchRuntime();

    SooSearchRuntime(const SooSearchRuntime&) = delete;
    SooSearchRuntime& operator=(const SooSearchRuntime&) = delete;

    AiSearchResult search(const soo::State& state, const soo::Match& match,
                          const std::vector<int32_t>& rejected, int simulations);

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

