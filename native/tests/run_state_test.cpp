#include <filesystem>
#include <array>
#include <string>

#include "check.hpp"
#include "diamond_orchestration/run_state.hpp"

namespace {

diamond_orchestration::RunState initial_state() {
    return diamond_orchestration::RunState::initialize(
        "native-resume", "Soo", "sha256:compatibility", "protocol-v1", 17);
}

}  // namespace

int main(int argc, char** argv) {
    REQUIRE(argc == 2, "usage: run_state_test <scratch>");
    const auto scratch = std::filesystem::path(argv[1]);
    std::filesystem::remove_all(scratch);

    diamond_orchestration::RunStateStore store(scratch);
    auto state = store.initialize(initial_state());
    CHECK_EQ(state.generation(), uint64_t{0});
    CHECK_EQ(state.stage(), diamond_orchestration::RunStage::initialize);
    CHECK_EQ(state.payload().size(), std::size_t{19});
    CHECK(state.payload().contains("protocol_ids"));
    CHECK(!state.payload().contains("protocol_id"));

    state = store.transition(state, diamond_orchestration::RunStage::self_play, "initialize-v1");
    CHECK_EQ(state.generation(), uint64_t{1});
    CHECK_EQ(state.stage(), diamond_orchestration::RunStage::self_play);
    CHECK_EQ(store.load("Soo", "native-resume").generation(), uint64_t{1});

    const auto seed = state.derive_seed({diamond_support::JsonValue{std::string("self-play")},
                                         diamond_support::JsonValue{int64_t{3}}});
    CHECK_EQ(seed, state.derive_seed({diamond_support::JsonValue{std::string("self-play")},
                                      diamond_support::JsonValue{int64_t{3}}}));
    auto changed = state.with_progress({{"training_step", diamond_support::JsonValue{int64_t{23}}}});
    changed = store.save(changed);
    CHECK_EQ(changed.generation(), uint64_t{2});
    CHECK_EQ(std::get<int64_t>(changed.payload().at("training_step").value), int64_t{23});

    bool saw_stale = false;
    try {
        (void)store.transition(initial_state(), diamond_orchestration::RunStage::self_play, "stale");
    } catch (const diamond_orchestration::RunStateError&) {
        saw_stale = true;
    }
    CHECK(saw_stale);

    constexpr std::array remaining = {
        diamond_orchestration::RunStage::replay_ingest,
        diamond_orchestration::RunStage::train,
        diamond_orchestration::RunStage::save_candidate,
        diamond_orchestration::RunStage::promotion_arena,
        diamond_orchestration::RunStage::rating_benchmark,
        diamond_orchestration::RunStage::promote_or_reject,
        diamond_orchestration::RunStage::persist,
        diamond_orchestration::RunStage::complete,
    };
    for (const auto next : remaining) changed = store.transition(changed, next, "completed-v2");
    const auto next_iteration = store.start_next_iteration(changed);
    CHECK_EQ(next_iteration.stage(), diamond_orchestration::RunStage::initialize);
    CHECK_EQ(next_iteration.generation(), changed.generation() + 1);
    CHECK_EQ(std::get<int64_t>(next_iteration.payload().at("iteration").value), int64_t{1});
    CHECK_EQ(std::get<int64_t>(next_iteration.payload().at("training_step").value), int64_t{23});
    return soo_test::report("run_state_test");
}
