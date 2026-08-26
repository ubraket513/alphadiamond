#include <filesystem>
#include <array>
#include <fstream>
#include <iterator>
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

    const std::string operation_id =
        "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const diamond_support::JsonValue operation_result{
        diamond_support::JsonValue::Object{{"accepted", diamond_support::JsonValue{int64_t{7}}}}};
    CHECK(!store.load_operation_result(state, operation_id));
    store.commit_operation_result(state, operation_id, operation_result);
    CHECK_EQ(diamond_support::canonical_json(*store.load_operation_result(state, operation_id)),
             diamond_support::canonical_json(operation_result));
    store.commit_operation_result(state, operation_id, operation_result);
    std::ifstream ledger(scratch / "soo" / "native-resume" / "ledger.jsonl");
    std::string ledger_event;
    CHECK(static_cast<bool>(std::getline(ledger, ledger_event)));
    CHECK(!std::getline(ledger, ledger_event));
    ledger.close();
    {
        std::ofstream torn(scratch / "soo" / "native-resume" / "ledger.jsonl",
                           std::ios::binary | std::ios::app);
        torn << "{\"torn\"";
    }
    store.commit_operation_result(state, operation_id, operation_result);
    {
        std::ifstream repaired(scratch / "soo" / "native-resume" / "ledger.jsonl",
                               std::ios::binary);
        const std::string contents((std::istreambuf_iterator<char>(repaired)), {});
        CHECK(contents.ends_with('\n'));
        CHECK(contents.find("torn") == std::string::npos);
    }
    bool saw_conflict = false;
    try {
        store.commit_operation_result(state, operation_id,
                                      diamond_support::JsonValue{diamond_support::JsonValue::Object{
                                          {"accepted", diamond_support::JsonValue{int64_t{8}}}}});
    } catch (const diamond_orchestration::RunStateError&) {
        saw_conflict = true;
    }
    CHECK(saw_conflict);

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
