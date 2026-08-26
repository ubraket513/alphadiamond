#include <filesystem>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "check.hpp"
#include "diamond_orchestration/coordinator.hpp"

namespace {

class Interrupted final {};

diamond_orchestration::RunState initial_state() {
    return diamond_orchestration::RunState::initialize(
        "native-coordinator", "Soo", "sha256:compatibility", "protocol-v1", 17);
}

diamond_support::JsonValue stage_payload(
    diamond_orchestration::RunStage stage,
    const diamond_orchestration::RunState& state) {
    return diamond_support::JsonValue{diamond_support::JsonValue::Object{
        {"stage", diamond_support::JsonValue{static_cast<int64_t>(stage)}},
        {"training_step", state.payload().at("training_step")},
    }};
}

diamond_orchestration::StageOutcome
stage_outcome(diamond_orchestration::RunStage stage, const diamond_orchestration::RunState& state,
              diamond_support::JsonValue::Object progress = {}) {
    return {.result = stage_payload(stage, state), .progress = std::move(progress)};
}

}  // namespace

int main(int argc, char** argv) {
    REQUIRE(argc == 2, "usage: coordinator_resume_test <scratch>");
    const auto scratch = std::filesystem::path(argv[1]);
    std::filesystem::remove_all(scratch);

    diamond_orchestration::RunStateStore store(scratch);
    const auto initial = store.initialize(initial_state());
    std::vector<diamond_orchestration::RunStage> first_attempt;
    std::string interrupted_operation;

    diamond_orchestration::Coordinator interrupted(
        store, stage_payload, [&](diamond_orchestration::RunStage stage,
                   const diamond_orchestration::RunState& state,
                   const std::string& operation_id) {
            first_attempt.push_back(stage);
            CHECK_EQ(operation_id, diamond_orchestration::Coordinator::operation_id(
                                       state, stage_payload(stage, state)));
            CHECK(operation_id.starts_with("sha256:"));
            CHECK_EQ(operation_id.size(), std::size_t{71});
            if (stage == diamond_orchestration::RunStage::train) {
                interrupted_operation = operation_id;
                throw Interrupted{};
            }
            return stage_outcome(stage, state);
        });

    bool saw_interruption = false;
    try {
        (void)interrupted.run(initial);
    } catch (const Interrupted&) {
        saw_interruption = true;
    }
    CHECK(saw_interruption);
    CHECK_EQ(first_attempt.size(), std::size_t{4});
    CHECK_EQ(store.load("Soo", "native-coordinator").stage(),
             diamond_orchestration::RunStage::train);
    CHECK_EQ(store.load("Soo", "native-coordinator").generation(), uint64_t{3});

    std::vector<diamond_orchestration::RunStage> resumed_stages;
    std::string resumed_train_operation;
    diamond_orchestration::Coordinator resumed(
        store, stage_payload,
        [&](diamond_orchestration::RunStage stage, const diamond_orchestration::RunState& state,
            const std::string& operation_id) {
            resumed_stages.push_back(stage);
            if (stage == diamond_orchestration::RunStage::train)
                resumed_train_operation = operation_id;
            return stage_outcome(stage, state);
        });
    const auto complete = resumed.resume("Soo", "native-coordinator");

    CHECK_EQ(resumed_train_operation, interrupted_operation);
    CHECK_EQ(resumed_stages.size(), std::size_t{6});
    CHECK_EQ(resumed_stages.front(), diamond_orchestration::RunStage::train);
    CHECK_EQ(complete.stage(), diamond_orchestration::RunStage::complete);
    CHECK_EQ(complete.generation(), uint64_t{9});
    const auto& completions = std::get<diamond_support::JsonValue::Object>(
        complete.payload().at("stage_completions").value);
    CHECK_EQ(completions.size(), std::size_t{9});
    CHECK_EQ(std::get<std::string>(completions.at("TRAIN").value), interrupted_operation);

    diamond_orchestration::RunStateStore activated_store(scratch / "activated");
    const auto activated_initial = activated_store.initialize(initial_state());
    bool interrupted_after_result = false;
    diamond_orchestration::Coordinator activated(
        activated_store, stage_payload,
        [&](diamond_orchestration::RunStage stage, const diamond_orchestration::RunState& state,
            const std::string& operation_id) {
            if (stage != diamond_orchestration::RunStage::train)
                return stage_outcome(stage, state);
            const auto outcome = stage_outcome(stage, state);
            activated_store.commit_operation_result(
                state, operation_id, diamond_orchestration::Coordinator::durable_result(outcome));
            throw Interrupted{};
        });
    try {
        (void)activated.run(activated_initial);
    } catch (const Interrupted&) {
        interrupted_after_result = true;
    }
    CHECK(interrupted_after_result);
    std::vector<diamond_orchestration::RunStage> activated_resume_calls;
    diamond_orchestration::Coordinator activated_resume(
        activated_store, stage_payload,
        [&](diamond_orchestration::RunStage stage, const diamond_orchestration::RunState& state,
            const std::string&) {
            activated_resume_calls.push_back(stage);
            return stage_outcome(stage, state);
        });
    const auto activated_complete = activated_resume.resume("Soo", "native-coordinator");
    CHECK_EQ(activated_complete.stage(), diamond_orchestration::RunStage::complete);
    CHECK_EQ(activated_resume_calls.size(), std::size_t{5});
    CHECK_EQ(activated_resume_calls.front(), diamond_orchestration::RunStage::save_candidate);

    diamond_orchestration::RunStateStore bounded_store(scratch / "bounded");
    const auto bounded_initial = bounded_store.initialize(initial_state());
    std::vector<std::string> bounded_operations;
    diamond_orchestration::Coordinator bounded(
        bounded_store, stage_payload,
        [&](diamond_orchestration::RunStage stage, const diamond_orchestration::RunState& state,
            const std::string& operation_id) {
            bounded_operations.push_back(operation_id);
            diamond_support::JsonValue::Object progress;
            if (stage == diamond_orchestration::RunStage::train) {
                const auto iteration = std::get<int64_t>(state.payload().at("iteration").value);
                progress.emplace("training_step", diamond_support::JsonValue{iteration + 1});
            }
            return stage_outcome(stage, state, std::move(progress));
        });
    const auto two_iterations = bounded.run_bounded(bounded_initial, 2, std::nullopt);
    CHECK_EQ(two_iterations.stage(), diamond_orchestration::RunStage::complete);
    CHECK_EQ(std::get<int64_t>(two_iterations.payload().at("iteration").value), int64_t{1});
    CHECK_EQ(two_iterations.generation(), uint64_t{19});
    CHECK_EQ(std::get<int64_t>(two_iterations.payload().at("training_step").value), int64_t{2});
    CHECK_EQ(bounded_operations.size(), std::size_t{18});
    CHECK_EQ(std::set<std::string>(bounded_operations.begin(), bounded_operations.end()).size(),
             bounded_operations.size());
    return soo_test::report("coordinator_resume_test");
}
