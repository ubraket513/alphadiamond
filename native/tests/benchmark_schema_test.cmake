cmake_minimum_required(VERSION 3.21)

foreach(required IN ITEMS
        TRAINING_BENCHMARK CHECKPOINT_BENCHMARK REPLAY_BENCHMARK
        SELFPLAY_BENCHMARK END_TO_END_BENCHMARK MODEL_ARTIFACT SCRATCH)
    if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
        message(FATAL_ERROR "benchmark schema test is missing -D${required}")
    endif()
endforeach()

foreach(executable IN ITEMS
        TRAINING_BENCHMARK CHECKPOINT_BENCHMARK REPLAY_BENCHMARK
        SELFPLAY_BENCHMARK END_TO_END_BENCHMARK)
    if(NOT EXISTS "${${executable}}")
        message(FATAL_ERROR "benchmark executable is missing: ${${executable}}")
    endif()
endforeach()
if(NOT EXISTS "${MODEL_ARTIFACT}/metadata.json")
    message(FATAL_ERROR "Soo benchmark artifact is missing: ${MODEL_ARTIFACT}")
endif()

file(REMOVE_RECURSE "${SCRATCH}")
file(MAKE_DIRECTORY "${SCRATCH}")

function(run_json output executable)
    execute_process(
        COMMAND "${executable}" ${ARGN}
        RESULT_VARIABLE result
        OUTPUT_VARIABLE stdout
        ERROR_VARIABLE stderr
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR
            "benchmark failed (${result}): ${executable}\nstdout:\n${stdout}\nstderr:\n${stderr}")
    endif()
    string(JSON schema_version GET "${stdout}" schema_version)
    if(NOT schema_version EQUAL 1)
        message(FATAL_ERROR "benchmark emitted an unsupported schema: ${executable}")
    endif()
    set(${output} "${stdout}" PARENT_SCOPE)
endfunction()

function(check_common json expected_name)
    string(JSON actual_name GET "${json}" benchmark)
    string(JSON repetitions GET "${json}" workload repetitions)
    string(JSON sample_count LENGTH "${json}" samples_seconds)
    string(JSON source_commit GET "${json}" provenance source_commit)
    string(JSON dirty_type TYPE "${json}" provenance dirty)
    foreach(field IN ITEMS min median max range)
        string(JSON summary_value GET "${json}" summary_seconds ${field})
    endforeach()
    if(NOT actual_name STREQUAL expected_name)
        message(FATAL_ERROR "benchmark name mismatch: ${actual_name} != ${expected_name}")
    endif()
    if(NOT sample_count EQUAL repetitions OR sample_count LESS 1)
        message(FATAL_ERROR "${expected_name} sample/repetition count mismatch")
    endif()
    if(source_commit STREQUAL "" OR NOT dirty_type STREQUAL "BOOLEAN")
        message(FATAL_ERROR "${expected_name} provenance is incomplete")
    endif()
endfunction()

function(check_sha256 value label)
    string(LENGTH "${value}" digest_length)
    if(NOT digest_length EQUAL 64 OR NOT value MATCHES "^[0-9a-f]+$")
        message(FATAL_ERROR "${label} is not a lowercase SHA-256: ${value}")
    endif()
endfunction()

run_json(training_json "${TRAINING_BENCHMARK}"
    --artifact "${MODEL_ARTIFACT}" --device cpu --batch-size 2
    --warmups 0 --repetitions 1 --threads 1)
check_common("${training_json}" training_step)
string(JSON training_model_sha GET "${training_json}" domain model_sha256)
string(JSON training_runtime_sha GET "${training_json}" domain runtime_sha256)
string(JSON training_step GET "${training_json}" domain training_step)
string(JSON cuda_memory GET "${training_json}" domain peak_cuda_memory_available)
check_sha256("${training_model_sha}" "training model digest")
check_sha256("${training_runtime_sha}" "training runtime digest")
if(NOT training_step EQUAL 1 OR cuda_memory)
    message(FATAL_ERROR "training benchmark CPU contract mismatch")
endif()

run_json(checkpoint_json "${CHECKPOINT_BENCHMARK}"
    --scratch "${SCRATCH}/checkpoint" --repetitions 1)
check_common("${checkpoint_json}" checkpoint)
string(JSON checkpoint_operation GET "${checkpoint_json}" domain operation)
string(JSON checkpoint_model_sha GET "${checkpoint_json}" domain model_sha256)
string(JSON checkpoint_optimizer_sha GET "${checkpoint_json}" domain optimizer_sha256)
string(JSON checkpoint_step GET "${checkpoint_json}" domain training_step)
check_sha256("${checkpoint_model_sha}" "checkpoint model digest")
check_sha256("${checkpoint_optimizer_sha}" "checkpoint optimizer digest")
if(NOT checkpoint_operation STREQUAL "save_validate_v3" OR NOT checkpoint_step EQUAL 0)
    message(FATAL_ERROR "checkpoint benchmark v3 contract mismatch")
endif()

run_json(replay_json "${REPLAY_BENCHMARK}"
    --scratch "${SCRATCH}/replay" --repetitions 1 --pool-size 8 --batch-size 4)
check_common("${replay_json}" replay)
string(JSON selection_slots GET "${replay_json}" domain selection_slots)
if(selection_slots LESS 1 OR selection_slots GREATER 4)
    message(FATAL_ERROR "replay benchmark selection accounting mismatch")
endif()

run_json(selfplay_json "${SELFPLAY_BENCHMARK}"
    --artifact "${MODEL_ARTIFACT}" --device cpu --lanes 2 --threads 1
    --max-batch 2 --max-wait-us 200 --simulations 1 --max-moves 2
    --warmups 0 --repetitions 1 --scratch "${SCRATCH}/selfplay")
check_common("${selfplay_json}" selfplay)
string(JSON selfplay_model_sha GET "${selfplay_json}" domain model_sha256)
string(JSON selfplay_runtime_sha GET "${selfplay_json}" domain runtime_sha256)
string(JSON attempted GET "${selfplay_json}" domain attempted_episodes)
string(JSON completed GET "${selfplay_json}" domain completed_episodes)
string(JSON aborted GET "${selfplay_json}" domain aborted_episodes)
string(JSON evaluations GET "${selfplay_json}" domain evaluations)
string(JSON batches GET "${selfplay_json}" domain batches)
string(JSON batch_max GET "${selfplay_json}" domain batch_max)
check_sha256("${selfplay_model_sha}" "self-play model digest")
check_sha256("${selfplay_runtime_sha}" "self-play runtime digest")
math(EXPR accounted "${completed} + ${aborted}")
if(NOT attempted EQUAL 2 OR NOT accounted EQUAL attempted OR evaluations LESS 1
   OR batches LESS 1 OR batch_max GREATER 2)
    message(FATAL_ERROR "self-play benchmark accounting mismatch")
endif()

run_json(end_to_end_json "${END_TO_END_BENCHMARK}"
    --device cpu --games 4 --lanes 2 --threads 1 --batch-size 2
    --training-steps 2 --warmups 0 --repetitions 1
    --scratch "${SCRATCH}/end-to-end")
check_common("${end_to_end_json}" end_to_end)
foreach(field IN ITEMS
        first_checkpoint_model_digest final_checkpoint_model_digest replay_manifest_digest)
    string(JSON digest GET "${end_to_end_json}" domain ${field})
    check_sha256("${digest}" "end-to-end ${field}")
endforeach()
set(accounting_fields
    completed_games aborted_games new_samples requested_training_steps
    completed_training_steps resume_step final_step)
set(accounting_expected 8 0 8 4 4 2 4)
foreach(index RANGE 0 6)
    list(GET accounting_fields ${index} field)
    list(GET accounting_expected ${index} expected)
    string(JSON actual GET "${end_to_end_json}" domain ${field})
    if(NOT actual EQUAL expected)
        message(FATAL_ERROR "end-to-end ${field} mismatch: ${actual} != ${expected}")
    endif()
endforeach()
string(JSON boundary_restart GET "${end_to_end_json}" domain boundary_restart_resume)
string(JSON duplicate_work GET "${end_to_end_json}" domain duplicate_durable_work)
string(JSON batch_count LENGTH "${end_to_end_json}" domain training_batch_sizes)
if(NOT boundary_restart OR duplicate_work OR NOT batch_count EQUAL 4)
    message(FATAL_ERROR "end-to-end resume contract mismatch")
endif()
foreach(batch_index RANGE 0 3)
    string(JSON batch_size GET "${end_to_end_json}" domain training_batch_sizes ${batch_index})
    if(NOT batch_size EQUAL 2)
        message(FATAL_ERROR "end-to-end training batch accounting mismatch")
    endif()
endforeach()

file(WRITE "${SCRATCH}/benchmark-contract-results.json"
    "{\"schema_version\":1,\"benchmarks\":[${training_json},${checkpoint_json},"
    "${replay_json},${selfplay_json},${end_to_end_json}]}\n")
message(STATUS "benchmark JSON contracts passed (5/5)")
