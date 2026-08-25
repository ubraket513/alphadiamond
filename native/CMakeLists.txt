option(DIAMOND_BUILD_BENCHMARKS "Build explicit native benchmark executables" OFF)
option(DIAMOND_ENABLE_CUDA_TESTS "Register tests that require a real CUDA-enabled LibTorch host" OFF)

set(SOO_CORE_SOURCES
    src/board.cpp
    src/encoder.cpp
    src/prior.cpp
    src/rules.cpp
    src/topology_gen.cpp
    src/topology_io.cpp
)

set(SOO_SEARCH_SOURCES
    src/batcher.cpp
    src/evaluator.cpp
    src/mcts.cpp
    src/mcts3p.cpp
    src/profile.cpp
    src/selfplay.cpp
)

add_library(soo_core STATIC ${SOO_CORE_SOURCES})
add_library(alphadiamond::soo_core ALIAS soo_core)

target_include_directories(soo_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(soo_core PUBLIC cxx_std_20)
target_compile_options(soo_core PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

add_library(soo_search STATIC ${SOO_SEARCH_SOURCES})
add_library(alphadiamond::soo_search ALIAS soo_search)

target_link_libraries(soo_search
    PUBLIC
        alphadiamond::soo_core
        Threads::Threads
)

target_include_directories(soo_search
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(soo_search PUBLIC cxx_std_20)
target_compile_options(soo_search PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

add_library(diamond_pipeline STATIC
    src/json.cpp
    src/replay.cpp
    src/replay_store.cpp
    src/training_sample.cpp
)
add_library(alphadiamond::diamond_pipeline ALIAS diamond_pipeline)
target_include_directories(diamond_pipeline PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_features(diamond_pipeline PUBLIC cxx_std_20)
target_compile_options(diamond_pipeline PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

if(BUILD_TESTING AND DIAMOND_BUILD_NATIVE_TESTS)
    # Golden fixtures are committed data, not build output: the tests below run
    # on a machine with no Python, no pybind and no pytest.
    set(DIAMOND_GOLDEN_DIR "${PROJECT_SOURCE_DIR}/tests/golden")

    function(diamond_add_native_test name)
        add_executable(${name} tests/${name}.cpp)
        target_link_libraries(${name} PRIVATE alphadiamond::soo_search)
        target_compile_features(${name} PRIVATE cxx_std_20)
        target_compile_options(${name} PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra>
            $<$<CXX_COMPILER_ID:MSVC>:/W4>
        )
        add_test(NAME ${name} COMMAND ${name} ${ARGN})
    endfunction()

    diamond_add_native_test(action_codec_test)
    diamond_add_native_test(topology_test "${DIAMOND_GOLDEN_DIR}/topology")
    diamond_add_native_test(rules_golden_test
        "${DIAMOND_GOLDEN_DIR}" "${DIAMOND_GOLDEN_DIR}/topology")
    diamond_add_native_test(mcts_golden_test
        "${DIAMOND_GOLDEN_DIR}" "${DIAMOND_GOLDEN_DIR}/topology")
    diamond_add_native_test(mcts_stochastic_test
        "${DIAMOND_GOLDEN_DIR}" "${DIAMOND_GOLDEN_DIR}/topology")
    diamond_add_native_test(mcts3p_golden_test
        "${DIAMOND_GOLDEN_DIR}" "${DIAMOND_GOLDEN_DIR}/topology")
    diamond_add_native_test(budget_test
        "${DIAMOND_GOLDEN_DIR}" "${DIAMOND_GOLDEN_DIR}/topology")
    diamond_add_native_test(batcher_test)
    diamond_add_native_test(selfplay_test "${DIAMOND_GOLDEN_DIR}")
    diamond_add_native_test(replay_schema_test "${DIAMOND_GOLDEN_DIR}/replay-v1" "${CMAKE_CURRENT_BINARY_DIR}/replay-schema-scratch")
    diamond_add_native_test(replay_store_test "${DIAMOND_GOLDEN_DIR}/replay-v1" "${CMAKE_CURRENT_BINARY_DIR}/replay-store-scratch")
    target_link_libraries(replay_schema_test PRIVATE alphadiamond::diamond_pipeline)
    target_link_libraries(replay_store_test PRIVATE alphadiamond::diamond_pipeline)

    add_executable(golden_manifest_test tests/golden_manifest_test.cpp)
    target_link_libraries(golden_manifest_test PRIVATE alphadiamond::diamond_pipeline)
    target_compile_features(golden_manifest_test PRIVATE cxx_std_20)
    add_test(NAME golden_manifest_test
        COMMAND golden_manifest_test "${DIAMOND_GOLDEN_DIR}")
    set_tests_properties(golden_manifest_test PROPERTIES LABELS contract)
endif()

if(DIAMOND_BUILD_LIBTORCH)
    find_package(Torch REQUIRED)
    add_library(diamond_model STATIC
        src/deployment_artifact.cpp
        src/json.cpp
        src/soo_model.cpp
        src/soo_evaluator.cpp
    )
    add_library(alphadiamond::diamond_model ALIAS diamond_model)
    target_link_libraries(diamond_model PUBLIC alphadiamond::soo_core ${TORCH_LIBRARIES})
    target_include_directories(diamond_model
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            ${TORCH_INCLUDE_DIRS}
    )
    target_compile_features(diamond_model PUBLIC cxx_std_20)
    target_compile_definitions(diamond_model PUBLIC _DISABLE_EXTENDED_ALIGNED_STORAGE)
    if(MSVC)
        target_compile_options(diamond_model PRIVATE /bigobj)
    endif()

    add_library(diamond_training STATIC
        src/checkpoint.cpp
        src/device.cpp
        src/training_sample.cpp
        src/trainer.cpp
    )
    add_library(alphadiamond::diamond_training ALIAS diamond_training)
    target_link_libraries(diamond_training PUBLIC alphadiamond::diamond_model ${TORCH_LIBRARIES})
    target_include_directories(diamond_training
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            ${TORCH_INCLUDE_DIRS}
    )
    target_compile_features(diamond_training PUBLIC cxx_std_20)
    target_compile_definitions(diamond_training PUBLIC _DISABLE_EXTENDED_ALIGNED_STORAGE)
    if(MSVC)
        target_compile_options(diamond_training PRIVATE /bigobj)
    endif()

    add_library(diamond_release STATIC src/promotion.cpp)
    add_library(alphadiamond::diamond_release ALIAS diamond_release)
    target_link_libraries(diamond_release PUBLIC alphadiamond::diamond_training)
    target_include_directories(diamond_release PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )
    target_compile_features(diamond_release PUBLIC cxx_std_20)

    add_executable(alphadiamond-checkpoint src/checkpoint_main.cpp)
    target_link_libraries(alphadiamond-checkpoint PRIVATE alphadiamond::diamond_training)
    target_compile_features(alphadiamond-checkpoint PRIVATE cxx_std_20)

    add_executable(alphadiamond-release src/release_main.cpp)
    target_link_libraries(alphadiamond-release PRIVATE alphadiamond::diamond_release)
    target_compile_features(alphadiamond-release PRIVATE cxx_std_20)

    install(TARGETS alphadiamond-checkpoint alphadiamond-release
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

    # The direct native self-play -> replay -> training path is deliberately
    # separate from the schema-only replay library: it owns LibTorch models.
    add_library(diamond_native_pipeline STATIC
        src/model_pool.cpp
        src/pipeline.cpp
    )
    add_library(alphadiamond::diamond_native_pipeline ALIAS diamond_native_pipeline)
    target_link_libraries(diamond_native_pipeline PUBLIC
        alphadiamond::diamond_pipeline
        alphadiamond::diamond_training
        alphadiamond::soo_search
        ${TORCH_LIBRARIES}
    )
    target_include_directories(diamond_native_pipeline PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        ${TORCH_INCLUDE_DIRS}
    )
    target_compile_features(diamond_native_pipeline PUBLIC cxx_std_20)
    target_compile_definitions(diamond_native_pipeline PUBLIC _DISABLE_EXTENDED_ALIGNED_STORAGE)
    if(MSVC)
        target_compile_options(diamond_native_pipeline PRIVATE /bigobj)
    endif()

    add_library(diamond_orchestration STATIC
        src/arena.cpp
        src/config.cpp
        src/coordinator.cpp
        src/rating.cpp
        src/rating_store.cpp
        src/report.cpp
        src/run_state.cpp
        src/schedule.cpp
        src/training_wiring.cpp
    )
    add_library(alphadiamond::diamond_orchestration ALIAS diamond_orchestration)
    target_link_libraries(diamond_orchestration PUBLIC alphadiamond::diamond_pipeline)
    target_include_directories(diamond_orchestration PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )
    target_compile_features(diamond_orchestration PUBLIC cxx_std_20)

    if(DIAMOND_BUILD_BENCHMARKS)
        add_executable(training_step_benchmark benchmarks/training_step_benchmark.cpp)
        target_link_libraries(training_step_benchmark PRIVATE alphadiamond::diamond_training)

        add_executable(checkpoint_benchmark benchmarks/checkpoint_benchmark.cpp)
        target_link_libraries(checkpoint_benchmark PRIVATE alphadiamond::diamond_training)

        add_executable(replay_benchmark benchmarks/replay_benchmark.cpp)
        target_link_libraries(replay_benchmark PRIVATE alphadiamond::diamond_pipeline)

        add_executable(selfplay_benchmark benchmarks/selfplay_benchmark.cpp)
        target_link_libraries(selfplay_benchmark PRIVATE alphadiamond::soo_search)

        add_executable(end_to_end_benchmark benchmarks/end_to_end_benchmark.cpp)
        target_link_libraries(end_to_end_benchmark PRIVATE
            alphadiamond::diamond_native_pipeline)

        foreach(benchmark IN ITEMS
                training_step_benchmark checkpoint_benchmark replay_benchmark
                selfplay_benchmark end_to_end_benchmark)
            target_compile_features(${benchmark} PRIVATE cxx_std_20)
        endforeach()
    endif()

    add_executable(alphadiamond-train src/train_main.cpp)
    target_link_libraries(alphadiamond-train PRIVATE
        alphadiamond::diamond_orchestration
        alphadiamond::diamond_native_pipeline)
    target_compile_features(alphadiamond-train PRIVATE cxx_std_20)
    install(TARGETS alphadiamond-train
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

    add_executable(alphadiamond-rating-sync src/rating_sync_main.cpp)
    target_link_libraries(alphadiamond-rating-sync PRIVATE
        alphadiamond::diamond_orchestration)
    target_compile_features(alphadiamond-rating-sync PRIVATE cxx_std_20)
    install(TARGETS alphadiamond-rating-sync
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

    add_executable(alphadiamond-pipeline src/pipeline_main.cpp)
    target_link_libraries(alphadiamond-pipeline PRIVATE alphadiamond::diamond_native_pipeline)
    target_compile_features(alphadiamond-pipeline PRIVATE cxx_std_20)

    add_executable(soo_libtorch_probe src/soo_libtorch_probe.cpp)
    target_link_libraries(soo_libtorch_probe PRIVATE alphadiamond::diamond_model alphadiamond::soo_search ${TORCH_LIBRARIES})
    target_compile_features(soo_libtorch_probe PRIVATE cxx_std_20)
    target_include_directories(soo_libtorch_probe PRIVATE ${TORCH_INCLUDE_DIRS})
    target_compile_definitions(soo_libtorch_probe PRIVATE _DISABLE_EXTENDED_ALIGNED_STORAGE)
    if(MSVC)
        target_compile_options(soo_libtorch_probe PRIVATE /bigobj)
    endif()

    add_executable(soo_evaluator_probe src/soo_evaluator_probe.cpp)
    target_link_libraries(soo_evaluator_probe PRIVATE alphadiamond::diamond_model)
    target_compile_features(soo_evaluator_probe PRIVATE cxx_std_20)
    target_compile_definitions(soo_evaluator_probe PRIVATE _DISABLE_EXTENDED_ALIGNED_STORAGE)
    if(MSVC)
        target_compile_options(soo_evaluator_probe PRIVATE /bigobj)
    endif()

    add_executable(soo_mcts_probe src/soo_mcts_probe.cpp)
    target_link_libraries(soo_mcts_probe PRIVATE alphadiamond::diamond_model alphadiamond::soo_search)
    target_compile_features(soo_mcts_probe PRIVATE cxx_std_20)
    target_compile_definitions(soo_mcts_probe PRIVATE _DISABLE_EXTENDED_ALIGNED_STORAGE)
    if(MSVC)
        target_compile_options(soo_mcts_probe PRIVATE /bigobj)
    endif()

    if(BUILD_TESTING)
        # Deployment artifacts are produced by the native release pipeline.
        # The tests register only for the artifacts that are present, so a
        # checkout without them configures cleanly instead of failing late.
        set(DIAMOND_SOO_ARTIFACT "${PROJECT_SOURCE_DIR}/artifacts/soo-spike")
        set(DIAMOND_MIN_ARTIFACT "${PROJECT_SOURCE_DIR}/artifacts/min-spike")

        add_executable(soo_artifact_contract tests/artifact_contract.cpp)
        target_link_libraries(soo_artifact_contract PRIVATE alphadiamond::diamond_model)
        target_compile_features(soo_artifact_contract PRIVATE cxx_std_20)

        add_executable(model_index_test tests/model_index_test.cpp)
        target_link_libraries(model_index_test PRIVATE alphadiamond::diamond_model)
        target_include_directories(model_index_test PRIVATE tests)
        target_compile_features(model_index_test PRIVATE cxx_std_20)
        add_test(NAME model_index_test
                 COMMAND model_index_test "${CMAKE_CURRENT_BINARY_DIR}/model-index-scratch")

        add_executable(promotion_test tests/promotion_test.cpp)
        target_link_libraries(promotion_test PRIVATE alphadiamond::diamond_release)
        target_compile_features(promotion_test PRIVATE cxx_std_20)
        add_test(NAME promotion_test
                 COMMAND promotion_test "${CMAKE_CURRENT_BINARY_DIR}/promotion-scratch")
        set_tests_properties(promotion_test PROPERTIES LABELS release)

        add_executable(final_pipeline_smoke_test tests/final_pipeline_smoke_test.cpp)
        target_link_libraries(final_pipeline_smoke_test PRIVATE
            alphadiamond::diamond_release
            alphadiamond::diamond_orchestration)
        target_include_directories(final_pipeline_smoke_test PRIVATE tests ${TORCH_INCLUDE_DIRS})
        target_compile_features(final_pipeline_smoke_test PRIVATE cxx_std_20)
        target_compile_definitions(final_pipeline_smoke_test PRIVATE _DISABLE_EXTENDED_ALIGNED_STORAGE)
        if(MSVC)
            target_compile_options(final_pipeline_smoke_test PRIVATE /bigobj)
        endif()

        add_executable(model_parity_test tests/model_parity_test.cpp)
        target_link_libraries(model_parity_test PRIVATE alphadiamond::diamond_model ${TORCH_LIBRARIES})
        target_include_directories(model_parity_test PRIVATE ${TORCH_INCLUDE_DIRS} tests)
        target_compile_features(model_parity_test PRIVATE cxx_std_20)
        target_compile_definitions(model_parity_test PRIVATE _DISABLE_EXTENDED_ALIGNED_STORAGE)
        if(MSVC)
            target_compile_options(model_parity_test PRIVATE /bigobj)
        endif()

        if(EXISTS "${DIAMOND_SOO_ARTIFACT}/metadata.json")
            if(EXISTS "${DIAMOND_MIN_ARTIFACT}/metadata.json")
                add_test(NAME soo_artifact_contract
                         COMMAND soo_artifact_contract
                                 "${DIAMOND_SOO_ARTIFACT}" "${DIAMOND_MIN_ARTIFACT}")
                add_test(NAME model_parity_test
                         COMMAND model_parity_test
                                 soo "${DIAMOND_SOO_ARTIFACT}" min "${DIAMOND_MIN_ARTIFACT}")
                add_test(NAME final_pipeline_smoke_test
                         COMMAND final_pipeline_smoke_test
                                 "${CMAKE_CURRENT_BINARY_DIR}/final-pipeline-smoke-scratch"
                                 "${DIAMOND_SOO_ARTIFACT}" "${DIAMOND_MIN_ARTIFACT}")
                set_tests_properties(final_pipeline_smoke_test PROPERTIES LABELS pipeline)
            else()
                add_test(NAME soo_artifact_contract
                         COMMAND soo_artifact_contract "${DIAMOND_SOO_ARTIFACT}")
                add_test(NAME model_parity_test
                         COMMAND model_parity_test soo "${DIAMOND_SOO_ARTIFACT}")
            endif()
            if(WIN32)
                list(GET TORCH_INCLUDE_DIRS 0 DIAMOND_TORCH_INCLUDE_DIR)
                get_filename_component(DIAMOND_TORCH_PREFIX
                    "${DIAMOND_TORCH_INCLUDE_DIR}/../.." ABSOLUTE)
                set_tests_properties(model_parity_test PROPERTIES
                    ENVIRONMENT "PATH=${DIAMOND_TORCH_PREFIX}/torch/lib;$ENV{PATH}")
                if(TEST final_pipeline_smoke_test)
                    set_tests_properties(final_pipeline_smoke_test PROPERTIES
                        ENVIRONMENT "PATH=${DIAMOND_TORCH_PREFIX}/torch/lib;$ENV{PATH}")
                endif()
            endif()
        endif()

        add_executable(training_vector_manifest_test tests/training_vector_manifest_test.cpp)
        target_link_libraries(training_vector_manifest_test PRIVATE alphadiamond::diamond_model)
        target_compile_features(training_vector_manifest_test PRIVATE cxx_std_20)
        add_test(NAME training_vector_manifest_test
                 COMMAND training_vector_manifest_test
                         "${PROJECT_SOURCE_DIR}/tests/golden/training-v1")

    add_executable(training_step_parity_test tests/training_step_parity_test.cpp)
        target_link_libraries(training_step_parity_test PRIVATE alphadiamond::diamond_training)
        target_include_directories(training_step_parity_test PRIVATE tests ${TORCH_INCLUDE_DIRS})
        target_compile_features(training_step_parity_test PRIVATE cxx_std_20)
        target_compile_definitions(training_step_parity_test PRIVATE _DISABLE_EXTENDED_ALIGNED_STORAGE)
        if(MSVC)
            target_compile_options(training_step_parity_test PRIVATE /bigobj)
        endif()

        add_executable(checkpoint_v2_roundtrip_test tests/checkpoint_v2_roundtrip_test.cpp)
        target_link_libraries(checkpoint_v2_roundtrip_test PRIVATE alphadiamond::diamond_training)
        target_include_directories(checkpoint_v2_roundtrip_test PRIVATE tests ${TORCH_INCLUDE_DIRS})
        target_compile_features(checkpoint_v2_roundtrip_test PRIVATE cxx_std_20)
        target_compile_definitions(checkpoint_v2_roundtrip_test PRIVATE _DISABLE_EXTENDED_ALIGNED_STORAGE)
        add_test(NAME checkpoint_v2_roundtrip_test
                 COMMAND checkpoint_v2_roundtrip_test "${CMAKE_CURRENT_BINARY_DIR}/checkpoint-v2-roundtrip-scratch")
        set_tests_properties(checkpoint_v2_roundtrip_test PROPERTIES LABELS checkpoint)
        if(DIAMOND_ENABLE_CUDA_TESTS AND CAFFE2_USE_CUDA)
            target_compile_definitions(checkpoint_v2_roundtrip_test PRIVATE
                DIAMOND_TEST_WITH_CUDA=1)
            add_test(NAME checkpoint_v2_roundtrip_test_cuda
                     COMMAND checkpoint_v2_roundtrip_test
                             "${CMAKE_CURRENT_BINARY_DIR}/checkpoint-v2-roundtrip-cuda-scratch"
                             --cuda)
            set_tests_properties(checkpoint_v2_roundtrip_test_cuda PROPERTIES
                LABELS "cuda;checkpoint")
        endif()

        add_executable(checkpoint_v2_reject_test tests/checkpoint_v2_reject_test.cpp)
        target_link_libraries(checkpoint_v2_reject_test PRIVATE alphadiamond::diamond_training)
        target_include_directories(checkpoint_v2_reject_test PRIVATE tests ${TORCH_INCLUDE_DIRS})
        target_compile_features(checkpoint_v2_reject_test PRIVATE cxx_std_20)
        target_compile_definitions(checkpoint_v2_reject_test PRIVATE _DISABLE_EXTENDED_ALIGNED_STORAGE)
        add_test(NAME checkpoint_v2_reject_test
                 COMMAND checkpoint_v2_reject_test "${CMAKE_CURRENT_BINARY_DIR}/checkpoint-v2-reject-scratch")
        set_tests_properties(checkpoint_v2_reject_test PROPERTIES LABELS checkpoint)

        add_executable(checkpoint_v2_transaction_test tests/checkpoint_v2_transaction_test.cpp)
        target_link_libraries(checkpoint_v2_transaction_test PRIVATE alphadiamond::diamond_training)
        target_include_directories(checkpoint_v2_transaction_test PRIVATE tests ${TORCH_INCLUDE_DIRS})
        target_compile_features(checkpoint_v2_transaction_test PRIVATE cxx_std_20)
        target_compile_definitions(checkpoint_v2_transaction_test PRIVATE _DISABLE_EXTENDED_ALIGNED_STORAGE)
        add_test(NAME checkpoint_v2_transaction_test
                 COMMAND checkpoint_v2_transaction_test "${CMAKE_CURRENT_BINARY_DIR}/checkpoint-v2-transaction-scratch")
        set_tests_properties(checkpoint_v2_transaction_test PROPERTIES LABELS checkpoint)

        add_executable(inference_coordinator_test tests/inference_coordinator_test.cpp)
        target_link_libraries(inference_coordinator_test PRIVATE alphadiamond::diamond_native_pipeline)
        target_include_directories(inference_coordinator_test PRIVATE tests ${TORCH_INCLUDE_DIRS})
        target_compile_features(inference_coordinator_test PRIVATE cxx_std_20)
        target_compile_definitions(inference_coordinator_test PRIVATE _DISABLE_EXTENDED_ALIGNED_STORAGE)
        add_test(NAME inference_coordinator_test
                 COMMAND inference_coordinator_test "${CMAKE_CURRENT_BINARY_DIR}/inference-coordinator-scratch")
        set_tests_properties(inference_coordinator_test PROPERTIES LABELS "cpu;pipeline")
        if(DIAMOND_ENABLE_CUDA_TESTS AND CAFFE2_USE_CUDA)
            target_compile_definitions(inference_coordinator_test PRIVATE DIAMOND_TEST_WITH_CUDA=1)
            add_test(NAME inference_coordinator_test_cuda
                     COMMAND inference_coordinator_test
                             "${CMAKE_CURRENT_BINARY_DIR}/inference-coordinator-cuda-scratch" --cuda)
            set_tests_properties(inference_coordinator_test_cuda PROPERTIES LABELS "cuda;pipeline")
        endif()

        add_executable(native_pipeline_smoke_test tests/native_pipeline_smoke_test.cpp)
        target_link_libraries(native_pipeline_smoke_test PRIVATE alphadiamond::diamond_native_pipeline)
        target_include_directories(native_pipeline_smoke_test PRIVATE tests ${TORCH_INCLUDE_DIRS})
        target_compile_features(native_pipeline_smoke_test PRIVATE cxx_std_20)
        target_compile_definitions(native_pipeline_smoke_test PRIVATE _DISABLE_EXTENDED_ALIGNED_STORAGE)
        add_test(NAME native_pipeline_smoke_test
                 COMMAND native_pipeline_smoke_test "${CMAKE_BINARY_DIR}/np-smoke")
        set_tests_properties(native_pipeline_smoke_test PROPERTIES LABELS pipeline)

        add_executable(run_state_test tests/run_state_test.cpp)
        target_link_libraries(run_state_test PRIVATE alphadiamond::diamond_orchestration)
        target_include_directories(run_state_test PRIVATE tests)
        target_compile_features(run_state_test PRIVATE cxx_std_20)
        add_test(NAME run_state_test
                 COMMAND run_state_test "${CMAKE_CURRENT_BINARY_DIR}/run-state-scratch")
        set_tests_properties(run_state_test PROPERTIES LABELS orchestration)

        add_executable(config_test tests/config_test.cpp)
        target_link_libraries(config_test PRIVATE alphadiamond::diamond_orchestration)
        target_compile_features(config_test PRIVATE cxx_std_20)
        add_test(NAME config_test
                 COMMAND config_test "${PROJECT_SOURCE_DIR}/configs/alphazero")
        set_tests_properties(config_test PROPERTIES LABELS orchestration)

        add_executable(coordinator_resume_test tests/coordinator_resume_test.cpp)
        target_link_libraries(coordinator_resume_test PRIVATE alphadiamond::diamond_orchestration)
        target_include_directories(coordinator_resume_test PRIVATE tests)
        target_compile_features(coordinator_resume_test PRIVATE cxx_std_20)
        add_test(NAME coordinator_resume_test
                 COMMAND coordinator_resume_test "${CMAKE_CURRENT_BINARY_DIR}/coordinator-resume-scratch")
        set_tests_properties(coordinator_resume_test PROPERTIES LABELS orchestration)

        add_executable(arena_schedule_test tests/arena_schedule_test.cpp)
        target_link_libraries(arena_schedule_test PRIVATE alphadiamond::diamond_orchestration)
        target_include_directories(arena_schedule_test PRIVATE tests)
        target_compile_features(arena_schedule_test PRIVATE cxx_std_20)
        add_test(NAME arena_schedule_test COMMAND arena_schedule_test)
        set_tests_properties(arena_schedule_test PROPERTIES LABELS orchestration)

        add_executable(rating_fixture_test tests/rating_fixture_test.cpp)
        target_link_libraries(rating_fixture_test PRIVATE alphadiamond::diamond_orchestration)
        target_include_directories(rating_fixture_test PRIVATE tests)
        target_compile_features(rating_fixture_test PRIVATE cxx_std_20)
        add_test(NAME rating_fixture_test COMMAND rating_fixture_test)
        set_tests_properties(rating_fixture_test PROPERTIES LABELS orchestration)

        add_executable(rating_registry_test tests/rating_registry_test.cpp)
        target_link_libraries(rating_registry_test PRIVATE alphadiamond::diamond_orchestration)
        target_include_directories(rating_registry_test PRIVATE tests)
        target_compile_features(rating_registry_test PRIVATE cxx_std_20)
        add_test(NAME rating_registry_test COMMAND rating_registry_test)
        set_tests_properties(rating_registry_test PROPERTIES LABELS orchestration)

        add_executable(rating_store_transaction_test tests/rating_store_transaction_test.cpp)
        target_link_libraries(rating_store_transaction_test PRIVATE alphadiamond::diamond_orchestration)
        target_include_directories(rating_store_transaction_test PRIVATE tests)
        target_compile_features(rating_store_transaction_test PRIVATE cxx_std_20)
        add_test(NAME rating_store_transaction_test COMMAND rating_store_transaction_test)
        set_tests_properties(rating_store_transaction_test PROPERTIES LABELS orchestration)

        add_executable(rating_sync_test tests/rating_sync_test.cpp)
        target_link_libraries(rating_sync_test PRIVATE alphadiamond::diamond_orchestration)
        target_include_directories(rating_sync_test PRIVATE tests)
        target_compile_features(rating_sync_test PRIVATE cxx_std_20)
        add_dependencies(rating_sync_test alphadiamond-rating-sync)
        add_test(NAME rating_sync_test
                 COMMAND rating_sync_test "$<TARGET_FILE:alphadiamond-rating-sync>")
        set_tests_properties(rating_sync_test PROPERTIES LABELS orchestration)

        add_executable(cli_contract_test tests/cli_contract_test.cpp)
        target_link_libraries(cli_contract_test PRIVATE alphadiamond::diamond_orchestration)
        add_dependencies(cli_contract_test alphadiamond-train)
        target_include_directories(cli_contract_test PRIVATE tests)
        target_compile_features(cli_contract_test PRIVATE cxx_std_20)
        add_test(NAME cli_contract_test
                 COMMAND cli_contract_test "${CMAKE_CURRENT_BINARY_DIR}/cli-contract-scratch"
                         "$<TARGET_FILE:alphadiamond-train>")
        set_tests_properties(cli_contract_test PROPERTIES LABELS orchestration)

        add_executable(device_test tests/device_test.cpp)
        target_link_libraries(device_test PRIVATE alphadiamond::diamond_training)
        add_dependencies(device_test alphadiamond-train)
        target_include_directories(device_test PRIVATE tests ${TORCH_INCLUDE_DIRS})
        target_compile_features(device_test PRIVATE cxx_std_20)
        target_compile_definitions(device_test PRIVATE _DISABLE_EXTENDED_ALIGNED_STORAGE)
        add_test(NAME device_test
                 COMMAND device_test "${CMAKE_CURRENT_BINARY_DIR}/device-test-scratch"
                         "$<TARGET_FILE:alphadiamond-train>")
        set_tests_properties(device_test PROPERTIES LABELS "cpu;device")
        # CUDA acceptance is opt-in. CPU presets do not register this test;
        # the test is enabled only on a CUDA-enabled LibTorch host.
        if(DIAMOND_ENABLE_CUDA_TESTS AND CAFFE2_USE_CUDA)
            add_test(NAME device_test_cuda
                     COMMAND device_test "${CMAKE_CURRENT_BINARY_DIR}/device-test-cuda-scratch"
                                 "$<TARGET_FILE:alphadiamond-train>" --cuda)
            set_tests_properties(device_test_cuda PROPERTIES LABELS "cuda;device")
        endif()
        add_test(NAME training_step_parity_test
                 COMMAND training_step_parity_test
                         "${PROJECT_SOURCE_DIR}/tests/golden/training-v1")
        set_tests_properties(training_step_parity_test PROPERTIES LABELS training)
        if(DIAMOND_ENABLE_CUDA_TESTS AND CAFFE2_USE_CUDA)
            add_test(NAME training_step_parity_test_cuda
                     COMMAND training_step_parity_test
                             "${PROJECT_SOURCE_DIR}/tests/golden/training-v1" --cuda)
            set_tests_properties(training_step_parity_test_cuda PROPERTIES
                LABELS "cuda;training")
        endif()
        if(WIN32)
            list(GET TORCH_INCLUDE_DIRS 0 DIAMOND_TORCH_INCLUDE_DIR)
            get_filename_component(DIAMOND_TORCH_PREFIX
                "${DIAMOND_TORCH_INCLUDE_DIR}/../.." ABSOLUTE)
            get_filename_component(DIAMOND_RUNTIME_ROOT
                "${DIAMOND_TORCH_PREFIX}/../.." ABSOLUTE)
            set(DIAMOND_TEST_PATH
                "${DIAMOND_RUNTIME_ROOT}/Library/bin;${DIAMOND_RUNTIME_ROOT}/Scripts;${DIAMOND_RUNTIME_ROOT};${DIAMOND_TORCH_PREFIX}/torch/lib;$ENV{PATH}")
            string(REPLACE ";" "\\;" DIAMOND_TEST_PATH "${DIAMOND_TEST_PATH}")
            set_tests_properties(checkpoint_v2_roundtrip_test checkpoint_v2_reject_test
                                 checkpoint_v2_transaction_test inference_coordinator_test
                                 native_pipeline_smoke_test promotion_test device_test
                                 training_step_parity_test PROPERTIES
                ENVIRONMENT "PATH=${DIAMOND_TEST_PATH}")
            if(TEST device_test_cuda)
                set_tests_properties(device_test_cuda PROPERTIES ENVIRONMENT "PATH=${DIAMOND_TEST_PATH}")
            endif()
            if(TEST inference_coordinator_test_cuda)
                set_tests_properties(inference_coordinator_test_cuda PROPERTIES
                    ENVIRONMENT "PATH=${DIAMOND_TEST_PATH}")
            endif()
            if(TEST checkpoint_v2_roundtrip_test_cuda)
                set_tests_properties(checkpoint_v2_roundtrip_test_cuda PROPERTIES
                    ENVIRONMENT "PATH=${DIAMOND_TEST_PATH}")
            endif()
            if(TEST training_step_parity_test_cuda)
                set_tests_properties(training_step_parity_test_cuda PROPERTIES
                    ENVIRONMENT "PATH=${DIAMOND_TEST_PATH}")
            endif()
        endif()
    endif()
endif()

install(TARGETS soo_core soo_search
    EXPORT alphadiamondTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT alphadiamondTargets
    FILE alphadiamondTargets.cmake
    NAMESPACE alphadiamond::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/alphadiamond
)
