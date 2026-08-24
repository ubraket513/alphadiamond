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

if(DIAMOND_BUILD_PYBIND)
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(diamond_native MODULE ../native/bindings.cpp)
    target_link_libraries(diamond_native PRIVATE alphadiamond::soo_search)
    target_compile_features(diamond_native PRIVATE cxx_std_20)
endif()

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
        # Artifact fixtures are exported, not committed:
        #   python tools/export_deployment.py artifacts/soo-spike --family soo
        #   python tools/export_deployment.py artifacts/min-spike --family min
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
            endif()
        endif()

        add_executable(training_vector_manifest_test tests/training_vector_manifest_test.cpp)
        target_link_libraries(training_vector_manifest_test PRIVATE alphadiamond::diamond_model)
        target_compile_features(training_vector_manifest_test PRIVATE cxx_std_20)
        add_test(NAME training_vector_manifest_test
                 COMMAND training_vector_manifest_test
                         "${PROJECT_SOURCE_DIR}/tests/golden/training-v1")
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
