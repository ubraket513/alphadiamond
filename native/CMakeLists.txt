set(SOO_CORE_SOURCES
    src/board.cpp
    src/encoder.cpp
    src/prior.cpp
    src/rules.cpp
)

set(SOO_SEARCH_SOURCES
    src/batcher.cpp
    src/evaluator.cpp
    src/mcts.cpp
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
    add_executable(soo_native_smoke tests/smoke.cpp)
    target_link_libraries(soo_native_smoke PRIVATE alphadiamond::soo_search)
    target_compile_features(soo_native_smoke PRIVATE cxx_std_20)
    add_test(NAME soo_native_smoke COMMAND soo_native_smoke)
endif()

if(DIAMOND_BUILD_LIBTORCH_PROBE)
    find_package(Torch REQUIRED)
    add_library(diamond_model STATIC
        src/deployment_artifact.cpp
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

    add_executable(soo_native_model_probe src/soo_native_model_probe.cpp)
    target_link_libraries(soo_native_model_probe PRIVATE alphadiamond::diamond_model)
    target_compile_features(soo_native_model_probe PRIVATE cxx_std_20)
    target_compile_definitions(soo_native_model_probe PRIVATE _DISABLE_EXTENDED_ALIGNED_STORAGE)
    if(MSVC)
        target_compile_options(soo_native_model_probe PRIVATE /bigobj)
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
        add_executable(soo_artifact_contract tests/artifact_contract.cpp)
        target_link_libraries(soo_artifact_contract PRIVATE alphadiamond::diamond_model)
        target_compile_features(soo_artifact_contract PRIVATE cxx_std_20)
        add_test(NAME soo_artifact_contract
                 COMMAND soo_artifact_contract "${PROJECT_SOURCE_DIR}/artifacts/soo-spike")
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
