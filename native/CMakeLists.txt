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
