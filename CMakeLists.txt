cmake_minimum_required(VERSION 3.21)

project(alphadiamond LANGUAGES CXX)

option(DIAMOND_BUILD_PYBIND "Build the optional pybind11 extension" OFF)
option(DIAMOND_BUILD_NATIVE_TESTS "Build portable native smoke tests" ON)
option(DIAMOND_BUILD_LIBTORCH_PROBE "Build the Q2 LibTorch parity probe" OFF)
option(DIAMOND_BUILD_QT "Build the native Qt/QML shell" OFF)
option(DIAMOND_BUILD_QT_SOO "Build Qt shell with native LibTorch Soo search" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)
include(CTest)
find_package(Threads REQUIRED)

add_subdirectory(native)

if(DIAMOND_BUILD_QT OR DIAMOND_BUILD_QT_SOO)
    add_subdirectory(native/qt)
endif()

# Packaging. Models ship *beside* the executable rather than compiled into it,
# so a release can carry a different default model, and a model can be replaced,
# without relinking. Build the directory first:
#
#   python tools/export_deployment.py artifacts/soo-spike --family soo
#   python tools/build_model_index.py dist/models --artifact soo=artifacts/soo-spike
#
# then `cmake --build --preset native-release --target package`.
set(DIAMOND_MODELS_DIR "${CMAKE_SOURCE_DIR}/dist/models"
    CACHE PATH "models/ directory to bundle into the package")
if(EXISTS "${DIAMOND_MODELS_DIR}/index.json")
    install(DIRECTORY "${DIAMOND_MODELS_DIR}/" DESTINATION models)
endif()
install(FILES "${CMAKE_SOURCE_DIR}/README.md" DESTINATION .)

set(CPACK_PACKAGE_NAME "AlphaDiamond")
set(CPACK_PACKAGE_VENDOR "alphadiamond")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Diamond engine, Qt console and AlphaZero stack")
set(CPACK_PACKAGE_VERSION "0.1.0")
set(CPACK_GENERATOR "ZIP")
set(CPACK_PACKAGE_CHECKSUM SHA256)
set(CPACK_VERBATIM_VARIABLES ON)
include(CPack)
