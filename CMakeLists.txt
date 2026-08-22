cmake_minimum_required(VERSION 3.21)

project(alphadiamond LANGUAGES CXX)

option(DIAMOND_BUILD_PYBIND "Build the optional pybind11 extension" OFF)
option(DIAMOND_BUILD_NATIVE_TESTS "Build portable native smoke tests" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)
include(CTest)
find_package(Threads REQUIRED)

add_subdirectory(native)
