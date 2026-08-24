if(NOT DEFINED SOURCE_DIR)
    message(FATAL_ERROR "SOURCE_DIR is required")
endif()

set(pattern "(^|[;&|[:space:]])(python([0-9.]+)?|pip[0-9.]*|pytest|ruff)([[:space:]]|$)|pybind11_add_module|find_package\\([[:space:]]*pybind11")
execute_process(
    COMMAND git -C "${SOURCE_DIR}" grep -n -E "${pattern}" --
        .github/workflows CMakeLists.txt CMakePresets.json Makefile "tools/*.sh"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE matches
    ERROR_VARIABLE error
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(result EQUAL 0)
    message(FATAL_ERROR "unsupported Python/pybind command remains:\n${matches}")
elseif(NOT result EQUAL 1)
    message(FATAL_ERROR "git grep failed: ${error}")
endif()
