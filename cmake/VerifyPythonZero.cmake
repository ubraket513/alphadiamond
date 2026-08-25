if(NOT DEFINED SOURCE_DIR)
    message(FATAL_ERROR "SOURCE_DIR is required")
endif()

execute_process(
    COMMAND git -C "${SOURCE_DIR}" ls-files "*.py"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE tracked_python
    ERROR_VARIABLE error
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "git ls-files failed: ${error}")
endif()
if(NOT tracked_python STREQUAL "")
    message(FATAL_ERROR "tracked Python sources remain:\n${tracked_python}")
endif()
