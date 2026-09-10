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
# The only Python allowed in the tree: offline analysis summarizers cited by
# the reports in docs/. Nothing in the engine, the trainer or the test suite.
set(allowed_python
    "tools/summarize_min_serial_search.py"
    "tools/summarize_min_local_audit.py")
string(REPLACE "
" ";" tracked_python "${tracked_python}")
list(REMOVE_ITEM tracked_python ${allowed_python})
string(REPLACE ";" "
" tracked_python "${tracked_python}")
if(NOT tracked_python STREQUAL "")
    message(FATAL_ERROR "tracked Python sources remain:\n${tracked_python}")
endif()
