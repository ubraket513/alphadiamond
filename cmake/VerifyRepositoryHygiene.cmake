if(NOT DEFINED SOURCE_DIR)
    message(FATAL_ERROR "SOURCE_DIR is required")
endif()

execute_process(
    COMMAND git -C "${SOURCE_DIR}" ls-files
    RESULT_VARIABLE result
    OUTPUT_VARIABLE tracked
    ERROR_VARIABLE error
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "git ls-files failed: ${error}")
endif()

string(REPLACE "\r\n" "\n" tracked "${tracked}")
string(REPLACE "\n" ";" tracked_files "${tracked}")
set(allowed_weight "runtime/runs/soo/cpu8h-soo-20260819/latest.pt")
foreach(path IN LISTS tracked_files)
    if(path MATCHES "(^|/)(build[^/]*|dist|__pycache__)(/|$)" OR
       path MATCHES "\\.py[co]$")
        message(FATAL_ERROR "tracked build/runtime output is forbidden: ${path}")
    endif()
    if(path MATCHES "\\.(pt|pth)$" AND NOT path STREQUAL allowed_weight)
        message(FATAL_ERROR "unapproved tracked weight: ${path}")
    endif()
    if(EXISTS "${SOURCE_DIR}/${path}" AND NOT IS_DIRECTORY "${SOURCE_DIR}/${path}")
        file(SIZE "${SOURCE_DIR}/${path}" size)
        if(size GREATER 26214400 AND NOT path STREQUAL allowed_weight)
            message(FATAL_ERROR "tracked file exceeds 25 MiB policy: ${path}")
        endif()
    endif()
endforeach()

if(NOT EXISTS "${SOURCE_DIR}/tests/golden/MANIFEST.json")
    message(FATAL_ERROR "tests/golden/MANIFEST.json is required")
endif()
