execute_process(
    COMMAND "${MIN_VACANCY_REALIZABILITY}" --help
    RESULT_VARIABLE help_result
    OUTPUT_VARIABLE help_out
    ERROR_VARIABLE help_error)
if(NOT help_result EQUAL 0)
    message(FATAL_ERROR "--help failed: ${help_error}")
endif()
foreach(flag IN ITEMS --checkpoint --config --replay --arm --device --steps --batch-size
                      --eval-samples --eval-batch --seed --expected-source-commit
                      --expected-config-sha256 --expected-model-sha256 --out --checkpoint-out)
    string(FIND "${help_out}" "${flag}" position)
    if(position EQUAL -1)
        message(FATAL_ERROR "--help omitted ${flag}")
    endif()
endforeach()

execute_process(
    COMMAND "${MIN_VACANCY_REALIZABILITY}"
            --checkpoint missing --config missing --replay missing --arm invalid --device cpu
            --steps 1 --batch-size 1 --eval-samples 1 --eval-batch 1 --seed 1
            --expected-source-commit 0000000000000000000000000000000000000000
            --expected-config-sha256 0000000000000000000000000000000000000000000000000000000000000000
            --expected-model-sha256 0000000000000000000000000000000000000000000000000000000000000000
            --out report.json
    RESULT_VARIABLE invalid_result
    ERROR_VARIABLE invalid_error)
if(invalid_result EQUAL 0 OR NOT invalid_error MATCHES "arm must be head or full")
    message(FATAL_ERROR "invalid arm was not rejected before I/O: ${invalid_error}")
endif()

execute_process(
    COMMAND "${MIN_VACANCY_REALIZABILITY}"
            --checkpoint missing --config missing --replay missing --arm head --device cpu
            --steps 1 --batch-size 1 --eval-samples 1 --eval-batch 1 --seed 1
            --expected-source-commit 0000000000000000000000000000000000000000
            --expected-config-sha256 0000000000000000000000000000000000000000000000000000000000000000
            --expected-model-sha256 0000000000000000000000000000000000000000000000000000000000000000
            --out report.json
    RESULT_VARIABLE provenance_result
    ERROR_VARIABLE provenance_error)
if(provenance_result EQUAL 0 OR NOT provenance_error MATCHES "source commit mismatch")
    message(FATAL_ERROR "source mismatch was not gated before I/O: ${provenance_error}")
endif()
