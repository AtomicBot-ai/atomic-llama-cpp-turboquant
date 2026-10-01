llama_add_compile_flags()

function(llama_build source)
    set(TEST_SOURCES ${source} ${ARGN})

    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_link_libraries(${TEST_TARGET} PRIVATE llama llama-common)
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
endfunction()

function(llama_test target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    set(TEST_TARGET ${target})

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

function(llama_test_cmd target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND ${target}
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# Builds and runs a test source file.
# Optional args:
# - NAME: name of the executable & test target (defaults to the source file name without extension)
# - LABEL: label for the test (defaults to main)
# - ARGS: arguments to pass to the test executable
# - WORKING_DIRECTORY
function(llama_build_and_test source)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_SOURCES ${source} ${LLAMA_TEST_UNPARSED_ARGUMENTS})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
    target_link_libraries(${TEST_TARGET} PRIVATE llama-common)

    add_test(
        NAME ${TEST_TARGET}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_TARGET} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# build test-tokenizer-0 target once and add many tests
llama_build(test-tokenizer-0.cpp)

llama_test(test-tokenizer-0 NAME test-tokenizer-0-bert-bge          ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-bert-bge.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-command-r         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-command-r.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-coder    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-coder.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-llm      ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-llm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-falcon            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gemma-4           ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gemma-4.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gpt-2             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-bpe         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-spm         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-mpt               ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-phi-3             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-phi-3.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen2             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen35            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen35.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-refact            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-starcoder         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

if (NOT WIN32)
    llama_test_cmd(
        ${CMAKE_CURRENT_SOURCE_DIR}/test-tokenizers-repo.sh
        NAME test-tokenizers-ggml-vocabs
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        ARGS https://huggingface.co/ggml-org/vocabs ${PROJECT_SOURCE_DIR}/models/ggml-vocabs
    )
endif()

if (LLAMA_LLGUIDANCE)
    llama_build_and_test(test-grammar-llguidance.cpp ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
endif ()

llama_build(test-recurrent-state-rollback.cpp)

if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    # these tests are disabled on Windows because they use internal functions not exported with LLAMA_API (when building with shared libraries)
    llama_build_and_test(test-sampling.cpp)
    llama_build_and_test(test-reasoning-budget.cpp)
    llama_build_and_test(test-grammar-parser.cpp)
    llama_build_and_test(test-grammar-integration.cpp)
    llama_build_and_test(test-llama-grammar.cpp)
    llama_build_and_test(test-batch-alloc.cpp)
    llama_build_and_test(test-chat.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    target_include_directories(test-chat PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-chat PRIVATE server-context)
    # TODO: disabled on loongarch64 because the ggml-ci node lacks Python 3.8
    if (NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "loongarch64")
        llama_build_and_test(test-json-schema-to-grammar.cpp   WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
        target_include_directories(test-json-schema-to-grammar PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    endif()

    if (NOT GGML_BACKEND_DL)
        llama_build(test-quantize-stats.cpp)
    endif()

    llama_build(test-gbnf-validator.cpp)

    # build test-tokenizer-1-bpe target once and add many tests
    llama_build(test-tokenizer-1-bpe.cpp)

    # TODO: disabled due to slowness
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-aquila    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-aquila.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-falcon    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-2     ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-neox  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-neox.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-llama-bpe ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf --ignore-merges)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-mpt       ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-refact    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-starcoder ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

    # build test-tokenizer-1-spm target once and add many tests
    llama_build(test-tokenizer-1-spm.cpp)

    llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-llama-spm ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
    #llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-baichuan  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-baichuan.gguf)

    # llama_build_and_test(test-double-float.cpp) # SLOW

    llama_build_and_test(test-llama-archs.cpp)

    set(MODEL_DIR "${CMAKE_CURRENT_BINARY_DIR}/test-models/")
    file(MAKE_DIRECTORY "${MODEL_DIR}")

    llama_test(
        test-llama-archs
        NAME test-generate-models
        LABEL main
        ARGS -o "${MODEL_DIR}"
    )
    set_tests_properties(test-generate-models PROPERTIES
        FIXTURES_SETUP generate-models
    )

    llama_test(
        test-recurrent-state-rollback
        LABEL main
        ARGS -m "${MODEL_DIR}/qwen35-dense.gguf"
    )
    set_tests_properties(test-recurrent-state-rollback PROPERTIES
        FIXTURES_REQUIRED generate-models
    )
endif()

llama_build_and_test(test-chat-peg-parser.cpp peg-parser/simple-tokenize.cpp)
llama_build_and_test(test-jinja.cpp)
llama_test(test-jinja NAME test-jinja-py ARGS -py LABEL python)
llama_build_and_test(test-chat-auto-parser.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
llama_build_and_test(test-chat-template.cpp)
llama_build_and_test(test-log.cpp)

if (LLAMA_BUILD_TOOLS AND NOT EMSCRIPTEN)
    # decision API library (tools/decision); kept outside the WIN32 shared block on purpose
    llama_build_and_test(test-decision-json.cpp ARGS ${PROJECT_SOURCE_DIR}/tests/decision/golden)
    llama_build_and_test(test-decision-request.cpp)
    llama_build_and_test(test-decision-calib.cpp ARGS ${PROJECT_SOURCE_DIR}/tests/decision)
    llama_build_and_test(test-decision-router.cpp)
    llama_build_and_test(test-decision-reference.cpp ARGS ${PROJECT_SOURCE_DIR}/tests/decision/golden/router_cases.jsonl)
    foreach (test_target test-decision-json test-decision-request test-decision-calib test-decision-router test-decision-reference)
        target_link_libraries(${test_target} PRIVATE decision)
    endforeach()

    # llama-laya-cli -m <checkpoint dir>: GGUF cache (converted, then cache hit), Python bytes, same output as -m GGUF
    add_test(NAME test-decision-checkpoint-cli
             COMMAND ${CMAKE_COMMAND} -DCLI=$<TARGET_FILE:llama-laya-cli> -DFIXTURES=${PROJECT_SOURCE_DIR}/tests/laya/convert
                     "-DINPUTS=${PROJECT_SOURCE_DIR}/tests/laya/demo_input.json|${PROJECT_SOURCE_DIR}/tests/laya/golden/score_en.json"
                     -DWORK=${CMAKE_CURRENT_BINARY_DIR}/decision-checkpoint-cli
                     -P ${PROJECT_SOURCE_DIR}/tests/decision/checkpoint-cli.cmake)
    set_tests_properties(test-decision-checkpoint-cli PROPERTIES LABELS main)

    # laya HF checkpoint -> GGUF converter (tools/laya/laya-convert): tiny fixtures vs Python golden sha256
    llama_build_and_test(test-laya-convert.cpp ${PROJECT_SOURCE_DIR}/examples/gguf-hash/deps/sha256/sha256.c
                         ARGS ${PROJECT_SOURCE_DIR}/tests/laya/convert ${CMAKE_CURRENT_BINARY_DIR}/laya-convert-scratch)
    target_link_libraries     (test-laya-convert PRIVATE laya-convert)
    target_include_directories(test-laya-convert PRIVATE ${PROJECT_SOURCE_DIR}/vendor ${PROJECT_SOURCE_DIR}/examples/gguf-hash/deps)

    # the same cases against a fresh Python conversion (exit 77 = skip without torch / transformers; LAYA_REF_PYTHON picks the interpreter)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if (Python3_Interpreter_FOUND)
        add_test(NAME test-laya-convert-py
                 COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/laya/make_tiny_hf_laya.py --check $<TARGET_FILE:llama-laya-convert>)
        set_tests_properties(test-laya-convert-py PROPERTIES LABELS python SKIP_RETURN_CODE 77)
    endif()

    # laya GPU backends (Phase 4b): exit code 77 = no GPU / iGPU device in the build (skipped)
    llama_build_and_test(test-laya-backend-precision.cpp LABEL laya)
    target_link_libraries(test-laya-backend-precision PRIVATE laya)
    set_tests_properties(test-laya-backend-precision PROPERTIES SKIP_RETURN_CODE 77)

    # the replay test needs the tiny random laya GGUF, which a Python fixture generates (numpy, pyyaml:
    # tools/server/tests/requirements.txt); -DPython3_EXECUTABLE=<venv>/bin/python picks the interpreter
    llama_build(test-laya-backend-parity.cpp)
    target_link_libraries(test-laya-backend-parity PRIVATE laya)
    set(LAYA_TINY_PY_OK OFF)
    if (Python3_Interpreter_FOUND)
        # the fixture imports numpy and gguf-py (which needs pyyaml): without them the test is not registered
        execute_process(COMMAND ${Python3_EXECUTABLE} -c "import numpy, yaml"
                        RESULT_VARIABLE LAYA_TINY_PY_RC OUTPUT_QUIET ERROR_QUIET)
        if (LAYA_TINY_PY_RC EQUAL 0)
            set(LAYA_TINY_PY_OK ON)
        else()
            message(STATUS "test-laya-backend-parity: ${Python3_EXECUTABLE} has no numpy / pyyaml, not registered")
        endif()
    endif()
    if (LAYA_TINY_PY_OK)
        set(LAYA_TINY_GGUF "${CMAKE_CURRENT_BINARY_DIR}/laya-tiny-backend.gguf")
        add_test(NAME test-laya-tiny-gguf
                 COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/decision/make_tiny_laya.py ${LAYA_TINY_GGUF})
        set_tests_properties(test-laya-tiny-gguf PROPERTIES FIXTURES_SETUP laya-tiny LABELS laya)
        llama_test(test-laya-backend-parity LABEL laya
                   ARGS ${LAYA_TINY_GGUF} ${PROJECT_SOURCE_DIR}/tests/laya/parity/tiers.json)
        set_tests_properties(test-laya-backend-parity PROPERTIES SKIP_RETURN_CODE 77 FIXTURES_REQUIRED laya-tiny)
    endif()
endif()

llama_build_and_test(
    test-peg-parser.cpp
    peg-parser/simple-tokenize.cpp
    peg-parser/test-basic.cpp
    peg-parser/test-gbnf-generation.cpp
    peg-parser/test-json-parser.cpp
    peg-parser/test-json-serialization.cpp
    peg-parser/test-python-dict-parser.cpp
    peg-parser/test-unicode.cpp
    peg-parser/tests.h
)

if (NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "s390x")
    set(MODEL_NAME "tinyllamas/stories15M-q4_0.gguf")
    set(MODEL_HASH "SHA256=66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739")
else()
    set(MODEL_NAME "tinyllamas/stories15M-be.Q4_0.gguf")
    set(MODEL_HASH "SHA256=9aec857937849d976f30397e97eb1cabb53eb9dcb1ce4611ba8247fb5f44c65d")
endif()
set(MODEL_DEST "${CMAKE_BINARY_DIR}/${MODEL_NAME}")

add_test(NAME test-download-model COMMAND ${CMAKE_COMMAND}
    -DDEST=${MODEL_DEST}
    -DNAME=${MODEL_NAME}
    -DHASH=${MODEL_HASH}
    -P ${CMAKE_SOURCE_DIR}/cmake/download-models.cmake
)
set_tests_properties(test-download-model PROPERTIES FIXTURES_SETUP test-download-model)

llama_build_and_test(test-thread-safety.cpp ARGS -m "${MODEL_DEST}" -ngl 99 -p "The meaning of life is" -n 128 -c 256 -ub 32 -np 4 -t 2)
set_tests_properties(test-thread-safety PROPERTIES FIXTURES_REQUIRED test-download-model)

llama_build_and_test(test-arg-parser.cpp)
llama_build_and_test(test-model-resolution.cpp)
# the test serves its repos from an httplib server, and the library links it privately
target_link_libraries(test-model-resolution PRIVATE cpp-httplib)

if (NOT LLAMA_SANITIZE_ADDRESS AND NOT GGML_SCHED_NO_REALLOC)
  # TODO: repair known memory leaks
  llama_build_and_test(test-opt.cpp)
endif()
llama_build_and_test(test-backend-ops.cpp)

add_executable(test-flash-attn-bias test-flash-attn-bias.cpp)
target_link_libraries(test-flash-attn-bias PRIVATE ggml)
add_test(NAME test-flash-attn-bias COMMAND test-flash-attn-bias)

add_executable(test-flash-attn-generic-hash test-flash-attn-generic-hash.cpp)
target_link_libraries(test-flash-attn-generic-hash PRIVATE ggml)
add_test(NAME test-flash-attn-generic-hash COMMAND test-flash-attn-generic-hash)

llama_build_and_test(test-model-load-cancel.cpp LABEL "model")
llama_build_and_test(test-autorelease.cpp       LABEL "model")
llama_build_and_test(test-backend-sampler.cpp   LABEL "model")

# Test for state restore with fragmented KV cache
# Requires a model, uses same args pattern as test-thread-safety
llama_build_and_test(test-state-restore-fragmented.cpp LABEL "model" ARGS -m "${MODEL_DEST}")
set_tests_properties(test-state-restore-fragmented PROPERTIES FIXTURES_REQUIRED test-download-model)

# Test state save/load functionality
llama_build_and_test(test-save-load-state.cpp LABEL "model" ARGS -m "${MODEL_DEST}")
set_tests_properties(test-save-load-state PROPERTIES FIXTURES_REQUIRED test-download-model)

if (APPLE)
    llama_build(test-rset-release.cpp)
endif()

if (NOT GGML_BACKEND_DL)
    # these tests use the backends directly and cannot be built with dynamic loading
    llama_build_and_test(test-barrier.cpp)
    llama_build_and_test(test-quantize-fns.cpp)
    llama_build_and_test(test-quantize-perf.cpp)
    llama_build_and_test(test-rope.cpp)
    llama_build_and_test(test-col2im-1d.cpp)
endif()

# libmtmd
set(LLAMA_TEST_NAME test-mtmd-c-api)
llama_build_and_test(test-mtmd-c-api.c)
target_link_libraries(${LLAMA_TEST_NAME} PRIVATE mtmd)
unset(LLAMA_TEST_NAME)

# GGUF model data fetcher library for tests that need real model metadata
# Only compile when cpp-httplib has SSL support (CPPHTTPLIB_OPENSSL_SUPPORT)
if (TARGET cpp-httplib)
    get_target_property(_cpp_httplib_defs cpp-httplib INTERFACE_COMPILE_DEFINITIONS)
    if (_cpp_httplib_defs MATCHES "CPPHTTPLIB_OPENSSL_SUPPORT")
        add_library(gguf-model-data STATIC gguf-model-data.cpp)
        target_link_libraries(gguf-model-data PRIVATE llama-common cpp-httplib)
        target_include_directories(gguf-model-data PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

        add_executable(test-gguf-model-data test-gguf-model-data.cpp)
        target_link_libraries(test-gguf-model-data PRIVATE gguf-model-data llama-common)
        llama_test(test-gguf-model-data LABEL "model")

        # test-quant-type-selection requires gguf-model-data for remote model metadata
        llama_build_and_test(test-quant-type-selection.cpp LABEL "model")
        target_link_libraries(test-quant-type-selection PRIVATE gguf-model-data)
        target_compile_definitions(test-quant-type-selection PRIVATE
            SNAPSHOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/snapshots")
    endif()
endif()

# dummy executable - not installed
get_filename_component(TEST_TARGET test-c.c NAME_WE)
add_executable(${TEST_TARGET} test-c.c)
target_link_libraries(${TEST_TARGET} PRIVATE llama)

if (NOT LLAMA_USE_SYSTEM_GGML)
    # Needs non-public ggml-impl.h
    llama_build_and_test(test-gguf.cpp)

    # Needs non-public ggml{,-backend}-impl.h
    llama_build_and_test(test-alloc.cpp)
endif()

llama_build(test-export-graph-ops.cpp)
if (TARGET gguf-model-data)
    target_link_libraries(test-export-graph-ops PRIVATE gguf-model-data)
    target_compile_definitions(test-export-graph-ops PRIVATE LLAMA_HF_FETCH)
endif()
