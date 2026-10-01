# laya

# Self-contained laya model (encoder + typed decision head) that loads its own
# GGUF and builds the inference graph with ggml directly, following the
# tools/mtmd pattern. The library links only ggml. Input building for the CLI
# and the decision server (laya-decide) lives in tools/decision.

add_library(laya STATIC
            laya.cpp
            laya.h
            laya-unicode.cpp
            laya-unicode.h)

target_link_libraries     (laya PUBLIC ggml)
target_include_directories(laya PUBLIC .)
target_compile_features   (laya PRIVATE cxx_std_17)

if (BUILD_SHARED_LIBS)
    # linked into llama-server-impl, which is SHARED in DL builds
    set_target_properties(laya PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# HF checkpoint directory -> GGUF (port of convert_hf_to_gguf.py for laya). Separate from laya so
# that the model library does not grow; links ggml (gguf API, ggml_quantize_chunk, ggml_fopen).
add_library(laya-convert STATIC
            laya-convert.cpp
            laya-convert.h)

target_link_libraries     (laya-convert PUBLIC ggml)
if (WIN32)
    # laya_utf8_to_wide (UTF-8 file names through the UTF-16 APIs), declared only on Windows
    target_link_libraries (laya-convert PRIVATE laya)
endif()
target_include_directories(laya-convert PUBLIC .)
target_include_directories(laya-convert PRIVATE ${PROJECT_SOURCE_DIR}/vendor)
target_compile_features   (laya-convert PRIVATE cxx_std_17)
set_target_properties     (laya-convert PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Gate the CLI on LLAMA_BUILD_TOOLS so that library-only builds skip it.
if (LLAMA_BUILD_TOOLS)
    set(TARGET llama-laya-cli)
    add_executable(${TARGET} laya-cli.cpp)
    set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME llama-laya-cli)
    if(LLAMA_TOOLS_INSTALL)
        install(TARGETS ${TARGET} RUNTIME)
    endif()
    target_link_libraries  (${TARGET} PRIVATE decision laya Threads::Threads)
    target_compile_features(${TARGET} PRIVATE cxx_std_17)

    set(TARGET llama-laya-convert)
    add_executable(${TARGET} laya-convert-cli.cpp)
    set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME llama-laya-convert)
    if(LLAMA_TOOLS_INSTALL)
        install(TARGETS ${TARGET} RUNTIME)
    endif()
    target_link_libraries  (${TARGET} PRIVATE decision laya-convert) # decision: decision_utf8_args (Windows argv)
    target_compile_features(${TARGET} PRIVATE cxx_std_17)
endif()
