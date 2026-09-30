# laya

# Self-contained laya model (encoder + typed decision head) that loads its own
# GGUF and builds the inference graph with ggml directly, following the
# tools/mtmd pattern. The library links only ggml. Input building for the CLI
# and the decision server (laya-decide) lives in tools/decision.

add_library(laya STATIC
            laya.cpp
            laya.h)

target_link_libraries     (laya PUBLIC ggml)
target_include_directories(laya PUBLIC .)
target_compile_features   (laya PRIVATE cxx_std_17)

if (BUILD_SHARED_LIBS)
    # linked into llama-server-impl, which is SHARED in DL builds
    set_target_properties(laya PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

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
endif()
