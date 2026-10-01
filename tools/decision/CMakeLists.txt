# decision

# Decision API v1 library: py-json, decision.spec, request checks, calibration,
# executor cards, the engines (laya for now) and the GGUF cache for checkpoint
# directories. Static and PIC: the tools and tests link it, and so does
# llama-server-impl, which is SHARED in DL builds.

add_library(decision STATIC
            decision.cpp
            decision.h
            decision-checkpoint.cpp
            decision-checkpoint.h
            decision-json.cpp
            decision-json.h
            decision-spec.cpp
            decision-spec.h
            decision-request.cpp
            decision-request.h
            decision-calib.cpp
            decision-calib.h
            decision-router.cpp
            decision-router.h
            engine-laya.cpp
            laya-decide.cpp
            laya-decide.h
            ${PROJECT_SOURCE_DIR}/examples/gguf-hash/deps/sha256/sha256.c)

# laya-convert: -m <checkpoint dir> is converted to a cached GGUF (decision-checkpoint.h)
target_link_libraries     (decision PUBLIC laya laya-convert ggml)
target_include_directories(decision PUBLIC . ${PROJECT_SOURCE_DIR}/vendor)
target_include_directories(decision PRIVATE ${PROJECT_SOURCE_DIR}/examples/gguf-hash/deps) # sha256, also used by llama-gguf-hash
target_compile_features   (decision PUBLIC cxx_std_17)
set_target_properties     (decision PROPERTIES POSITION_INDEPENDENT_CODE ON)

if (LLAMA_BUILD_TOOLS)
    set(TARGET llama-decision-bench)
    add_executable(${TARGET} decision-bench.cpp)
    set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME llama-decision-bench)
    if(LLAMA_TOOLS_INSTALL)
        install(TARGETS ${TARGET} RUNTIME)
    endif()
    target_link_libraries  (${TARGET} PRIVATE decision laya Threads::Threads)
    target_compile_features(${TARGET} PRIVATE cxx_std_17)
endif()
