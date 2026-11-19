set(TARGET llama-expert-source)
add_executable(${TARGET} expert-source.cpp
    ${PROJECT_SOURCE_DIR}/src/llama-expert-source.cpp
    ${PROJECT_SOURCE_DIR}/src/llama-expert-store.cpp
    ${PROJECT_SOURCE_DIR}/src/llama-memory-budget.cpp)
target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(${TARGET} PRIVATE ggml Threads::Threads)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()

add_executable(llama-expert-infer inference.cpp)
target_include_directories(llama-expert-infer PRIVATE ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/vendor)
target_link_libraries(llama-expert-infer PRIVATE llama Threads::Threads)
target_compile_features(llama-expert-infer PRIVATE cxx_std_17)

option(ATOMIC_ENGINE_IO_URING_PROBE "Enable the Linux expert read cancellation diagnostic" OFF)
if (ATOMIC_ENGINE_IO_URING_PROBE)
    if (NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        message(FATAL_ERROR "The expert read cancellation diagnostic requires Linux")
    endif()
    find_path(ATOMIC_ENGINE_URING_INCLUDE_DIR liburing.h)
    find_library(ATOMIC_ENGINE_URING_LIBRARY uring)
    if (NOT ATOMIC_ENGINE_URING_INCLUDE_DIR OR NOT ATOMIC_ENGINE_URING_LIBRARY)
        message(FATAL_ERROR "The expert read cancellation diagnostic requires liburing headers and library")
    endif()
    target_include_directories(llama-expert-infer PRIVATE ${ATOMIC_ENGINE_URING_INCLUDE_DIR})
    target_link_libraries(llama-expert-infer PRIVATE ${ATOMIC_ENGINE_URING_LIBRARY})
    target_compile_definitions(llama-expert-infer PRIVATE ATOMIC_ENGINE_IO_URING_PROBE)
endif()

# Opt-in diagnostic; also builds against a pristine upstream library.
add_executable(llama-native-probe EXCLUDE_FROM_ALL native-probe.cpp)
target_include_directories(llama-native-probe PRIVATE ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/vendor)
target_link_libraries(llama-native-probe PRIVATE llama Threads::Threads)
target_compile_features(llama-native-probe PRIVATE cxx_std_17)

# Standalone P10B diagnostic. A CPU build can exercise device discovery.
add_executable(llama-discrete-probe EXCLUDE_FROM_ALL discrete-probe.cpp
    ${PROJECT_SOURCE_DIR}/src/llama-memory-budget.cpp)
target_include_directories(llama-discrete-probe PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(llama-discrete-probe PRIVATE ggml Threads::Threads)
target_compile_features(llama-discrete-probe PRIVATE cxx_std_17)

option(ATOMIC_ENGINE_CUDA_VMM_PROBE "Configure the standalone Linux CUDA VMM probe" OFF)
if (ATOMIC_ENGINE_CUDA_VMM_PROBE)
    if (NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        message(FATAL_ERROR "The CUDA VMM probe requires Linux")
    endif()
    find_package(CUDAToolkit 11 REQUIRED)
    add_executable(llama-cuda-vmm-probe EXCLUDE_FROM_ALL cuda-vmm-probe.cpp
        ${PROJECT_SOURCE_DIR}/src/llama-memory-budget.cpp)
    target_include_directories(llama-cuda-vmm-probe PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(llama-cuda-vmm-probe PRIVATE CUDA::cuda_driver Threads::Threads)
    target_compile_features(llama-cuda-vmm-probe PRIVATE cxx_std_17)
endif()

# Opt-in Qwen4Exp packed-slice qualification; not part of aggregate verification.
add_executable(llama-qwen38-cuda-probe EXCLUDE_FROM_ALL qwen38-cuda-probe.cpp
    ${PROJECT_SOURCE_DIR}/src/llama-expert-source.cpp
    ${PROJECT_SOURCE_DIR}/src/llama-expert-store.cpp
    ${PROJECT_SOURCE_DIR}/src/llama-memory-budget.cpp
    ${PROJECT_SOURCE_DIR}/src/llama-memory-contract.cpp
    ${PROJECT_SOURCE_DIR}/src/llama-expert-executor.cpp)
target_include_directories(llama-qwen38-cuda-probe PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(llama-qwen38-cuda-probe PRIVATE ggml Threads::Threads)
target_compile_features(llama-qwen38-cuda-probe PRIVATE cxx_std_17)

add_executable(llama-qwen38-cuda-admission EXCLUDE_FROM_ALL qwen38-cuda-admission.cpp)
target_include_directories(llama-qwen38-cuda-admission PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(llama-qwen38-cuda-admission PRIVATE llama)
target_compile_features(llama-qwen38-cuda-admission PRIVATE cxx_std_17)

add_executable(llama-qwen38-cuda-compare EXCLUDE_FROM_ALL qwen38-cuda-compare.cpp)
target_include_directories(llama-qwen38-cuda-compare PRIVATE ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/vendor)
target_link_libraries(llama-qwen38-cuda-compare PRIVATE llama)
target_compile_features(llama-qwen38-cuda-compare PRIVATE cxx_std_17)
