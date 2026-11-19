# Fork-owned test registration. Inclusion does not run tests.

# Internal source adapter; compile directly so the public llama ABI stays unchanged.
llama_build_and_test(test-expert-source.cpp ${PROJECT_SOURCE_DIR}/src/llama-expert-source.cpp LINK ggml Threads::Threads)
llama_build_and_test(test-expert-store.cpp
    ${PROJECT_SOURCE_DIR}/src/llama-expert-source.cpp
    ${PROJECT_SOURCE_DIR}/src/llama-expert-store.cpp
    ${PROJECT_SOURCE_DIR}/src/llama-memory-budget.cpp
    LINK ggml Threads::Threads)
set_tests_properties(test-expert-store PROPERTIES TIMEOUT 30)
llama_build_and_test(test-expert-executor.cpp
    ${PROJECT_SOURCE_DIR}/src/llama-expert-source.cpp
    ${PROJECT_SOURCE_DIR}/src/llama-expert-store.cpp
    ${PROJECT_SOURCE_DIR}/src/llama-memory-budget.cpp
    ${PROJECT_SOURCE_DIR}/src/llama-memory-contract.cpp
    ${PROJECT_SOURCE_DIR}/src/llama-expert-executor.cpp
    LINK ggml Threads::Threads)
set_tests_properties(test-expert-executor PROPERTIES TIMEOUT 30)
llama_build_and_test(test-memory-contract.cpp
    ${PROJECT_SOURCE_DIR}/src/llama-memory-budget.cpp
    ${PROJECT_SOURCE_DIR}/src/llama-memory-contract.cpp
    LINK ggml Threads::Threads)
set_tests_properties(test-memory-contract PROPERTIES TIMEOUT 30)

llama_build_and_test(test-memory-policy.cpp
    ${PROJECT_SOURCE_DIR}/src/llama-memory-policy.cpp LINK Threads::Threads)
set_tests_properties(test-memory-policy PROPERTIES TIMEOUT 10)

target_include_directories(test-memory-policy PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/ggml/include)

llama_build_and_test(test-prefill-window-scheduler.cpp LINK ggml Threads::Threads)
set_tests_properties(test-prefill-window-scheduler PROPERTIES TIMEOUT 30)
