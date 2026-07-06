#include "../tools/server/server-loop-guard.h"
#include "../tools/server/server-task.h"
#include "chat-peg-parser.h"

#undef NDEBUG
#include <cassert>
#include <random>
#include <vector>

static common_reasoning_loop_guard_params test_params() {
    common_reasoning_loop_guard_params params;
    params.min_reasoning_tokens = 1024;
    params.window_tokens = 2048;
    params.max_period = 512;
    params.min_repeated_coverage = 768;
    params.check_interval = 32;
    return params;
}

static void accept_many(server_loop_guard & guard, const std::vector<llama_token> & tokens, server_loop_guard_region region) {
    for (llama_token token : tokens) {
        guard.accept(token, region);
    }
}

static std::vector<llama_token> repeat_block(const std::vector<llama_token> & block, int repeats) {
    std::vector<llama_token> tokens;
    tokens.reserve(block.size() * (size_t) repeats);
    for (int i = 0; i < repeats; ++i) {
        tokens.insert(tokens.end(), block.begin(), block.end());
    }
    return tokens;
}

static void assert_triggered(const server_loop_guard_result & result, const char * kind) {
    assert(result.triggered);
    assert(result.kind == kind);
    assert(result.coverage >= 768);
}

static void assert_not_triggered(const server_loop_guard_result & result) {
    assert(!result.triggered);
}

static void test_task_result_state_suppresses_prefilled_thinking_close_tag() {
    const std::string think_start = "<think>";
    const std::string think_end = "</think>";

    common_chat_parser_params parser_params;
    parser_params.reasoning_format = COMMON_REASONING_FORMAT_NONE;
    parser_params.generation_prompt = think_start;
    parser_params.thinking_start_tag = think_start;
    parser_params.thinking_end_tag = think_end;
    parser_params.parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        return p.literal(think_start) + p.content(p.rest()) + p.end();
    });

    task_result_state state(parser_params);
    std::vector<common_chat_msg_diff> diffs;

    for (size_t i = 0; i < think_end.size(); ++i) {
        diffs.clear();
        state.update_chat_msg(think_end.substr(i, 1), i + 1 < think_end.size(), diffs, false);
        assert(diffs.empty());
        assert(state.chat_msg.content.empty());
        assert(state.chat_msg.reasoning_content.empty());
    }

    diffs.clear();
    state.update_chat_msg("\nSession title", false, diffs, false);
    assert(state.chat_msg.content == "Session title");
    assert(diffs.size() == 1);
    assert(diffs[0].content_delta == "Session title");
}

static void test_task_result_state_suppresses_generated_thinking_block_in_content_mode() {
    const std::string think_start = "<think>";
    const std::string think_end = "</think>";

    common_chat_parser_params parser_params;
    parser_params.reasoning_format = COMMON_REASONING_FORMAT_NONE;
    parser_params.generation_prompt = think_start;
    parser_params.thinking_start_tag = think_start;
    parser_params.thinking_end_tag = think_end;
    parser_params.parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        return p.literal(think_start) + p.content(p.rest()) + p.end();
    });

    task_result_state state(parser_params);
    std::vector<common_chat_msg_diff> diffs;

    const std::string generated_reasoning = think_start + "\ninternal reasoning";
    for (size_t i = 0; i < generated_reasoning.size(); ++i) {
        diffs.clear();
        state.update_chat_msg(generated_reasoning.substr(i, 1), true, diffs, false);
        assert(diffs.empty());
        assert(state.chat_msg.content.empty());
        assert(state.chat_msg.reasoning_content.empty());
    }

    diffs.clear();
    state.update_chat_msg(think_end + "\nSession title", false, diffs, false);
    assert(state.chat_msg.content == "Session title");
    assert(diffs.size() == 1);
    assert(diffs[0].content_delta == "Session title");
}

static void test_task_result_state_suppresses_prefilled_open_thinking_content() {
    const std::string think_start = "<think>";
    const std::string think_end = "</think>";

    common_chat_parser_params parser_params;
    parser_params.reasoning_format = COMMON_REASONING_FORMAT_NONE;
    parser_params.generation_prompt = think_start;
    parser_params.thinking_start_tag = think_start;
    parser_params.thinking_end_tag = think_end;
    parser_params.parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        return p.content(p.rest()) + p.end();
    });

    task_result_state state(parser_params);
    std::vector<common_chat_msg_diff> diffs;

    const std::string generated_reasoning = "\ninternal reasoning";
    for (size_t i = 0; i < generated_reasoning.size(); ++i) {
        diffs.clear();
        state.update_chat_msg(generated_reasoning.substr(i, 1), true, diffs, false);
        assert(diffs.empty());
        assert(state.chat_msg.content.empty());
        assert(state.chat_msg.reasoning_content.empty());
    }

    diffs.clear();
    state.update_chat_msg(think_end + "\nSession title", false, diffs, false);
    assert(state.chat_msg.content == "Session title");
    assert(diffs.size() == 1);
    assert(diffs[0].content_delta == "Session title");
}

static void test_task_result_state_suppresses_leading_thinking_block_without_prefill() {
    const std::string think_start = "<think>";
    const std::string think_end = "</think>";

    common_chat_parser_params parser_params;
    parser_params.reasoning_format = COMMON_REASONING_FORMAT_NONE;
    parser_params.thinking_start_tag = think_start;
    parser_params.thinking_end_tag = think_end;
    parser_params.parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        return p.content(p.rest()) + p.end();
    });

    task_result_state state(parser_params);
    std::vector<common_chat_msg_diff> diffs;

    const std::string generated_reasoning = think_start + "\ninternal reasoning";
    for (size_t i = 0; i < generated_reasoning.size(); ++i) {
        diffs.clear();
        state.update_chat_msg(generated_reasoning.substr(i, 1), true, diffs, false);
        assert(diffs.empty());
        assert(state.chat_msg.content.empty());
        assert(state.chat_msg.reasoning_content.empty());
    }

    diffs.clear();
    state.update_chat_msg(think_end + "\nSession title", false, diffs, false);
    assert(state.chat_msg.content == "Session title");
    assert(diffs.size() == 1);
    assert(diffs[0].content_delta == "Session title");
}

static void test_task_result_state_streams_forced_open_reasoning() {
    const std::string think_start = "<think>";
    const std::string think_end = "</think>";

    common_chat_parser_params parser_params;
    parser_params.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    parser_params.generation_prompt = think_start;
    parser_params.thinking_start_tag = think_start;
    parser_params.thinking_end_tag = think_end;
    parser_params.parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto reasoning = p.optional(think_start + p.reasoning(p.until(think_end)) + think_end + p.space());
        return p.sequence({ reasoning, p.content(p.rest()), p.end() });
    });

    task_result_state state(parser_params);
    std::vector<common_chat_msg_diff> diffs;

    const std::string generated_reasoning = "internal reasoning";
    state.update_chat_msg(generated_reasoning, true, diffs, false);

    assert(state.chat_msg.reasoning_content == generated_reasoning);
    assert(state.chat_msg.content.empty());
    assert(diffs.size() == 1);
    assert(diffs[0].reasoning_content_delta == generated_reasoning);
    assert(diffs[0].content_delta.empty());
}

int main() {
    test_task_result_state_suppresses_prefilled_thinking_close_tag();
    test_task_result_state_suppresses_generated_thinking_block_in_content_mode();
    test_task_result_state_suppresses_prefilled_open_thinking_content();
    test_task_result_state_suppresses_leading_thinking_block_without_prefill();
    test_task_result_state_streams_forced_open_reasoning();

    {
        server_loop_guard guard(test_params());
        accept_many(guard, std::vector<llama_token>(1200, 42), SERVER_LOOP_REGION_REASONING);
        const auto result = guard.check(SERVER_LOOP_REGION_REASONING);
        assert_triggered(result, "periodic_tail");
        assert(result.period == 1);
    }

    {
        server_loop_guard guard(test_params());
        accept_many(guard, repeat_block({1, 2}, 600), SERVER_LOOP_REGION_REASONING);
        const auto result = guard.check(SERVER_LOOP_REGION_REASONING);
        assert_triggered(result, "periodic_tail");
        assert(result.period == 2);
    }

    {
        server_loop_guard guard(test_params());
        accept_many(guard, repeat_block({1, 2, 3, 4, 5}, 240), SERVER_LOOP_REGION_REASONING);
        const auto result = guard.check(SERVER_LOOP_REGION_REASONING);
        assert_triggered(result, "periodic_tail");
        assert(result.period == 5);
    }

    {
        server_loop_guard guard(test_params());
        std::vector<llama_token> sentence;
        for (int i = 0; i < 32; ++i) {
            sentence.push_back(100 + i);
        }
        accept_many(guard, repeat_block(sentence, 40), SERVER_LOOP_REGION_REASONING);
        const auto result = guard.check(SERVER_LOOP_REGION_REASONING);
        assert(result.triggered);
        assert(result.kind == "periodic_tail" || result.kind == "ngram_dominance");
    }

    {
        server_loop_guard guard(test_params());
        std::vector<llama_token> block;
        for (int i = 0; i < 200; ++i) {
            block.push_back(1000 + i);
        }
        accept_many(guard, repeat_block(block, 6), SERVER_LOOP_REGION_REASONING);
        const auto result = guard.check(SERVER_LOOP_REGION_REASONING);
        assert_triggered(result, "periodic_tail");
        assert(result.period == 200);
    }

    {
        server_loop_guard guard(test_params());
        std::vector<llama_token> tokens;
        tokens.reserve(1200);
        std::mt19937 rng(1234);
        std::uniform_int_distribution<int> dist(0, 5);
        for (int i = 0; i < 1200; ++i) {
            tokens.push_back(200 + dist(rng));
        }
        accept_many(guard, tokens, SERVER_LOOP_REGION_REASONING);
        assert_triggered(guard.check(SERVER_LOOP_REGION_REASONING), "low_entropy");
    }

    {
        server_loop_guard guard(test_params());
        for (int i = 0; i < 20000; ++i) {
            const llama_token token = 1000 + (llama_token) ((i * 48271LL + (i / 37) * 7919LL + (i % 11) * 104729LL) % 50000);
            guard.accept(token, SERVER_LOOP_REGION_REASONING);
            if (guard.should_check(SERVER_LOOP_REGION_REASONING, false, false)) {
                assert_not_triggered(guard.check(SERVER_LOOP_REGION_REASONING));
            }
        }
    }

    {
        server_loop_guard guard(test_params());
        for (int line = 0; line < 220; ++line) {
            const std::vector<llama_token> code_line = {
                100, 101, 200 + line, 102, 300 + (line % 97), 103, 104, 400 + (line % 131), 105
            };
            accept_many(guard, code_line, SERVER_LOOP_REGION_VISIBLE);
        }
        assert_not_triggered(guard.check(SERVER_LOOP_REGION_VISIBLE));
    }

    {
        server_loop_guard guard(test_params());
        for (int row = 0; row < 180; ++row) {
            const std::vector<llama_token> table_row = {
                500, 600 + row, 501, 700 + (row * 7) % 997, 501, 800 + (row * 13) % 997, 500, 502
            };
            accept_many(guard, table_row, SERVER_LOOP_REGION_VISIBLE);
        }
        assert_not_triggered(guard.check(SERVER_LOOP_REGION_VISIBLE));
    }

    {
        server_loop_guard guard(test_params());
        for (int block = 0; block < 80; ++block) {
            for (int pos = 0; pos < 16; ++pos) {
                const bool noisy = pos % 4 == 0;
                guard.accept(noisy ? (llama_token) (10000 + block * 16 + pos) : (llama_token) (900 + pos),
                        SERVER_LOOP_REGION_REASONING);
            }
        }
        assert_not_triggered(guard.check(SERVER_LOOP_REGION_REASONING));
    }

    {
        common_reasoning_loop_guard_params params = test_params();
        params.min_reasoning_tokens = 0;
        params.min_repeated_coverage = 1;
        params.window_tokens = 64;
        params.max_period = 8;
        params.check_interval = 32;
        server_loop_guard guard(params);
        for (int i = 0; i < 31; ++i) {
            guard.accept(1, SERVER_LOOP_REGION_REASONING);
            assert(!guard.should_check(SERVER_LOOP_REGION_REASONING, false, false));
        }
        guard.accept(1, SERVER_LOOP_REGION_REASONING);
        assert(guard.should_check(SERVER_LOOP_REGION_REASONING, false, false));
    }

    {
        server_loop_guard guard(test_params());
        accept_many(guard, std::vector<llama_token>(1200, 42), SERVER_LOOP_REGION_REASONING);
        assert(guard.check(SERVER_LOOP_REGION_REASONING).triggered);
        guard.reset();
        for (int i = 0; i < 2000; ++i) {
            guard.accept((llama_token) (1000 + (i * 37) % 50000), SERVER_LOOP_REGION_REASONING);
        }
        assert_not_triggered(guard.check(SERVER_LOOP_REGION_REASONING));
    }

    {
        server_loop_guard guard(test_params());
        accept_many(guard, std::vector<llama_token>(1200, 42), SERVER_LOOP_REGION_REASONING);
        assert(guard.check(SERVER_LOOP_REGION_REASONING).triggered);

        common_reasoning_loop_guard_params params = test_params();
        params.mode = COMMON_REASONING_LOOP_GUARD_OFF;
        guard.configure(params);
        accept_many(guard, std::vector<llama_token>(1200, 42), SERVER_LOOP_REGION_REASONING);
        assert(!guard.should_check(SERVER_LOOP_REGION_REASONING, false, false));
        assert_not_triggered(guard.check(SERVER_LOOP_REGION_REASONING));
        assert(guard.seen(SERVER_LOOP_REGION_REASONING) == 1200);
    }

    return 0;
}
