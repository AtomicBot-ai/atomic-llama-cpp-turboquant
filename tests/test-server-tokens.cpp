// Phase C.2.0 — unit tests for server_tokens coexistence APIs introduced for MTP+mmproj dispatch.
//
// Scope:
//   - is_pure_text_continuation(from_idx)
//   - last_image_end_idx()
//   - get_text_tokens_post_media()
//
// These APIs are foundational and do not change runtime behavior; they expose information that
// future per-batch dispatch will use. This file covers what is testable WITHOUT loading a model
// or running the full mtmd pipeline:
//   - non-multimodal (has_mtmd=false) buffers
//   - empty multimodal (has_mtmd=true, no chunks) buffers
//   - empty buffer (size==0) edge cases
//
// The WITH-image cases require a real mtmd_input_chunk (which goes through the mtmd public API
// requiring an image file + clip model). Those are covered by integration tests in C.2.4 once
// the dispatch behavior is wired up.

#include "server-common.h"

#include <cstdio>
#include <cstdlib>

#define CHECK(cond, msg)                                                                              \
    do {                                                                                              \
        if (!(cond)) {                                                                                \
            std::fprintf(stderr, "FAIL %s:%d  %s   (cond: %s)\n", __FILE__, __LINE__, msg, #cond);    \
            std::exit(1);                                                                             \
        }                                                                                             \
    } while (0)

static void test_non_mtmd_empty_buffer() {
    llama_tokens t;
    server_tokens st(t, /*has_mtmd*/ false);

    CHECK(st.size() == 0,                              "empty size");
    CHECK(st.empty(),                                  "empty()");
    CHECK(st.last_image_end_idx() == 0,                "last_image_end_idx empty");
    CHECK(st.is_pure_text_continuation(0),             "pure-text @ 0 (empty)");
    CHECK(st.is_pure_text_continuation(100),           "pure-text @ 100 (empty/past-end)");

    llama_tokens out = st.get_text_tokens_post_media();
    CHECK(out.empty(),                                 "post-media tail empty for empty buffer");
}

static void test_non_mtmd_text_only() {
    llama_tokens t = {1, 2, 3, 4, 5};
    server_tokens st(t, /*has_mtmd*/ false);

    CHECK(st.size() == 5,                              "size==5");
    CHECK(!st.empty(),                                 "!empty");
    CHECK(st.last_image_end_idx() == 0,                "last_image_end_idx text-only -> 0");

    // is_pure_text_continuation always true when !has_mtmd
    CHECK(st.is_pure_text_continuation(0),             "pure-text @ 0");
    CHECK(st.is_pure_text_continuation(3),             "pure-text @ 3");
    CHECK(st.is_pure_text_continuation(5),             "pure-text @ 5 (at end)");
    CHECK(st.is_pure_text_continuation(999),           "pure-text @ 999 (past end)");

    // For non-mtmd, get_text_tokens_post_media returns all tokens (no NULL stripped because none present).
    llama_tokens out = st.get_text_tokens_post_media();
    CHECK(out.size() == 5,                             "post-media tail size matches buffer");
    for (size_t i = 0; i < out.size(); ++i) {
        CHECK(out[i] == t[i],                          "post-media tail token matches");
    }

    // get_text_tokens() must still return the canonical reference for non-mtmd path.
    const llama_tokens & ref = st.get_text_tokens();
    CHECK(ref.size() == 5,                             "get_text_tokens() size");
    CHECK(ref.data() != out.data(),                    "post-media tail is a distinct copy");
}

static void test_mtmd_empty_chunks() {
    // server_tokens with has_mtmd=true but no media chunks added: same observable behavior as non-mtmd
    // for the new APIs (per-API contract: empty map → return as text-only).
    // We construct via the llama_tokens ctor + force has_mtmd=true via the public mutable field
    // (server_tokens exposes has_mtmd as public — see server-common.h:126).
    llama_tokens t = {10, 20, 30};
    server_tokens st(t, /*has_mtmd*/ false);
    st.has_mtmd = true;  // simulate mtmd-enabled buffer with no chunks yet

    CHECK(st.last_image_end_idx() == 0,                "mtmd+empty-map: last_image_end_idx==0");
    CHECK(st.is_pure_text_continuation(0),             "mtmd+empty-map: pure @ 0");
    CHECK(st.is_pure_text_continuation(3),             "mtmd+empty-map: pure @ 3");
    CHECK(st.is_pure_text_continuation(999),           "mtmd+empty-map: pure @ past-end");

    llama_tokens out = st.get_text_tokens_post_media();
    CHECK(out.size() == 3,                             "mtmd+empty-map: tail returns all text");
    CHECK(out[0] == 10 && out[1] == 20 && out[2] == 30, "mtmd+empty-map: tail content matches");
}

static void test_pure_text_continuation_semantics() {
    // The contract: is_pure_text_continuation(from_idx) returns true iff there is NO image chunk
    // extending past from_idx. We can verify the non-mtmd / empty-mtmd branches here (the
    // with-image branch is exercised by integration tests once mtmd is wired up).
    llama_tokens t = {7, 8, 9};
    server_tokens st(t, false);

    CHECK(st.is_pure_text_continuation(0),             "from_idx<size: true");
    CHECK(st.is_pure_text_continuation(2),             "from_idx<size: true");
    CHECK(st.is_pure_text_continuation(3),             "from_idx==size: true");
    CHECK(st.is_pure_text_continuation(4),             "from_idx>size: true (past end)");
    CHECK(st.is_pure_text_continuation(SIZE_MAX),      "from_idx=SIZE_MAX: true (past end)");
}

int main() {
    test_non_mtmd_empty_buffer();
    std::printf("[server_tokens] non_mtmd_empty_buffer            OK\n");

    test_non_mtmd_text_only();
    std::printf("[server_tokens] non_mtmd_text_only               OK\n");

    test_mtmd_empty_chunks();
    std::printf("[server_tokens] mtmd_empty_chunks                OK\n");

    test_pure_text_continuation_semantics();
    std::printf("[server_tokens] pure_text_continuation_semantics OK\n");

    std::printf("ALL PASS — 4 test groups, server_tokens C.2.0 foundational API\n");
    return 0;
}
