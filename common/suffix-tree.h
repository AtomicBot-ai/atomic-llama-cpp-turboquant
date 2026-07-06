// Copyright 2025 Snowflake Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Ported from https://github.com/snowflakedb/ArcticInference
// Original: csrc/suffix_decoding/suffix_tree.h
//
// Suffix tree for model-free speculative decoding (SuffixDecoding).
// Paper: arxiv:2411.04975 (NeurIPS 2025 Spotlight)

#pragma once

#include <cassert>
#include <deque>
#include <memory>
#include <utility>
#include <vector>

#include "int32-map.h"

struct SuffixGroup;

struct SuffixNode {
    int64_t count = 0;
    int token = 0;
    int length = 0;
    int ref_seq = 0;
    int ref_idx = -1;

    Int32Map<int> endpoints;
    SuffixNode * parent = nullptr;
    Int32Map<std::unique_ptr<SuffixNode>> children;

    SuffixNode * head_child = nullptr;
    SuffixNode * tail_child = nullptr;
    SuffixNode * next_sibling = nullptr;
    SuffixNode * prev_sibling = nullptr;
    std::shared_ptr<SuffixGroup> group = nullptr;

    SuffixNode() = default;

    SuffixNode(int64_t count, int token, int length, int ref_seq, int ref_idx)
        : count(count), token(token), length(length),
          ref_seq(ref_seq), ref_idx(ref_idx) {}

    size_t memory_usage() const {
        size_t total = sizeof(*this);
        total += children.memory_usage();
        total += endpoints.memory_usage();
        return total;
    }
};

struct SuffixGroup {
    SuffixNode * head = nullptr;
    SuffixGroup * next = nullptr;
    SuffixGroup * prev = nullptr;

    SuffixGroup(SuffixNode * head) : head(head) {}
};

struct SuffixDraft {
    std::vector<int32_t> token_ids;
    std::vector<int32_t> parents;
    std::vector<float>   probs;
    float score = 0.0f;
    int match_len = 0;
};

class SuffixTree {
public:
    SuffixTree(int max_depth);

    int num_seqs() const { return static_cast<int>(_seqs.size()); }

    void append(int seq_id, int token);
    void extend(int seq_id, const int32_t * tokens, size_t n_tokens);
    void remove(int seq_id);

    SuffixDraft speculate(const int32_t * context, size_t n_context,
                          int max_spec_tokens,
                          float max_spec_factor,
                          float max_spec_offset,
                          float min_token_prob,
                          bool use_tree_spec);

    std::string check_integrity();
    size_t estimate_memory() const;

private:
    int _max_depth;
    std::unique_ptr<SuffixNode> _root;
    Int32Map<std::vector<int32_t>> _seqs;
    Int32Map<std::deque<SuffixNode *>> _active_nodes;

    std::pair<SuffixNode *, int> _match_context(const int32_t * context, size_t n_context);

    SuffixDraft _speculate_path(SuffixNode * node, int idx,
                                int max_spec_tokens, float min_token_prob);

    SuffixDraft _speculate_tree(SuffixNode * node, int idx,
                                int max_spec_tokens, float min_token_prob);

    std::string _check_node_integrity(SuffixNode * node);
};
