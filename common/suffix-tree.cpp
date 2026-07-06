// Copyright 2025 Snowflake Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Ported from https://github.com/snowflakedb/ArcticInference
// Original: csrc/suffix_decoding/suffix_tree.cc

#include "suffix-tree.h"

#include <cassert>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#define CHECK_OR_RETURN(cond) \
    if (!(cond)) return "Integrity check failed at line " + \
                        std::to_string(__LINE__) + ": " + #cond;

SuffixTree::SuffixTree(int max_depth)
    : _max_depth(max_depth), _root(new SuffixNode()) {
}

static void _remove_from_siblings(SuffixNode * node) {
    assert(node->parent);
    SuffixGroup * group = node->group.get();
    if (group->head == node) {
        if (node->next_sibling && node->next_sibling->count == node->count) {
            group->head = node->next_sibling;
            node->group.reset();
        } else {
            if (group->prev) { group->prev->next = group->next; }
            if (group->next) { group->next->prev = group->prev; }
            group->prev = group->next = nullptr;
        }
    } else {
        node->group.reset();
    }
    if (node->next_sibling) {
        node->next_sibling->prev_sibling = node->prev_sibling;
    } else {
        node->parent->tail_child = node->prev_sibling;
    }
    if (node->prev_sibling) {
        node->prev_sibling->next_sibling = node->next_sibling;
    } else {
        node->parent->head_child = node->next_sibling;
    }
    node->prev_sibling = node->next_sibling = nullptr;
}

static void _insert_into_siblings_before(SuffixNode * node, SuffixNode * other) {
    assert(node->parent);
    assert(node->parent == other->parent);
    if (other->prev_sibling) {
        other->prev_sibling->next_sibling = node;
    } else {
        node->parent->head_child = node;
    }
    node->next_sibling = other;
    node->prev_sibling = other->prev_sibling;
    other->prev_sibling = node;

    SuffixNode * prev_sibling = node->prev_sibling;
    if (prev_sibling && node->count == prev_sibling->count) {
        node->group = prev_sibling->group;
    } else if (node->count == other->count) {
        node->group = other->group;
        node->group->head = node;
    } else {
        SuffixGroup * group = node->group.get();
        if (!group) {
            group = new SuffixGroup(node);
            node->group.reset(group);
        }
        assert(group->head == node && !group->next && !group->prev);
        if (prev_sibling) {
            group->prev = prev_sibling->group.get();
            group->prev->next = group;
        }
        group->next = other->group.get();
        group->next->prev = group;
    }
}

static void _insert_into_siblings_after(SuffixNode * node, SuffixNode * other) {
    assert(node->parent);
    assert(node->parent == other->parent);
    if (other->next_sibling) {
        other->next_sibling->prev_sibling = node;
    } else {
        node->parent->tail_child = node;
    }
    node->prev_sibling = other;
    node->next_sibling = other->next_sibling;
    other->next_sibling = node;

    SuffixNode * next_sibling = node->next_sibling;
    if (next_sibling && node->count == next_sibling->count) {
        node->group = next_sibling->group;
        if (node->group->head == next_sibling) {
            node->group->head = node;
        }
    } else if (node->count == other->count) {
        node->group = other->group;
    } else {
        SuffixGroup * group = node->group.get();
        if (!group) {
            group = new SuffixGroup(node);
            node->group.reset(group);
        }
        assert(group->head == node && !group->next && !group->prev);
        if (next_sibling) {
            group->next = next_sibling->group.get();
            group->next->prev = group;
        }
        group->prev = other->group.get();
        group->prev->next = group;
    }
}

static void _replace_in_siblings(SuffixNode * old_node, SuffixNode * new_node) {
    assert(old_node->count == new_node->count);
    assert(old_node->parent);
    if (old_node->next_sibling) {
        old_node->next_sibling->prev_sibling = new_node;
    } else {
        old_node->parent->tail_child = new_node;
    }
    if (old_node->prev_sibling) {
        old_node->prev_sibling->next_sibling = new_node;
    } else {
        old_node->parent->head_child = new_node;
    }
    new_node->prev_sibling = old_node->prev_sibling;
    new_node->next_sibling = old_node->next_sibling;
    old_node->prev_sibling = old_node->next_sibling = nullptr;

    SuffixGroup * group = old_node->group.get();
    if (group->head == old_node) { group->head = new_node; }
    new_node->group = old_node->group;
    old_node->group.reset();
}

static void _increment_count(SuffixNode * node) {
    if (!node->parent) {
        node->count += 1;
        return;
    }
    if (!node->prev_sibling || node->prev_sibling->count > node->count + 1) {
        assert(node->group->head == node);
        if (!node->next_sibling || node->next_sibling->count < node->count) {
            assert(node->group.use_count() == 1);
            node->count += 1;
        } else {
            assert(node->next_sibling->count == node->count);
            SuffixGroup * orig_group = node->group.get();
            orig_group->head = node->next_sibling;
            SuffixGroup * new_group = new SuffixGroup(node);
            new_group->next = orig_group;
            if (orig_group->prev) {
                new_group->prev = orig_group->prev;
                new_group->prev->next = new_group;
            }
            orig_group->prev = new_group;
            node->group.reset(new_group);
            node->count += 1;
        }
    } else {
        assert(node->prev_sibling->count >= node->count);
        SuffixNode * other_node = node->prev_sibling->group->head;
        _remove_from_siblings(node);
        node->count += 1;
        _insert_into_siblings_before(node, other_node);
    }
}

static void _decrement_count(SuffixNode * node) {
    assert(node->count > 0);
    if (!node->parent) {
        node->count -= 1;
        return;
    }
    if (!node->next_sibling || node->next_sibling->count < node->count - 1) {
        if (!node->prev_sibling || node->prev_sibling->count > node->count) {
            assert(node->group.use_count() == 1);
            node->count -= 1;
        } else {
            assert(node->prev_sibling->count == node->count);
            SuffixGroup * orig_group = node->group.get();
            SuffixGroup * new_group = new SuffixGroup(node);
            new_group->prev = orig_group;
            if (orig_group->next) {
                new_group->next = orig_group->next;
                new_group->next->prev = new_group;
            }
            orig_group->next = new_group;
            node->group.reset(new_group);
            node->count -= 1;
        }
    } else if (node->next_sibling->count == node->count - 1) {
        assert(node->next_sibling->group->head == node->next_sibling);
        node->next_sibling->group->head = node;
        if (node->group->head == node) {
            assert(node->group.use_count() == 1);
            SuffixGroup * group = node->group.get();
            if (group->prev) { group->prev->next = group->next; }
            group->next->prev = group->prev;
        }
        node->group = node->next_sibling->group;
        node->count -= 1;
    } else {
        assert(node->next_sibling->count == node->count);
        SuffixGroup * other_group = node->group->next;
        _remove_from_siblings(node);
        node->count -= 1;
        if (!other_group) {
            _insert_into_siblings_after(node, node->parent->tail_child);
        } else {
            _insert_into_siblings_before(node, other_group->head);
        }
    }
}

void SuffixTree::append(int seq_id, int token) {
    if (!_seqs.contains(seq_id)) {
        assert(!_active_nodes.contains(seq_id));
        _seqs.emplace(seq_id);
        _active_nodes.emplace(seq_id);
    }

    std::vector<int32_t> & seq = _seqs[seq_id];
    std::deque<SuffixNode *> & active_nodes = _active_nodes[seq_id];

    active_nodes.push_back(_root.get());
    _root->endpoints[seq_id] = static_cast<int32_t>(seq.size());
    _root->count += 1;

    if (active_nodes.size() > static_cast<size_t>(_max_depth)) {
        active_nodes.pop_front();
    }
    seq.push_back(token);
    int32_t seq_len = static_cast<int32_t>(seq.size());

    for (SuffixNode *& active_node : active_nodes) {
        SuffixNode * node = active_node;
        SuffixNode * child = nullptr;
        auto it = node->children.find(token);
        if (it != node->children.end()) {
            child = it->second.get();
        }

        assert(node->endpoints.contains(seq_id));
        assert(node->endpoints[seq_id] == static_cast<int>(seq.size()) - 1);

        if (child == nullptr) {
            if (node->count == 1 && node != _root.get()) {
                // Case 1a: extend leaf
                assert(node->children.empty());
                assert(node->ref_seq == seq_id);
                node->length += 1;
                node->endpoints[seq_id] += 1;
            } else {
                // Case 1b: create new child
                SuffixNode * new_child = new SuffixNode(1, token, 1, seq_id, seq_len - 1);
                new_child->parent = node;
                new_child->endpoints[seq_id] = seq_len;
                node->children.emplace(token, new_child);
                node->endpoints.erase(seq_id);

                if (node->children.size() == 1) {
                    assert(!node->head_child && !node->tail_child);
                    node->head_child = node->tail_child = new_child;
                    new_child->group.reset(new SuffixGroup(new_child));
                } else {
                    assert(node->tail_child);
                    _insert_into_siblings_after(new_child, node->tail_child);
                }
                active_node = new_child;
            }
        } else if (node->count == child->count + 1 && node != _root.get()) {
            assert(node->children.size() == 1);
            assert(node->endpoints.size() == 1);
            if (child->length == 1) {
                // Case 2a: fuse node with child
                child->count += 1;
                child->token = node->token;
                child->length = node->length + 1;
                child->ref_seq = seq_id;
                child->ref_idx = seq_len - child->length;
                child->endpoints[seq_id] = seq_len;
                child->parent = node->parent;
                _replace_in_siblings(node, child);

                SuffixNode * parent = node->parent;
                assert(parent->children.contains(node->token));
                assert(parent->children[node->token].get() == node);
                SuffixNode * tmp = node->children[token].release();
                parent->children[child->token].reset(tmp);
                active_node = child;
            } else {
                // Case 2b: extend node into child
                node->length += 1;
                node->endpoints[seq_id] += 1;
                node->ref_seq = seq_id;
                node->ref_idx = seq_len - node->length;

                child->length -= 1;
                child->ref_idx += 1;
                child->token = _seqs[child->ref_seq][child->ref_idx];
                if (child->token != token) {
                    SuffixNode * tmp = node->children[token].release();
                    node->children.emplace(child->token, tmp);
                    node->children.erase(token);
                }
            }
        } else {
            if (child->length == 1) {
                // Case 3a: move into child
                node->endpoints.erase(seq_id);
                child->endpoints[seq_id] = seq_len;
                _increment_count(child);
                active_node = child;
            } else {
                // Case 3b: split child
                SuffixNode * new_node = new SuffixNode(child->count, token, 1, seq_id, seq_len - 1);
                new_node->parent = node;
                _replace_in_siblings(child, new_node);

                node->children[token].release();
                node->children[token].reset(new_node);

                child->length -= 1;
                child->ref_idx += 1;
                child->token = _seqs[child->ref_seq][child->ref_idx];

                new_node->children.emplace(child->token, child);
                child->parent = new_node;

                node->endpoints.erase(seq_id);
                new_node->endpoints[seq_id] = seq_len;

                new_node->head_child = new_node->tail_child = child;
                child->group.reset(new SuffixGroup(child));

                _increment_count(new_node);
                active_node = new_node;
            }
        }
    }
}

void SuffixTree::extend(int seq_id, const int32_t * tokens, size_t n_tokens) {
    for (size_t i = 0; i < n_tokens; i++) {
        append(seq_id, tokens[i]);
    }
}

void SuffixTree::remove(int seq_id) {
    const std::vector<int32_t> & seq = _seqs[seq_id];
    std::vector<SuffixNode *> path;
    for (size_t start = 0; start < seq.size(); start++) {
        SuffixNode * node = _root.get();
        node->count--;
        size_t idx = start;
        path.clear();
        while (idx < seq.size()) {
            int token = seq[idx];
            if (!node->children.contains(token)) { break; }
            SuffixNode * child = node->children[token].get();
            if (child->count > 1) {
                _decrement_count(child);
            } else {
                assert(child->count == 1);
                _remove_from_siblings(child);
                node->children.erase(token);
                break;
            }
            if (child->endpoints.contains(seq_id)) {
                child->endpoints.erase(seq_id);
            }
            idx += child->length;
            node = child;
            path.push_back(node);
        }
        // merge node with its only child if counts match
        if (node != _root.get() && node->children.size() == 1) {
            const auto & it = *node->children.begin();
            std::unique_ptr<SuffixNode> & child_uptr = node->children[it.first];
            if (node->count == child_uptr->count) {
                child_uptr->token = node->token;
                child_uptr->length += node->length;
                child_uptr->ref_idx -= node->length;
                child_uptr->parent = node->parent;
                _replace_in_siblings(node, child_uptr.get());
                path.back() = node = child_uptr.release();
                node->parent->children[node->token].reset(node);
            }
        }
        // update ref_seq/ref_idx for nodes referencing removed sequence
        SuffixNode * leaf = node;
        int distance = 0;
        while (!leaf->children.empty()) {
            leaf = (*leaf->children.begin()).second.get();
            distance += leaf->length;
        }
        if (leaf->endpoints.empty() || leaf->endpoints.contains(seq_id)) {
            continue;
        }
        const auto & ref = *leaf->endpoints.begin();
        int32_t new_ref_seq = ref.first;
        int32_t ref_idx = ref.second - distance;
        while (!path.empty()) {
            SuffixNode * n = path.back();
            path.pop_back();
            ref_idx -= n->length;
            if (n->ref_seq == seq_id) {
                n->ref_seq = new_ref_seq;
                n->ref_idx = ref_idx;
            }
        }
    }
    _seqs.erase(seq_id);
    _active_nodes.erase(seq_id);
}

SuffixDraft SuffixTree::speculate(const int32_t * context, size_t n_context,
                                  int max_spec_tokens,
                                  float max_spec_factor,
                                  float max_spec_offset,
                                  float min_token_prob,
                                  bool use_tree_spec) {
    SuffixDraft best_draft;
    for (size_t match_len = 1; match_len < n_context; match_len++) {
        auto [node, idx] = _match_context(context + n_context - match_len, match_len);
        if (node == nullptr) { break; }
        int max_tokens = std::min(max_spec_tokens,
                                  static_cast<int>(match_len * max_spec_factor
                                                   + max_spec_offset + 1e-6));
        max_tokens = std::max(max_tokens, 0);
        SuffixDraft draft;
        if (use_tree_spec) {
            draft = _speculate_tree(node, idx, max_tokens, min_token_prob);
        } else {
            draft = _speculate_path(node, idx, max_tokens, min_token_prob);
        }
        if (draft.score >= best_draft.score) {
            best_draft = std::move(draft);
            best_draft.match_len = match_len;
        }
    }
    return best_draft;
}

std::pair<SuffixNode *, int> SuffixTree::_match_context(const int32_t * context, size_t n_context) {
    SuffixNode * node = _root.get();
    int idx = 0;
    const int32_t * ref_data = nullptr;
    for (size_t i = 0; i < n_context; i++) {
        int32_t token = context[i];
        if (idx >= node->length) {
            auto it = node->children.find(token);
            if (it == node->children.end()) { return {nullptr, -1}; }
            node = it->second.get();
            ref_data = _seqs[node->ref_seq].data() + node->ref_idx;
            idx = 0;
        }
        assert(idx < node->length);
        if (ref_data[idx] != token) { return {nullptr, -1}; }
        idx++;
    }
    return {node, idx};
}

SuffixDraft SuffixTree::_speculate_path(SuffixNode * node, int idx,
                                        int max_spec_tokens,
                                        float min_token_prob) {
    SuffixDraft ret;
    float prob = 1.0f;
    const int32_t * ref_data = _seqs[node->ref_seq].data() + node->ref_idx;
    while (static_cast<int>(ret.token_ids.size()) < max_spec_tokens && prob >= min_token_prob) {
        if (idx < node->length) {
            ret.parents.push_back(static_cast<int>(ret.token_ids.size()) - 1);
            ret.token_ids.push_back(ref_data[idx]);
            ret.probs.push_back(prob);
            ret.score += prob;
            idx++;
        } else {
            SuffixNode * child = node->head_child;
            if (child == nullptr) { break; }
            int64_t count = child->count;
            prob *= static_cast<float>(count) / node->count;
            node = child;
            ref_data = _seqs[node->ref_seq].data() + node->ref_idx;
            idx = 0;
        }
    }
    return ret;
}

struct SuffixHeapItem {
    float prob;
    SuffixNode * node;
    int idx;
    int parent;

    SuffixHeapItem(float p, SuffixNode * n, int i, int par)
        : prob(p), node(n), idx(i), parent(par) {}
};

struct SuffixHeapItemCmp {
    bool operator()(const SuffixHeapItem & a, const SuffixHeapItem & b) const {
        return a.prob < b.prob;
    }
};

SuffixDraft SuffixTree::_speculate_tree(SuffixNode * node, int idx,
                                        int max_spec_tokens,
                                        float min_token_prob) {
    SuffixDraft ret;
    std::priority_queue<SuffixHeapItem, std::vector<SuffixHeapItem>, SuffixHeapItemCmp> queue;
    queue.emplace(1.0f, node, idx, -1);
    while (static_cast<int>(ret.token_ids.size()) < max_spec_tokens && !queue.empty()) {
        SuffixHeapItem it = queue.top();
        queue.pop();
        if (it.idx < it.node->length) {
            int32_t token = _seqs[it.node->ref_seq][it.node->ref_idx + it.idx];
            ret.token_ids.push_back(token);
            ret.parents.push_back(it.parent);
            ret.probs.push_back(it.prob);
            ret.score += it.prob;
            queue.emplace(it.prob, it.node, it.idx + 1,
                          static_cast<int>(ret.token_ids.size()) - 1);
        } else {
            SuffixNode * child = it.node->head_child;
            while (child) {
                float prob = it.prob * child->count /
                    static_cast<float>(it.node->count);
                if (prob < min_token_prob) { break; }
                queue.emplace(prob, child, 0, it.parent);
                child = child->next_sibling;
            }
        }
    }
    return ret;
}

std::string SuffixTree::check_integrity() {
    std::queue<SuffixNode *> q;
    q.push(_root.get());
    while (!q.empty()) {
        SuffixNode * node = q.front();
        q.pop();
        std::string ret = _check_node_integrity(node);
        if (!ret.empty()) { return ret; }
        for (const auto & [token, child] : node->children) {
            q.push(child.get());
        }
    }
    std::unordered_map<SuffixNode *, int64_t> visit_count;
    for (const auto & [seq_id, seq] : _seqs) {
        for (size_t start = 0; start < seq.size(); start++) {
            size_t idx = start;
            SuffixNode * node = _root.get();
            visit_count[node]++;
            while (idx < seq.size() && static_cast<int>(idx - start) < _max_depth) {
                CHECK_OR_RETURN(node->children.contains(seq[idx]));
                node = node->children[seq[idx]].get();
                visit_count[node]++;
                CHECK_OR_RETURN(idx + node->length <= seq.size());
                for (int i = 0; i < node->length; ++i) {
                    int ref_seq = node->ref_seq;
                    int ref_idx = node->ref_idx + i;
                    CHECK_OR_RETURN(seq[idx + i] == _seqs[ref_seq][ref_idx]);
                }
                idx += node->length;
            }
            CHECK_OR_RETURN(node->endpoints.contains(seq_id));
        }
    }
    assert(q.empty());
    q.push(_root.get());
    while (!q.empty()) {
        SuffixNode * node = q.front();
        q.pop();
        CHECK_OR_RETURN(node->count == visit_count[node]);
        for (const auto & [token, child] : node->children) {
            q.push(child.get());
        }
    }
    return "";
}

std::string SuffixTree::_check_node_integrity(SuffixNode * node) {
    int64_t children_count = 0;
    for (const auto & [token, child] : node->children) {
        CHECK_OR_RETURN(child->parent == node);
        children_count++;
    }
    CHECK_OR_RETURN(children_count <= node->count);
    if (node == _root.get()) {
        CHECK_OR_RETURN(node->count >= 0);
        CHECK_OR_RETURN(node->parent == nullptr);
        CHECK_OR_RETURN(node->length == 0);
        CHECK_OR_RETURN(node->endpoints.empty());
        CHECK_OR_RETURN(node->ref_idx == -1);
    } else {
        CHECK_OR_RETURN(node->length > 0);
        CHECK_OR_RETURN(node->count > 0);
        for (const auto & [token, child] : node->children) {
            CHECK_OR_RETURN(child->count < node->count);
        }
        CHECK_OR_RETURN(_seqs.contains(node->ref_seq));
        CHECK_OR_RETURN(node->ref_idx >= 0);
        CHECK_OR_RETURN(node->ref_idx + node->length <= static_cast<int>(_seqs[node->ref_seq].size()));
        CHECK_OR_RETURN(node->token == _seqs[node->ref_seq][node->ref_idx]);
        CHECK_OR_RETURN(node->parent->children.contains(node->token));
        CHECK_OR_RETURN(node->parent->children[node->token].get() == node);
        for (auto [ep_seq_id, end_idx] : node->endpoints) {
            CHECK_OR_RETURN(_seqs.contains(ep_seq_id));
            CHECK_OR_RETURN(end_idx > 0 && end_idx <= static_cast<int>(_seqs[ep_seq_id].size()));
            SuffixNode * n = node;
            int idx = end_idx;
            do {
                CHECK_OR_RETURN(n->length <= idx);
                idx -= n->length;
                for (int i = 0; i < n->length; ++i) {
                    int tok = _seqs[n->ref_seq][n->ref_idx + i];
                    CHECK_OR_RETURN(_seqs[ep_seq_id][idx + i] == tok);
                }
                n = n->parent;
            } while (n != nullptr);
        }
    }
    if (!node->head_child && !node->tail_child) {
        CHECK_OR_RETURN(node->children.empty());
    } else {
        CHECK_OR_RETURN(node->head_child && node->tail_child);
        CHECK_OR_RETURN(node->head_child->prev_sibling == nullptr);
        CHECK_OR_RETURN(node->tail_child->next_sibling == nullptr);
        int count = 0;
        SuffixNode * child = node->head_child;
        SuffixNode * prev_child = nullptr;
        while (child != nullptr) {
            count++;
            CHECK_OR_RETURN(node->children.contains(child->token));
            CHECK_OR_RETURN(child->group != nullptr);
            if (prev_child) {
                CHECK_OR_RETURN(child->count <= prev_child->count);
                CHECK_OR_RETURN(child->prev_sibling == prev_child);
                CHECK_OR_RETURN(prev_child->next_sibling == child);
                if (child->count == prev_child->count) {
                    CHECK_OR_RETURN(child->group == prev_child->group);
                } else {
                    CHECK_OR_RETURN(child->group != prev_child->group);
                    CHECK_OR_RETURN(child->group->head == child);
                    CHECK_OR_RETURN(child->group->prev == prev_child->group.get());
                    CHECK_OR_RETURN(prev_child->group->next == child->group.get());
                }
            } else {
                CHECK_OR_RETURN(child == node->head_child);
            }
            prev_child = child;
            child = child->next_sibling;
        }
        CHECK_OR_RETURN(prev_child == node->tail_child);
        CHECK_OR_RETURN(count == static_cast<int>(node->children.size()));
    }
    return "";
}

size_t SuffixTree::estimate_memory() const {
    size_t total = sizeof(*this);
    std::vector<SuffixNode *> stack;
    stack.push_back(_root.get());
    while (!stack.empty()) {
        SuffixNode * node = stack.back();
        stack.pop_back();
        total += node->memory_usage();
        if (node->head_child) {
            SuffixGroup * group = node->head_child->group.get();
            while (group) {
                total += sizeof(*group);
                group = group->next;
            }
        }
        for (const auto & [token, child] : node->children) {
            stack.push_back(child.get());
        }
    }
    for (const auto & [seq_id, seq] : _seqs) {
        total += sizeof(seq) + seq.capacity() * sizeof(int32_t);
    }
    for (const auto & [seq_id, nodes] : _active_nodes) {
        total += sizeof(nodes) + nodes.size() * sizeof(SuffixNode *);
    }
    return total;
}
