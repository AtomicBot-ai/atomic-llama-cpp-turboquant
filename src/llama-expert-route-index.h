#pragma once

#include "ggml.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>

// One read of each strided ID; stable token/slot lists within ascending experts.
// No tensor pointers or routing state survive the next build. Oversized shapes
// use the executor's generic scan instead of growing this admitted workspace.
struct llama_expert_route_index {
    static constexpr size_t max_experts = 1024;
    static constexpr size_t max_tokens = 256;
    static constexpr size_t max_slots = 32;
    static constexpr size_t max_routes = max_tokens * max_slots;
    static constexpr uint16_t end = UINT16_MAX;
    static_assert(max_routes < end, "route links must fit below the sentinel");
    std::array<uint16_t, max_experts> first, last, active;
    std::array<uint16_t, max_routes> next;
    uint16_t count = 0;

    static bool fits(const ggml_tensor * ids, int64_t experts) {
        return ids->type == GGML_TYPE_I32 && experts > 0 && experts <= int64_t(max_experts) &&
            ids->ne[0] > 0 && ids->ne[0] <= int64_t(max_slots) &&
            ids->ne[1] >= 0 && ids->ne[1] <= int64_t(max_tokens);
    }

    void build(const ggml_tensor * ids, int64_t experts) {
        if (!fits(ids, experts)) { throw std::invalid_argument("route index shape exceeds its fixed capacity"); }
        std::fill_n(first.begin(), size_t(experts), end);
        count = 0;
        uint16_t at = 0;
        for (int64_t token = 0; token < ids->ne[1]; ++token) {
            for (int64_t slot = 0; slot < ids->ne[0]; ++slot, ++at) {
                int32_t expert;
                std::memcpy(&expert, static_cast<const char *>(ids->data) + token * ids->nb[1] + slot * ids->nb[0], sizeof(expert));
                if (expert < 0 || expert >= experts) { throw std::runtime_error("router produced an invalid expert ID"); }
                if (first[expert] == end) { first[expert] = at; }
                else { next[last[expert]] = at; }
                last[expert] = at;
                next[at] = end;
            }
        }
        for (int64_t expert = 0; expert < experts; ++expert) {
            if (first[expert] != end) { active[count++] = uint16_t(expert); }
        }
    }

    int64_t after(int64_t expert, int64_t limit) const {
        const auto found = std::upper_bound(active.begin(), active.begin() + count, expert);
        return found == active.begin() + count ? limit : *found;
    }
};
