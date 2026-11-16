#pragma once

#include "server-common.h"
#include "server-prefix-checksum.h"
#include "log.h"
#include "server-task.h"
#include "../../src/llama-atomic.h"
#include <algorithm>
#include <array>
#include <utility>
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <system_error>

// Two process-local complete checkpoints. Only files created inside the
// private directory are eligible for the exact-prefix restore path.
class server_prefix_checkpoints {
    struct checkpoint {
        std::filesystem::path path;
        llama_tokens tokens;
        size_t size = 0;
        uint32_t checksum = 0;
    };

    std::filesystem::path directory;
    std::array<checkpoint, 2> entries;

public:
    explicit server_prefix_checkpoints(const std::string & parent) {
#if defined(__linux__)
        std::string path = (std::filesystem::path(parent) / "atomic-prefix.XXXXXX").string();
        if (!mkdtemp(path.data())) {
            throw std::system_error(errno, std::generic_category(), "create prefix checkpoint directory");
        }
        directory = path;
        entries[0].path = directory / "middle.slot";
        entries[1].path = directory / "tail.slot";
#else
        (void) parent;
        throw std::runtime_error("disk prefix checkpoints require Linux");
#endif
    }

    server_prefix_checkpoints(const server_prefix_checkpoints &) = delete;
    server_prefix_checkpoints & operator=(const server_prefix_checkpoints &) = delete;

    ~server_prefix_checkpoints() {
        std::error_code ignored;
        std::filesystem::remove_all(directory, ignored);
    }

    void clear() {
        for (auto & entry : entries) {
            entry.tokens.clear();
            std::error_code ignored;
            std::filesystem::remove(entry.path, ignored);
        }
    }

    static bool eligible(const server_task & task) {
        return task.type == SERVER_TASK_TYPE_COMPLETION && !task.is_child() && !task.is_stateless() &&
                task.params.cache_prompt && task.params.n_cache_reuse == 0 &&
                !task.tokens.has_mtmd && task.n_tokens() >= 512;
    }

    bool capture(llama_context * ctx, const llama_tokens & tokens, size_t count) {
        constexpr size_t batch = 128;
        const size_t middle = tokens.size() / (2 * batch) * batch;
        const size_t tail = (tokens.size() - 1) / batch * batch;
        if (count != middle && count != tail) {
            return false;
        }

        auto & entry = entries[count == tail ? 1 : 0];
        if (entry.tokens.size() == count && std::equal(entry.tokens.begin(), entry.tokens.end(), tokens.begin())) {
            return false;
        }
        if (!llama_atomic_prefix_ready(ctx, count)) {
            return false;
        }

        llama_tokens prefix(tokens.begin(), tokens.begin() + count);
        const auto pending = directory / "pending.slot";
        try {
            llama_synchronize(ctx);
            const size_t size = llama_state_seq_save_file(ctx, pending.string().c_str(), 0, prefix.data(), prefix.size());
            const auto checksum = server_prefix_checksum::read(pending, size);
            if (!checksum) {
                throw std::runtime_error("prefix checkpoint could not be saved and checksummed");
            }
            std::filesystem::rename(pending, entry.path);
            entry.tokens = std::move(prefix);
            entry.size = size;
            entry.checksum = *checksum;
            return true;
        } catch (const std::exception &) {
            std::error_code ignored;
            std::filesystem::remove(pending, ignored);
            throw;
        }
    }

    // -1 means the chosen checkpoint could not be verified or loaded.
    // Reset the live slot and replay the full prompt in either case.
    int restore(llama_context * ctx, const llama_tokens & tokens, server_tokens & restored) {
        checkpoint * best = nullptr;
        for (auto & entry : entries) {
            const size_t n = entry.tokens.size();
            if (n && n < tokens.size() && (!best || n > best->tokens.size()) &&
                    std::equal(entry.tokens.begin(), entry.tokens.end(), tokens.begin())) {
                best = &entry;
            }
        }
        if (!best) {
            return 0;
        }
        // Allocate the replacement token list before loading model state.
        server_tokens prefix(best->tokens, false);
        const auto checksum = server_prefix_checksum::read(best->path, best->size);
        const bool valid = checksum && *checksum == best->checksum;
        if (!valid) {
            LOG_WRN("%s", "disk prefix checkpoint integrity check failed\n");
        }
        if (!valid || !llama_atomic_prefix_restore(ctx, best->path.string().c_str(), best->tokens.data(), best->tokens.size())) {
            best->tokens.clear();
            std::error_code ignored;
            std::filesystem::remove(best->path, ignored);
            return -1;
        }
        restored = std::move(prefix);
        return int(best->tokens.size());
    }
};
