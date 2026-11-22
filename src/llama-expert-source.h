#pragma once

#include "ggml.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Original, packed GGUF bytes. A tensor name is unique only within this source.
struct llama_expert_tensor_info {
    std::string name;
    size_t shard;
    size_t offset;
    size_t bytes;
    ggml_type type;
    std::array<int64_t, GGML_MAX_DIMS> ne;

    // Supports contiguous [input, output, expert] storage with >1 experts only. This is a storage
    // capability, not a claim that the model/backend supports streaming inference.
    bool can_read_experts() const;
    size_t expert_bytes() const;
};

// Metadata-only index with owned, pinned file handles. Explicit shards must be
// supplied in split order. No weights are mmap'd or allocated by this source.
// Files must remain immutable while open; stat checks detect ordinary changes,
// but are not a content hash or protection against concurrent malicious writes.
struct llama_expert_profile_data;
// Diagnostic replacement for one read. Borrow the pinned descriptor and
// destination only until return, including after any cancellation request.
using llama_expert_read_probe = bool (*)(void * user, int fd, size_t offset, void * dst, size_t size);

enum class llama_expert_read_layout { separate, adjacent, parallel_slices };

class llama_expert_source {
public:
    struct packed_group {
        size_t file_offset;
        size_t stride; // three original slices followed by their CRC32
        size_t experts;
    };
    struct packed_slice {
        const packed_group * group = nullptr;
        size_t offset = 0;
    };

    explicit llama_expert_source(const std::vector<std::string> & paths);
    ~llama_expert_source();

    const std::string & architecture() const;
    const std::vector<llama_expert_tensor_info> & tensors() const;
    const llama_expert_tensor_info & tensor(const std::string & name) const;
    // Configure before sharing the source. Missing files leave GGUF reads active.
    bool open_expert_pack(const std::string & path);
    size_t packed_stride() const;
    packed_slice packed_location(size_t tensor_index) const;
    void read_packed_expert(const packed_group & group, int64_t expert, void * dst);
    bool advise_packed_expert(const packed_group & group, int64_t expert);
    // Exact demand reads into independently admitted, unpublished host buffers.
    // The calling thread retains ownership of every destination until return.
    struct read_request {
        const llama_expert_tensor_info * tensor = nullptr;
        int64_t expert = 0;
        void * destination = nullptr;
    };
    void read_experts(const read_request * requests, size_t count, bool speculative = false,
                    bool concurrent_demand = false, llama_expert_read_layout layout = llama_expert_read_layout::separate);
#if defined(__APPLE__)
    // Exact contiguous expert slices into one already admitted Metal or staging range.
    void read_expert_range(const std::string & name, int64_t first, int64_t end, void * dst, size_t bytes);
#endif
    // Best-effort macOS file-cache hint for one known demand expert. No user
    // buffer, published weight, offset change or validation bypass. False means
    // unsupported, invalid or declined; demand must still use read_expert.
    bool advise_expert(const std::string & name, int64_t expert);
    // Read one complete packed row from a contiguous matrix into admitted storage.
    void read_row(const std::string & name, int64_t row, void * dst, size_t size);

    // Synchronous, serialized reads into caller-owned storage of exactly the
    // expert's size. The caller must reserve that storage before allocation.
    // Publish only after success. On I/O/change errors the destination is invalid
    // and this source becomes unusable; no partial expert is a successful read.
    // Argument errors do not poison the source. No repacking or dequantization.
    // Optional counters require exclusive caller ownership for the whole call.
    // A false midpoint or replacement-read result cancels without poisoning
    // the source. The destination must be discarded.
    bool read_expert(const std::string & name, int64_t expert, void * dst, size_t size,
                    llama_expert_profile_data * profile = nullptr,
                    bool (*mid_read)(void *) noexcept = nullptr, void * user = nullptr,
                    llama_expert_read_probe submitted_read = nullptr);

    llama_expert_source(const llama_expert_source &) = delete;
    llama_expert_source & operator=(const llama_expert_source &) = delete;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
