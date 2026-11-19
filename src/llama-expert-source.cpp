#include "llama-expert-source.h"
#include "llama-expert-profile.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <sys/stat.h>

#if defined(__APPLE__)
#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <sys/uio.h>
#if defined(__ARM_FEATURE_CRC32)
#include <arm_acle.h>
#endif
#endif

#if defined(_WIN32)
#include <io.h>
#define expert_fstat _fstat64
#define expert_stat _stat64
#define expert_fileno _fileno
#define expert_fseek _fseeki64
#else
#include <unistd.h>
#define expert_fstat fstat
#define expert_stat stat
#define expert_fileno fileno
#define expert_fseek fseeko
#endif

namespace {

using file_ptr = std::unique_ptr<FILE, decltype(&std::fclose)>;

struct file_state {
    struct expert_stat value;

    explicit file_state(FILE * file) {
        if (expert_fstat(expert_fileno(file), &value) != 0 || value.st_size < 0) {
            throw std::runtime_error("cannot stat expert source");
        }
    }

    bool operator==(const file_state & other) const {
        const auto & b = other.value;
        bool same = value.st_dev == b.st_dev && value.st_ino == b.st_ino &&
                    value.st_size == b.st_size && value.st_mtime == b.st_mtime && value.st_ctime == b.st_ctime;
#if defined(__APPLE__)
        same = same && value.st_mtimespec.tv_nsec == b.st_mtimespec.tv_nsec &&
                       value.st_ctimespec.tv_nsec == b.st_ctimespec.tv_nsec;
#elif defined(__linux__)
        same = same && value.st_mtim.tv_nsec == b.st_mtim.tv_nsec && value.st_ctim.tv_nsec == b.st_ctim.tv_nsec;
#endif
        return same;
    }
};

uint16_t split_key(const gguf_context * meta, const char * name) {
    const int64_t id = gguf_find_key(meta, name);
    if (id < 0 || gguf_get_kv_type(meta, id) != GGUF_TYPE_UINT16) {
        throw std::runtime_error(std::string("missing or invalid GGUF key: ") + name);
    }
    return gguf_get_val_u16(meta, id);
}

std::string model_architecture(const gguf_context * meta) {
    const int64_t id = gguf_find_key(meta, "general.architecture");
    if (id < 0 || gguf_get_kv_type(meta, id) != GGUF_TYPE_STRING) {
        throw std::runtime_error("missing or invalid general.architecture");
    }
    return gguf_get_val_str(meta, id);
}

} // namespace

bool llama_expert_tensor_info::can_read_experts() const {
    // GGUF tensors are contiguous. Preserve complete quantization blocks along
    // dimension 0; never infer an expert axis from a flattened/fused 2D tensor.
    return ne[0] > 0 && ne[1] > 0 && ne[2] > 1 && ne[3] == 1 &&
           ne[0] % ggml_blck_size(type) == 0 && bytes % size_t(ne[2]) == 0;
}

size_t llama_expert_tensor_info::expert_bytes() const {
    if (!can_read_experts()) {
        throw std::runtime_error("unsupported expert layout: " + name);
    }
    return bytes / size_t(ne[2]);
}

struct llama_expert_source::impl {
    struct shard {
        file_ptr file;
        file_state state;

        explicit shard(const std::string & path) : file(ggml_fopen(path.c_str(), "rb"), &std::fclose),
            state(require_file(path)) {
            // No persistent userspace read-ahead buffer. Kernel page caching is
            // still enabled; this is not a process-footprint or direct-I/O cap.
            if (std::setvbuf(file.get(), nullptr, _IONBF, 0) != 0) {
                throw std::runtime_error("cannot configure expert source: " + path);
            }
        }

        FILE * require_file(const std::string & path) const {
            if (!file) {
                throw std::runtime_error("cannot open expert source '" + path + "': " + std::strerror(errno));
            }
            return file.get();
        }

        void check() const {
            if (!(state == file_state(file.get()))) {
                throw std::runtime_error("expert source changed; reopen the model");
            }
        }
    };

    std::vector<shard> shards;
    std::vector<llama_expert_tensor_info> tensors;
    std::map<std::string, size_t> by_name;
    std::string architecture;
    std::mutex mutex;
    bool failed = false;
    struct expert_pack {
        shard data;
        std::array<packed_group, 48> groups {};
        std::vector<packed_slice> slices;
        size_t max_stride = 0;
    };
    std::unique_ptr<expert_pack> pack;

    explicit impl(const std::vector<std::string> & paths) {
        if (paths.empty() || paths.size() > std::numeric_limits<uint16_t>::max()) {
            throw std::invalid_argument("expected a nonempty list of GGUF shards");
        }
        shards.reserve(paths.size());
        int64_t expected_tensors = -1;
        for (size_t i = 0; i < paths.size(); ++i) {
            shards.emplace_back(paths[i]);
            auto & s = shards.back();
            // Parse metadata from the same open handle used for reads, avoiding
            // a path-replacement race between indexing and opening the weights.
            gguf_context_ptr meta(gguf_init_from_file_ptr(s.file.get(), {true, nullptr}));
            if (!meta) {
                throw std::runtime_error("invalid GGUF: " + paths[i]);
            }
            if (i == 0) {
                architecture = model_architecture(meta.get());
            } else if (gguf_find_key(meta.get(), "general.architecture") >= 0 &&
                       model_architecture(meta.get()) != architecture) {
                throw std::runtime_error("GGUF shard architecture mismatch");
            }
            const bool split = gguf_find_key(meta.get(), "split.count") >= 0;
            if (split || paths.size() > 1) {
                if (split_key(meta.get(), "split.count") != paths.size() || split_key(meta.get(), "split.no") != i) {
                    throw std::runtime_error("incomplete or unordered GGUF shard set");
                }
                const int64_t key = gguf_find_key(meta.get(), "split.tensors.count");
                if (key < 0 || gguf_get_kv_type(meta.get(), key) != GGUF_TYPE_INT32) {
                    throw std::runtime_error("missing or invalid split.tensors.count");
                }
                const int64_t count = gguf_get_val_i32(meta.get(), key);
                if (count < 0 || (i > 0 && count != expected_tensors)) {
                    throw std::runtime_error("GGUF shard tensor count mismatch");
                }
                expected_tensors = count;
            }
            const uint64_t file_size = uint64_t(s.state.value.st_size);
            const size_t base = gguf_get_data_offset(meta.get());
            for (int64_t j = 0; j < gguf_get_n_tensors(meta.get()); ++j) {
                const size_t relative = gguf_get_tensor_offset(meta.get(), j);
                const size_t bytes = gguf_get_tensor_size(meta.get(), j);
                if (base > file_size || relative > file_size - base || bytes > file_size - base - relative ||
                    relative > std::numeric_limits<size_t>::max() - base) {
                    throw std::runtime_error("GGUF tensor lies outside its shard: " + paths[i]);
                }
                llama_expert_tensor_info info {gguf_get_tensor_name(meta.get(), j), i, base + relative, bytes,
                                               gguf_get_tensor_type(meta.get(), j), {}};
                std::copy_n(gguf_get_tensor_ne(meta.get(), j), GGML_MAX_DIMS, info.ne.begin());
                if (!by_name.emplace(info.name, tensors.size()).second) {
                    throw std::runtime_error("duplicate GGUF tensor: " + info.name);
                }
                tensors.push_back(std::move(info));
            }
            s.check();
        }
        if (expected_tensors >= 0 && uint64_t(expected_tensors) != tensors.size()) {
            throw std::runtime_error("GGUF shard set does not contain the declared tensor count");
        }
    }
};

llama_expert_source::llama_expert_source(const std::vector<std::string> & paths) : pimpl(new impl(paths)) {}
llama_expert_source::~llama_expert_source() = default;

const std::string & llama_expert_source::architecture() const { return pimpl->architecture; }
const std::vector<llama_expert_tensor_info> & llama_expert_source::tensors() const { return pimpl->tensors; }

bool llama_expert_source::open_expert_pack(const std::string & path) {
#if defined(__APPLE__) && defined(__ARM_FEATURE_CRC32)
    if (pimpl->architecture != "qwen4exp" || pimpl->pack) {
        throw std::invalid_argument("expert pack requires an unconfigured Qwen4Exp source");
    }
    struct expert_stat st;
    if (::stat(path.c_str(), &st) != 0 && errno == ENOENT) { return false; }
    auto pack = std::unique_ptr<impl::expert_pack>(new impl::expert_pack{impl::shard(path), {}, {}, 0});
    FILE * file = pack->data.file.get();
    auto read = [&](void * dst, size_t bytes) {
        if (std::fread(dst, 1, bytes, file) != bytes) { throw std::runtime_error("truncated expert pack header"); }
    };
    char magic[8];
    uint64_t count;
    read(magic, sizeof(magic));
    read(&count, sizeof(count));
    if (std::memcmp(magic, "ATEXPK1", sizeof(magic)) || count != pimpl->shards.size()) {
        throw std::runtime_error("invalid expert pack header");
    }
    size_t offset = sizeof(magic) + sizeof(count);
    for (const auto & shard : pimpl->shards) {
        const auto & state = shard.state.value;
        const auto nanos = [](const timespec & t) { return uint64_t(t.tv_sec) * 1000000000 + uint64_t(t.tv_nsec); };
        const std::array<uint64_t, 5> expected {uint64_t(state.st_dev), uint64_t(state.st_ino), uint64_t(state.st_size),
            nanos(state.st_mtimespec), nanos(state.st_ctimespec)};
        std::array<uint64_t, 5> actual;
        read(actual.data(), sizeof(actual));
        if (actual != expected) { throw std::runtime_error("expert pack belongs to different GGUF shards"); }
        offset += sizeof(actual);
    }
    const size_t file_bytes = pack->data.state.value.st_size;
    pack->slices.resize(pimpl->tensors.size());
    for (size_t layer = 0; layer < pack->groups.size(); ++layer) {
        auto & group = pack->groups[layer];
        group.file_offset = offset;
        for (const char * kind : {"gate", "up", "down"}) {
            const auto & info = tensor("blk." + std::to_string(layer) + ".ffn_" + kind + "_exps.weight");
            if (!info.can_read_experts() || info.ne[2] > 1024 ||
                    (group.experts && group.experts != size_t(info.ne[2]))) {
                throw std::runtime_error("unsupported expert pack tensor layout");
            }
            group.experts = info.ne[2];
            pack->slices[&info - pimpl->tensors.data()] = {&group, group.stride};
            const size_t bytes = info.expert_bytes();
            if (bytes > size_t(INT32_MAX) - sizeof(uint32_t) - group.stride) {
                throw std::runtime_error("expert pack record is too large");
            }
            group.stride += bytes;
        }
        group.stride += sizeof(uint32_t);
        if (offset > file_bytes || group.stride > (file_bytes - offset) / group.experts) {
            throw std::runtime_error("truncated expert pack payload");
        }
        offset += group.stride * group.experts;
        pack->max_stride = std::max(pack->max_stride, group.stride);
    }
    if (offset != file_bytes) { throw std::runtime_error("unexpected expert pack size"); }
    pack->data.check();
    pimpl->pack = std::move(pack);
    return true;
#else
    (void) path;
    return false;
#endif
}

size_t llama_expert_source::packed_stride() const { return pimpl->pack ? pimpl->pack->max_stride : 0; }

llama_expert_source::packed_slice llama_expert_source::packed_location(size_t tensor_index) const {
    return pimpl->pack ? pimpl->pack->slices[tensor_index] : packed_slice{};
}

void llama_expert_source::read_packed_expert(const packed_group & group, int64_t expert, void * dst) {
#if defined(__APPLE__) && defined(__ARM_FEATURE_CRC32)
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (pimpl->failed) { throw std::runtime_error("invalid expert source after earlier read failure"); }
    auto & pack = pimpl->pack->data;
    try {
        for (const auto & shard : pimpl->shards) { shard.check(); }
        pack.check();
        const off_t offset = group.file_offset + size_t(expert) * group.stride;
        ssize_t count;
        do {
            count = ::pread(expert_fileno(pack.file.get()), dst, group.stride, offset);
        } while (count < 0 && errno == EINTR);
        if (count != ssize_t(group.stride)) { throw std::runtime_error("incomplete expert pack read"); }
        pack.check();
        for (const auto & shard : pimpl->shards) { shard.check(); }

        const auto * bytes = static_cast<const uint8_t *>(dst);
        const size_t payload = group.stride - sizeof(uint32_t);
        uint32_t crc = UINT32_MAX;
        size_t i = 0;
        for (; payload - i >= sizeof(uint64_t); i += sizeof(uint64_t)) {
            uint64_t word;
            std::memcpy(&word, bytes + i, sizeof(word));
            crc = __crc32d(crc, word);
        }
        for (; i < payload; ++i) { crc = __crc32b(crc, bytes[i]); }
        uint32_t expected;
        std::memcpy(&expected, bytes + payload, sizeof(expected));
        if (~crc != expected) { throw std::runtime_error("expert pack checksum mismatch"); }
    } catch (...) {
        pimpl->failed = true;
        throw;
    }
#else
    (void) group; (void) expert; (void) dst;
    throw std::runtime_error("expert pack requires Apple silicon");
#endif
}

bool llama_expert_source::advise_packed_expert(const packed_group & group, int64_t expert) {
#if defined(__APPLE__)
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (pimpl->failed) { return false; }
    struct radvisory advice {off_t(group.file_offset + size_t(expert) * group.stride), int(group.stride)};
    return fcntl(expert_fileno(pimpl->pack->data.file.get()), F_RDADVISE, &advice) != -1;
#else
    (void) group; (void) expert;
    return false;
#endif
}

const llama_expert_tensor_info & llama_expert_source::tensor(const std::string & name) const {
    const auto it = pimpl->by_name.find(name);
    if (it == pimpl->by_name.end()) {
        throw std::invalid_argument("unknown expert tensor: " + name);
    }
    return pimpl->tensors[it->second];
}

void llama_expert_source::read_experts(const read_request * requests, size_t count, bool speculative,
        bool concurrent_demand, bool adjacent_ranges) {
    if (count == 1 && !speculative && !concurrent_demand) {
        const auto & req = requests[0];
        read_expert(req.tensor->name, req.expert, req.destination, req.tensor->expert_bytes());
        return;
    }
#if defined(__APPLE__)
    if (!count || count > 16) { throw std::invalid_argument("invalid expert read batch size"); }
    struct job {
        int fd;
        off_t offset;
        size_t bytes;
        void * destination;
        bool complete = false;
    };
    struct batch {
        std::array<job, 16> jobs;
        size_t count;
        size_t width;
        std::array<size_t, 17> ranges {};
        size_t range_count = 0;
    } work {{}, count, concurrent_demand ? 1 : std::min(count, size_t(4)), {}, 0};
    std::unique_lock<std::mutex> lock(pimpl->mutex);
    if (pimpl->failed) { throw std::runtime_error("expert source is invalid after an earlier read failure"); }
    for (size_t i = 0; i < count; ++i) {
        const auto & req = requests[i];
        const auto & info = *req.tensor;
        const size_t bytes = info.expert_bytes();
        if (!req.destination || req.expert < 0 || req.expert >= info.ne[2]) {
            throw std::invalid_argument("invalid batched expert read");
        }
        work.jobs[i] = {expert_fileno(pimpl->shards[info.shard].file.get()),
                        off_t(info.offset + size_t(req.expert) * bytes), bytes, req.destination};
    }
    // Offsets, handles and file identity are immutable. Positioned speculative
    // reads can overlap demand without keeping the source mutex through I/O.
    if (speculative || concurrent_demand) { lock.unlock(); }
    try {
        for (const auto & shard : pimpl->shards) { shard.check(); }
        // A null queue selects the calling thread's QoS (DISPATCH_APPLY_AUTO).
        auto read = [](void * opaque, size_t index) {
            auto & state = *static_cast<batch *>(opaque);
            for (; index < state.count; index += state.width) {
                auto & job = state.jobs[index];
                auto * dst = static_cast<char *>(job.destination);
                size_t read_bytes = 0;
                while (read_bytes < job.bytes) {
                    const auto n = ::pread(job.fd, dst + read_bytes, job.bytes - read_bytes, job.offset + read_bytes);
                    if (n < 0 && errno == EINTR) { continue; }
                    if (n <= 0) { break; }
                    read_bytes += n;
                }
                job.complete = read_bytes == job.bytes;
            }
        };
        if (adjacent_ranges && !speculative && !concurrent_demand) {
            if (__builtin_available(macOS 11.0, *)) {
                size_t groups = 0;
                for (size_t i = 0; i < work.count; ++i) {
                    if (!i || requests[i].tensor != requests[i - 1].tensor ||
                            work.jobs[i].fd != work.jobs[i - 1].fd ||
                            work.jobs[i].offset != work.jobs[i - 1].offset + off_t(work.jobs[i - 1].bytes)) {
                        work.ranges[groups++] = i;
                    }
                }
                work.ranges[groups] = work.count;
                // Read exact consecutive slices straight into each admitted
                // destination. A short read advances the vectors and offset.
                work.range_count = groups;
                work.width = std::min(groups, size_t(4));
                auto read_range = [](void * opaque, size_t first_group) {
                    auto & batch = *static_cast<struct batch *>(opaque);
                    for (size_t group = first_group; group < batch.range_count; group += batch.width) {
                        const size_t first = batch.ranges[group], end = batch.ranges[group + 1];
                        std::array<iovec, 16> vectors {};
                        for (size_t i = first; i < end; ++i) {
                            vectors[i] = {batch.jobs[i].destination, batch.jobs[i].bytes};
                        }
                        auto offset = batch.jobs[first].offset;
                        for (size_t i = first; i < end;) {
                            const auto bytes = ::preadv(batch.jobs[first].fd, vectors.data() + i, int(end - i), offset);
                            if (bytes < 0 && errno == EINTR) { continue; }
                            if (bytes <= 0) { return; }
                            offset += bytes;
                            size_t done = size_t(bytes);
                            while (i < end && done >= vectors[i].iov_len) {
                                done -= vectors[i].iov_len;
                                ++i;
                            }
                            if (i < end && done) {
                                vectors[i].iov_base = static_cast<char *>(vectors[i].iov_base) + done;
                                vectors[i].iov_len -= done;
                            }
                        }
                        for (size_t i = first; i < end; ++i) { batch.jobs[i].complete = true; }
                    }
                };
                dispatch_apply_f(work.width, nullptr, &work, read_range);
            } else {
                dispatch_apply_f(work.width, nullptr, &work, read);
            }
        } else if (concurrent_demand) {
            read(&work, 0);
        } else {
            dispatch_apply_f(work.width, nullptr, &work, read);
        }
        for (const auto & shard : pimpl->shards) { shard.check(); }
        if (!std::all_of(work.jobs.begin(), work.jobs.begin() + count, [](const job & j) { return j.complete; })) {
            throw std::runtime_error("incomplete batched expert read");
        }
    } catch (...) {
        if (!speculative) {
            if (!lock.owns_lock()) { lock.lock(); }
            pimpl->failed = true;
        }
        throw;
    }
#else
    (void) adjacent_ranges;
    if (speculative || concurrent_demand) { throw std::invalid_argument("speculative batched expert reads require Metal"); }
    for (size_t i = 0; i < count; ++i) {
        const auto & req = requests[i];
        read_expert(req.tensor->name, req.expert, req.destination, req.tensor->expert_bytes());
    }
#endif
}

bool llama_expert_source::advise_expert(const std::string & name, int64_t expert) {
#if defined(__APPLE__)
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    const auto found = pimpl->by_name.find(name);
    if (pimpl->failed || found == pimpl->by_name.end()) { return false; }
    const auto & info = pimpl->tensors[found->second];
    if (!info.can_read_experts() || expert < 0 || expert >= info.ne[2]) { return false; }
    const size_t bytes = info.expert_bytes();
    if (bytes > size_t(std::numeric_limits<int>::max())) { return false; }
    // The immutable index bounds this range. Only the subsequent demand read
    // validates the file and publishes bytes; a hint failure is not a read error.
    struct radvisory advice {off_t(info.offset + size_t(expert) * bytes), int(bytes)};
    return fcntl(expert_fileno(pimpl->shards[info.shard].file.get()), F_RDADVISE, &advice) != -1;
#else
    (void) name; (void) expert;
    return false;
#endif
}

void llama_expert_source::read_row(const std::string & name, int64_t row, void * dst, size_t size) {
    const auto & info = tensor(name);
    if (!dst || info.ne[2] != 1 || info.ne[3] != 1 || info.ne[0] % ggml_blck_size(info.type) ||
        row < 0 || row >= info.ne[1] || size != ggml_row_size(info.type, info.ne[0])) {
        throw std::invalid_argument("invalid streamed row: " + name);
    }
    const size_t offset = info.offset + size_t(row) * size;
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (pimpl->failed) { throw std::runtime_error("row source is invalid after an earlier read failure"); }
    auto & s = pimpl->shards[info.shard];
    try {
        s.check();
        if (expert_fseek(s.file.get(), offset, SEEK_SET) != 0 || std::fread(dst, 1, size, s.file.get()) != size) {
            throw std::runtime_error("incomplete streamed row: " + name);
        }
        s.check();
    } catch (...) {
        pimpl->failed = true;
        throw;
    }
}

bool llama_expert_source::read_expert(const std::string & name, int64_t expert, void * dst, size_t size,
                                    llama_expert_profile_data * profile, bool (*mid_read)(void *) noexcept, void * user,
                                    llama_expert_read_probe submitted_read) {
    llama_expert_timer timer(profile ? &profile->source_setup_ns : nullptr);
    const auto & info = tensor(name);
    const size_t stride = info.expert_bytes();
    if (expert < 0 || expert >= info.ne[2] || dst == nullptr || size != stride) {
        throw std::invalid_argument("invalid expert ID or destination size: " + name);
    }
    // Index construction bounded the complete tensor inside the file. These
    // operations cannot overflow for a valid expert within that tensor.
    const size_t offset = info.offset + size_t(expert) * stride;
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (pimpl->failed) {
        throw std::runtime_error("expert source is invalid after an earlier read failure");
    }
    auto & s = pimpl->shards[info.shard];
    try {
        timer.next(profile ? &profile->source_validate_ns : nullptr);
        s.check();
        if (submitted_read) {
            timer.next(profile ? &profile->source_transfer_ns : nullptr);
            const bool complete = submitted_read(user, expert_fileno(s.file.get()), offset, dst, size);
            timer.next(profile ? &profile->source_validate_ns : nullptr);
            s.check();
            return complete;
        }
        timer.next(profile ? &profile->source_seek_ns : nullptr);
        if (expert_fseek(s.file.get(), offset, SEEK_SET) != 0) {
            throw std::runtime_error("cannot seek expert source");
        }
        timer.next(profile ? &profile->source_transfer_ns : nullptr);
        const size_t first = mid_read && size > 1 ? size / 2 : size;
        if (std::fread(dst, 1, first, s.file.get()) != first) {
            throw std::runtime_error("incomplete expert read: " + name);
        }
        const bool complete = first == size || mid_read(user);
        if (complete && first < size) {
            if (std::fread(static_cast<char *>(dst) + first, 1, size - first, s.file.get()) != size - first) {
                throw std::runtime_error("incomplete expert read: " + name);
            }
        }
        timer.next(profile ? &profile->source_validate_ns : nullptr);
        s.check();
        return complete;
    } catch (...) {
        pimpl->failed = true;
        throw;
    }
}
