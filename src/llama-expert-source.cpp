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
#include <fcntl.h>
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

const llama_expert_tensor_info & llama_expert_source::tensor(const std::string & name) const {
    const auto it = pimpl->by_name.find(name);
    if (it == pimpl->by_name.end()) {
        throw std::invalid_argument("unknown expert tensor: " + name);
    }
    return pimpl->tensors[it->second];
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
