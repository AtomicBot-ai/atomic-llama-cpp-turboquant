// Positional reads on independently reopened handles to the indexed GGUF.
// Each lane writes directly into its own part of the admitted staging buffer.
namespace {
class spark_range_reader {
    struct close_handle { void operator()(HANDLE h) const { CloseHandle(h); } };
    using handle = std::unique_ptr<void, close_handle>;
    struct lane { handle file, event; };
    std::array<lane, 8> lanes;
public:
    bool open(HANDLE source) {
        for (auto & lane : lanes) {
            const auto f = ReOpenFile(source, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                    FILE_FLAG_OVERLAPPED);
            if (f == INVALID_HANDLE_VALUE) { return false; }
            lane.file.reset(f);
            lane.event.reset(CreateEventW(nullptr, TRUE, FALSE, nullptr));
            if (!lane.event) { return false; }
        }
        return true;
    }

    struct request { uint64_t offset; void * dst; DWORD bytes; };

    void read_many(const std::array<request, 8> & requests, size_t count) {
        std::array<OVERLAPPED, 8> ops{};
        size_t submitted = 0;
        bool ok = true;
        for (; submitted < count; ++submitted) {
            auto & lane = lanes[submitted];
            auto & op = ops[submitted];
            const auto & r = requests[submitted];
            op.Offset = DWORD(r.offset);
            op.OffsetHigh = DWORD(r.offset >> 32);
            op.hEvent = lane.event.get();
            if (!ResetEvent(op.hEvent) || (!ReadFile(lane.file.get(), r.dst, r.bytes, nullptr, &op) &&
                    GetLastError() != ERROR_IO_PENDING)) {
                ok = false;
                break;
            }
        }
        // Even on failure, every outstanding write must finish before returning
        // the staging buffer to the caller.
        for (size_t i = 0; i < submitted; ++i) {
            DWORD actual = 0;
            if (!GetOverlappedResult(lanes[i].file.get(), &ops[i], &actual, TRUE) ||
                    actual != requests[i].bytes) { ok = false; }
        }
        if (!ok) { throw std::runtime_error("incomplete Spark expert range read"); }
    }

    void read(uint64_t offset, void * dst, size_t bytes) {
        constexpr size_t page = 4096;
        const size_t slice = std::max(size_t(256) << 10, ((bytes + lanes.size()*page - 1) / (lanes.size()*page)) * page);
        std::array<request, 8> requests;
        size_t count = 0;
        for (size_t pos = 0; pos < bytes; pos += slice) {
            requests[count++] = {offset + pos, static_cast<char *>(dst) + pos, DWORD(std::min(slice, bytes - pos))};
        }
        read_many(requests, count);
    }
};
}
