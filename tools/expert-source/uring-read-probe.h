#pragma once

#include <liburing.h>

#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

struct uring_read_result {
    int read = 0;
    int cancel = 0;
    bool cancel_requested = false;
    bool completed_before_cancel = false;
    size_t expected_bytes = 0;
};

class uring_read_probe {
    io_uring ring {};

public:
    uring_read_probe() {
        const int rc = io_uring_queue_init(4, &ring, 0);
        if (rc < 0) {
            throw std::runtime_error("io_uring_queue_init: " + std::to_string(rc));
        }
    }

    ~uring_read_probe() { io_uring_queue_exit(&ring); }
    uring_read_probe(const uring_read_probe &) = delete;
    uring_read_probe & operator=(const uring_read_probe &) = delete;

    uring_read_result run(int fd, size_t offset, void * dst, size_t size,
            bool (*request_resize)(void *) noexcept, void * user) {
        if (size > INT_MAX) {
            throw std::invalid_argument("io_uring expert read exceeds CQE byte count");
        }
        constexpr uint64_t read_id = 1, cancel_id = 2;
        auto * read = io_uring_get_sqe(&ring);
        io_uring_prep_read(read, fd, dst, unsigned(size), offset);
        io_uring_sqe_set_flags(read, IOSQE_ASYNC);
        io_uring_sqe_set_data64(read, read_id);
        // A private ring owns at most two requests, each submitted separately.
        int rc = io_uring_submit(&ring);
        if (rc != 1) {
            throw std::runtime_error("io_uring read submission: " + std::to_string(rc));
        }

        uring_read_result result;
        result.expected_bytes = size;
        result.cancel_requested = !request_resize(user);
        bool cancel_submitted = false;
        if (result.cancel_requested) {
            io_uring_cqe * ready = nullptr;
            result.completed_before_cancel = io_uring_peek_cqe(&ring, &ready) == 0;
            auto * cancel = io_uring_get_sqe(&ring);
            io_uring_prep_cancel64(cancel, read_id, 0);
            io_uring_sqe_set_data64(cancel, cancel_id);
            rc = io_uring_submit(&ring);
            cancel_submitted = rc == 1;
        }

        // The staging lease must outlive the read CQE, including when the
        // cancel SQE failed to submit or its CQE arrives first.
        for (int remaining = cancel_submitted ? 2 : 1; remaining > 0; --remaining) {
            io_uring_cqe * cqe = nullptr;
            int wait_rc;
            do {
                wait_rc = io_uring_wait_cqe(&ring, &cqe);
            } while (wait_rc == -EINTR);
            if (wait_rc < 0) {
                // The diagnostic cannot safely unwind with unknown buffer ownership.
                std::fprintf(stderr, "io_uring wait failed: %d\n", wait_rc);
                std::abort();
            }
            if (io_uring_cqe_get_data64(cqe) == read_id) {
                result.read = cqe->res;
            } else if (io_uring_cqe_get_data64(cqe) == cancel_id) {
                result.cancel = cqe->res;
            } else {
                std::abort();
            }
            io_uring_cqe_seen(&ring, cqe);
        }
        if (result.cancel_requested && !cancel_submitted) {
            throw std::runtime_error("io_uring cancel submission: " + std::to_string(rc));
        }
        if (result.cancel_requested && result.cancel != 0 && result.cancel != -ENOENT && result.cancel != -EALREADY) {
            throw std::runtime_error("io_uring cancel completion: " + std::to_string(result.cancel));
        }
        if (result.read != int(size) && !(result.cancel_requested && result.read == -ECANCELED)) {
            throw std::runtime_error("io_uring read completion: " + std::to_string(result.read));
        }
        return result;
    }
};
