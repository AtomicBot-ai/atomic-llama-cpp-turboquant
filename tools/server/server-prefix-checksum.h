#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>

#if defined(__linux__) && defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <cstring>
#include <nmmintrin.h>
#endif

namespace server_prefix_checksum {

constexpr auto crc32c_table() {
    std::array<uint32_t, 256> table {};
    for (uint32_t i = 0; i < table.size(); ++i) {
        uint32_t crc = i;
        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc >> 1) ^ ((crc & 1) ? 0x82f63b78u : 0u);
        }
        table[i] = crc;
    }
    return table;
}

inline uint32_t update_portable(uint32_t crc, const unsigned char * data, size_t size) {
    static constexpr auto table = crc32c_table();
    for (size_t i = 0; i < size; ++i) {
        crc = table[(crc ^ data[i]) & 0xffu] ^ (crc >> 8);
    }
    return crc;
}

#if defined(__linux__) && defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("sse4.2")))
inline uint32_t update_sse42(uint32_t crc, const unsigned char * data, size_t size) {
    while (size >= sizeof(uint64_t)) {
        uint64_t word;
        std::memcpy(&word, data, sizeof(word));
        crc = uint32_t(_mm_crc32_u64(crc, word));
        data += sizeof(word);
        size -= sizeof(word);
    }
    return update_portable(crc, data, size);
}
#endif

// Includes the header and all tensors. The bounded read buffer is covered by
// the engine runtime envelope. This detects accidental corruption of private,
// process-local files. It does not authenticate files from external writers.
inline std::optional<uint32_t> read(const std::filesystem::path & path, size_t size) {
    std::ifstream file(path, std::ios::binary);
    if (!file || !size) {
        return std::nullopt;
    }
    auto update = update_portable;
#if defined(__linux__) && defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    if (__builtin_cpu_supports("sse4.2")) {
        update = update_sse42;
    }
#endif
    std::array<unsigned char, 16 * 1024> buffer;
    uint32_t crc = UINT32_MAX;
    while (size) {
        const size_t count = size < buffer.size() ? size : buffer.size();
        if (!file.read(reinterpret_cast<char *>(buffer.data()), std::streamsize(count))) {
            return std::nullopt;
        }
        crc = update(crc, buffer.data(), count);
        size -= count;
    }
    if (file.get() != std::char_traits<char>::eof() || !file.eof() || file.bad()) {
        return std::nullopt;
    }
    return ~crc;
}

} // namespace server_prefix_checksum
