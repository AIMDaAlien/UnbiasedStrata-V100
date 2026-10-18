#pragma once
#include <cstdint>
#include <cstddef>
#include <cerrno>
#include <poll.h>
#include <sys/socket.h>

namespace strata::amd {
inline constexpr uint32_t magic = 0x414d4431, version = 2;
inline constexpr int H = 2560, FF = 640, TOKENS = 256, ENTRIES = TOKENS * 10;
inline constexpr size_t STAGING = 4 * 1024 * 1024;
enum Op : uint32_t { Hello = 1, Load, Run, Cancel, Shutdown };
struct Message {
    uint32_t tag = magic, protocol = version, op = 0, status = 0;
    uint64_t sequence = 0, epoch = 0, bytes = 0;
    int32_t layer = 0, expert = 0, gu_type = -1, d_type = -1;
    int32_t tokens = 0, entries = 0;
    double transfer_ms = 0, compute_ms = 0;
};
struct Shared {
    alignas(64) uint8_t staging[STAGING];
    alignas(64) float x[TOKENS * H];
    int32_t ids[ENTRIES], token[ENTRIES];
    alignas(64) float result[ENTRIES * H];
};
inline bool send(int fd, const Message& m) {
    ssize_t n;
    do { n = ::send(fd, &m, sizeof(m), MSG_NOSIGNAL); } while (n < 0 && errno == EINTR);
    return n == sizeof(m);
}
inline bool receive(int fd, Message& m, int timeout_ms = 30000) {
    pollfd p{fd, POLLIN, 0};
    int n;
    do { n = ::poll(&p, 1, timeout_ms); } while (n < 0 && errno == EINTR);
    if (n <= 0 || !(p.revents & POLLIN)) return false;
    ssize_t bytes;
    do { bytes = ::recv(fd, &m, sizeof(m), MSG_TRUNC); } while (bytes < 0 && errno == EINTR);
    return bytes == sizeof(m) && m.tag == magic && m.protocol == version;
}
} // namespace strata::amd
