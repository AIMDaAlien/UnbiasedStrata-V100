#include "strata/core/amd_experts.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

namespace strata::core {
namespace a = amd;
namespace cpu = strata::kernels::cpu;
using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point t) { return std::chrono::duration<double,std::milli>(Clock::now()-t).count(); }
AmdExperts::~AmdExperts() { close_worker(); }
void AmdExperts::fail_worker() {
    ++epoch_;
    if (pid_ > 0) {
        kill(pid_, SIGKILL);
        while (waitpid(pid_, nullptr, 0) < 0 && errno == EINTR) {}
        pid_ = -1;
    }
    if (socket_ >= 0) { ::close(socket_); socket_ = -1; }
}
void AmdExperts::close_worker() {
    if (socket_ >= 0 && !pending_work_) {
        a::Message shutdown; shutdown.op = a::Shutdown;
        exchange(shutdown);
    }
    fail_worker();
    pending_work_ = false;
    if (shared_) { munmap(shared_, sizeof(a::Shared)); shared_ = nullptr; }
    if (memory_ >= 0) { ::close(memory_); memory_ = -1; }
}
bool AmdExperts::exchange(a::Message& message) {
    message.sequence = ++sequence_; message.epoch = epoch_;
    a::Message reply;
    if (socket_ < 0 || !a::send(socket_, message) || !a::receive(socket_, reply) ||
        reply.sequence != message.sequence || reply.epoch != message.epoch || reply.op != message.op || reply.status) {
        fail_worker(); return false;
    }
    message = reply; return true;
}
bool AmdExperts::open_worker(const std::string& worker, const std::string& device, uint64_t budget,
    const std::vector<std::pair<int32_t,int32_t>>& ranked, const ExpertCache& primary,
    ExpertSource& source, std::string& error) {
    const auto& layout = cpu::expert_layout();
    if (!layout.native || layout.n_layers != 48 || layout.n_expert != 512 ||
        device.find("GPU-") != 0 || budget == 0 || budget > (12ULL << 30)) {
        error = "AMD helper requires native experts, explicit ROCr UUID and budget <= 12 GiB"; return false;
    }
    pairs_.clear();
    assigned_.assign(48*512, 0); source_ = &source;
    memory_ = memfd_create("strata-amd-batch", MFD_CLOEXEC | MFD_ALLOW_SEALING);
    if (memory_ < 0 || ftruncate(memory_, sizeof(a::Shared)) ||
        fcntl(memory_, F_ADD_SEALS, F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL)) {
        error = "AMD shared memory allocation failed"; close_worker(); return false;
    }
    shared_ = static_cast<a::Shared*>(mmap(nullptr, sizeof(a::Shared), PROT_READ | PROT_WRITE, MAP_SHARED, memory_, 0));
    if (shared_ == MAP_FAILED) { shared_ = nullptr; error = "AMD mmap failed"; close_worker(); return false; }
    int fds[2];
    if (socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0, fds)) {
        error = "AMD socketpair failed"; close_worker(); return false;
    }
    socket_ = fds[0];
    pid_ = fork();
    if (pid_ == 0) {
        const int s = fcntl(fds[1], F_DUPFD_CLOEXEC, 10), m = fcntl(memory_, F_DUPFD_CLOEXEC, 10);
        if (s < 0 || m < 0 || dup2(s, 3) < 0 || dup2(m, 4) < 0) _exit(126);
        syscall(SYS_close_range, 5U, ~0U, 0);
        execl(worker.c_str(), worker.c_str(), "3", "4", device.c_str(), static_cast<char*>(nullptr));
        _exit(127);
    }
    ::close(fds[1]);
    if (pid_ < 0) { error = "AMD worker fork failed"; close_worker(); return false; }
    a::Message hello; hello.op = a::Hello; hello.bytes = budget;
    if (!exchange(hello)) { error = "AMD worker handshake failed"; close_worker(); return false; }
    uint64_t used = 0;
    for (const auto& [layer, expert] : ranked) {
        if (layer < 0 || layer >= 48 || expert < 0 || expert >= 512 ||
            assigned_[(size_t)layer*512+expert] || primary.slot_of(layer, expert) >= 0) continue;
        const auto& fmt = layout.fmt[(size_t)layer];
        if (fmt.n_embd != a::H || fmt.n_ff != a::FF) {
            error = "AMD helper expert geometry must be 2560/640"; close_worker(); return false;
        }
        if (fmt.bytes > a::STAGING) { error = "AMD expert exceeds bounded staging buffer"; close_worker(); return false; }
        if (fmt.bytes > hello.bytes - used) continue;
        if (!source.copy_blob(layer, expert, shared_->staging)) {
            error = "AMD weight staging failed"; close_worker(); return false;
        }
        a::Message load; load.op = a::Load; load.layer = layer; load.expert = expert;
        load.gu_type = fmt.gu_type; load.d_type = fmt.d_type; load.bytes = fmt.bytes;
        if (!exchange(load)) { error = "AMD expert upload failed"; close_worker(); return false; }
        used += fmt.bytes;
        assigned_[(size_t)layer*512+expert] = 1;
        pairs_.emplace_back(layer, expert);
    }
    if (pairs_.empty()) { error = "AMD helper has no assigned experts"; close_worker(); return false; }
    std::fprintf(stderr, "strata AMD helper: %zu disjoint experts, %.2f GiB, 2 GiB free headroom\n",
                 pairs_.size(), (double)used/1073741824.0);
    return true;
}
bool AmdExperts::has(int64_t l, int64_t e) const {
    return l >= 0 && l < 48 && e >= 0 && e < 512 && !assigned_.empty() && assigned_[(size_t)l*512+e];
}
bool AmdExperts::begin(int64_t layer, const float* x, const int32_t* ids, int64_t tokens,
    int64_t k, const int32_t* kind, const int32_t* primary, std::string& error) {
    if (pending_work_ || tokens < 1 || tokens > a::TOKENS || k < 1 || k > 10) {
        error = "AMD helper invalid or overlapping batch"; return false;
    }
    entries_.assign((size_t)tokens*k, 0); rows_.clear();
    const auto started = Clock::now();
    std::memcpy(shared_->x, x, (size_t)tokens*a::H*sizeof(float));
    for (int i = 0; i < tokens*k; ++i) {
        const int e = ids[i];
        if (!has(layer, e) || (kind && kind[i] >= 0) ||
            (primary && primary[(size_t)layer*512+e] >= 0)) continue;
        const size_t row = rows_.size();
        shared_->ids[row] = e; shared_->token[row] = i/(int)k;
        entries_[(size_t)i] = 1; rows_.push_back(i);
    }
    if (rows_.empty()) return true;
    pending_ = a::Message{}; pending_.op = a::Run; pending_.layer = (int32_t)layer;
    pending_.tokens = (int32_t)tokens; pending_.entries = (int32_t)rows_.size();
    pending_.sequence = ++sequence_; pending_.epoch = epoch_;
    pending_work_ = true;
    if (socket_ >= 0 && !a::send(socket_, pending_)) fail_worker();
    transferred += (uint64_t)tokens*a::H*4 + rows_.size()*(a::H*4+8);
    transfer_ms += ms(started);
    return true;
}
bool AmdExperts::finish(float* out, std::string& error) {
    if (!pending_work_) return true;
    const auto started = Clock::now();
    a::Message reply;
    const bool valid = socket_ >= 0 && a::receive(socket_, reply) && reply.sequence == pending_.sequence &&
        reply.epoch == pending_.epoch && reply.op == a::Run && !reply.status;
    if (!valid) {
        fail_worker(); // wait for exit before using shared buffers; stale writes cannot race CPU fallback
        std::fprintf(stderr, "strata AMD helper: failed batch; recomputing %zu rows on CPU\n", rows_.size());
        const auto& fmt = cpu::expert_layout().fmt[(size_t)pending_.layer];
        uint8_t act[cpu::kNativeActBytes], h[cpu::kNativeHBytes];
        float ff[a::FF];
        for (size_t i = 0; i < rows_.size(); ++i) {
            const uint8_t* blob = source_->blob(pending_.layer, shared_->ids[i]);
            if (!blob) { error = "AMD fallback could not read expert"; pending_work_ = false; return false; }
            cpu::native_quant_act(fmt, shared_->x + (size_t)shared_->token[i]*a::H, act);
            const void* ap[1] = {act}; float* fp[1] = {ff};
            cpu::native_gu_rows(fmt, blob, ap, 1, fp, 0, a::FF);
            cpu::native_quant_h(fmt, ff, h);
            const void* hp[1] = {h}; float* op[1] = {out+(size_t)rows_[i]*a::H};
            cpu::native_down_rows(fmt, blob, hp, 1, op, 0, a::H);
            ++fallback_rows;
        }
    } else {
        device_transfer_ms += reply.transfer_ms; device_compute_ms += reply.compute_ms;
        for (size_t i = 0; i < rows_.size(); ++i)
            std::memcpy(out+(size_t)rows_[i]*a::H, shared_->result+i*a::H, a::H*sizeof(float));
    }
    wait_ms += ms(started); pending_work_ = false;
    return true;
}
void AmdExperts::cancel() {
    // Cancellation invalidates the generation and terminates any outstanding writes.
    fail_worker(); pending_work_ = false; entries_.clear(); rows_.clear();
}
} // namespace strata::core
