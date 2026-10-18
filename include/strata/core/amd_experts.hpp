#pragma once
#include "strata/core/remote_experts.hpp"
#ifdef __linux__
#include "strata/core/amd_protocol.hpp"
#include <sys/types.h>

namespace strata::core {
class AmdExperts final : public RemoteExperts {
public:
    ~AmdExperts() override;
    bool open_worker(const std::string& worker, const std::string& device, uint64_t budget,
                     const std::vector<std::pair<int32_t,int32_t>>& ranked,
                     const ExpertCache& primary, ExpertSource& source, std::string& error);
    bool begin(int64_t layer, const float* x, const int32_t* ids, int64_t tokens,
               int64_t k, const int32_t* kind, const int32_t* primary, std::string& error) override;
    bool owns(int64_t i) const override { return i >= 0 && (size_t)i < entries_.size() && entries_[i]; }
    bool has(int64_t layer, int64_t expert) const;
    bool finish(float* out, std::string& error) override;
    void cancel();
    void close_worker();
    const std::vector<std::pair<int32_t,int32_t>>& pairs() const { return pairs_; }
    double transfer_ms = 0, wait_ms = 0, device_transfer_ms = 0, device_compute_ms = 0;
    uint64_t transferred = 0, fallback_rows = 0;
private:
    bool exchange(amd::Message& message);
    void fail_worker();
    int socket_ = -1, memory_ = -1;
    pid_t pid_ = -1;
    amd::Shared* shared_ = nullptr;
    uint64_t sequence_ = 0, epoch_ = 1;
    amd::Message pending_{};
    bool pending_work_ = false;
    ExpertSource* source_ = nullptr;
    std::vector<uint8_t> assigned_, entries_;
    std::vector<int32_t> rows_;
    std::vector<std::pair<int32_t,int32_t>> pairs_;
};
} // namespace strata::core
#else
namespace strata::amd { inline constexpr int TOKENS = 256, ENTRIES = TOKENS * 10; }
namespace strata::core {
class AmdExperts final : public RemoteExperts {
public:
    bool open_worker(const std::string&, const std::string&, uint64_t,
                     const std::vector<std::pair<int32_t,int32_t>>&, const ExpertCache&,
                     ExpertSource&, std::string& error) { error = "AMD helper requires Linux"; return false; }
    bool has(int64_t, int64_t) const { return false; }
    const std::vector<std::pair<int32_t,int32_t>>& pairs() const { return pairs_; }
    double transfer_ms = 0, wait_ms = 0, device_transfer_ms = 0, device_compute_ms = 0;
    uint64_t transferred = 0, fallback_rows = 0;
private:
    std::vector<std::pair<int32_t,int32_t>> pairs_;
};
}
#endif
