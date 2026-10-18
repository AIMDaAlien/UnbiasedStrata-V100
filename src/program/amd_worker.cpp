// A separate HIP process. Only expert weights and bounded activation batches cross IPC.
#include "strata/core/amd_protocol.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace a = strata::amd;
namespace k = strata::kernels;
static void checked(hipError_t e) {
    if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e));
}
struct Weight { void* ptr = nullptr; k::NativeExpertLayout layout; };
int main(int argc, char** argv) {
    // Inherited descriptors and explicit ROCr UUID; there is no listener or network connection.
    if (argc != 4 || std::strncmp(argv[3], "GPU-", 4) != 0) return 2;
    const int socket = std::atoi(argv[1]), memory = std::atoi(argv[2]);
    setenv("ROCR_VISIBLE_DEVICES", argv[3], 1);
    struct stat st{};
    if (fstat(memory, &st) || st.st_size != sizeof(a::Shared)) return 2;
    auto* shared = static_cast<a::Shared*>(mmap(nullptr, sizeof(a::Shared), PROT_READ | PROT_WRITE,
                                              MAP_SHARED, memory, 0));
    if (shared == MAP_FAILED) return 2;
    a::Message request;
    uint64_t sequence = 0, epoch = 0, budget = 0, used = 0;
    std::map<std::pair<int,int>, Weight> weights;
    float *x = nullptr, *out = nullptr;
    void *q8 = nullptr, *scratch = nullptr;
    void* arena = nullptr;
    unsigned long long* ptr = nullptr;
    int32_t *start = nullptr, *dst = nullptr, *token = nullptr, *groups = nullptr;
    hipStream_t stream = nullptr;
    hipEvent_t timing[4]{};
    try {
        checked(hipSetDevice(0));
        hipDeviceProp_t properties{};
        checked(hipGetDeviceProperties(&properties, 0));
        if (std::string(properties.gcnArchName).find("gfx1200") != 0)
            throw std::runtime_error("worker device is not gfx1200");
        checked(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        for (auto& event : timing) checked(hipEventCreate(&event));
        checked(hipMalloc(&x, sizeof(shared->x)));
        checked(hipMalloc(&out, sizeof(shared->result)));
        checked(hipMalloc(&q8, a::TOKENS * a::H / 32 * 36));
        checked(hipMalloc(&scratch, k::native_expert_scratch_bytes(a::ENTRIES, a::FF)));
        checked(hipMalloc(&ptr, a::ENTRIES * sizeof(*ptr)));
        checked(hipMalloc(&start, (a::ENTRIES + 1) * sizeof(*start)));
        checked(hipMalloc(&dst, a::ENTRIES * sizeof(*dst)));
        checked(hipMalloc(&token, a::ENTRIES * sizeof(*token)));
        checked(hipMalloc(&groups, sizeof(*groups)));
        while (a::receive(socket, request, -1)) {
            auto reply = request;
            reply.status = 0;
            // Old generations and sequence numbers can never publish valid output.
            if (request.sequence <= sequence || request.epoch < epoch) break;
            sequence = request.sequence;
            epoch = request.epoch;
            if (request.op == a::Hello) {
                if (arena) break;
                size_t free = 0, total = 0;
                checked(hipMemGetInfo(&free, &total));
                constexpr uint64_t reserve = 2ULL << 30, cap = 12ULL << 30;
                budget = std::min(request.bytes, std::min(cap, free > reserve ? free - reserve : 0));
                if (!budget) throw std::runtime_error("no VRAM above the 2 GiB headroom");
                // Individual HIP allocations round small blobs up substantially on
                // this device. One bounded arena preserves the actual VRAM budget.
                checked(hipMalloc(&arena, budget));
                reply.bytes = budget;
            } else if (request.op == a::Load) {
                const auto layout = k::native_expert_layout(request.gu_type, request.d_type, a::H, a::FF);
                const auto key = std::make_pair(request.layer, request.expert);
                if (request.layer < 0 || request.layer >= 48 || request.expert < 0 || request.expert >= 512 ||
                    !k::native_expert_supported(request.gu_type, request.d_type, a::H, a::FF) ||
                    request.bytes != layout.bytes || request.bytes > a::STAGING ||
                    request.bytes > budget - used || weights.count(key)) {
                    reply.status = 1;
                } else {
                    Weight w;
                    w.layout = layout;
                    w.ptr = static_cast<uint8_t*>(arena) + used;
                    checked(hipMemcpyAsync(w.ptr, shared->staging, layout.bytes, hipMemcpyHostToDevice, stream));
                    checked(hipStreamSynchronize(stream));
                    weights.emplace(key, w);
                    used += layout.bytes;
                }
            } else if (request.op == a::Run) {
                if (request.tokens < 1 || request.tokens > a::TOKENS ||
                    request.entries < 1 || request.entries > a::ENTRIES) break;
                std::map<int, std::vector<int>> grouped;
                k::NativeExpertLayout layout;
                for (int i = 0; i < request.entries; ++i) {
                    const int id = shared->ids[i];
                    const auto it = weights.find({request.layer, id});
                    if (it == weights.end() || shared->token[i] < 0 || shared->token[i] >= request.tokens)
                        throw std::runtime_error("invalid routed expert or token");
                    layout = it->second.layout;
                    grouped[id].push_back(i);
                }
                std::vector<unsigned long long> pointers;
                std::vector<int32_t> starts{0}, destinations, tokens;
                for (const auto& group : grouped) {
                    pointers.push_back(reinterpret_cast<unsigned long long>(weights.at({request.layer, group.first}).ptr));
                    for (int entry : group.second) {
                        destinations.push_back(entry);
                        tokens.push_back(shared->token[entry]);
                    }
                    starts.push_back(static_cast<int32_t>(destinations.size()));
                }
                const int32_t ng = static_cast<int32_t>(pointers.size());
                checked(hipEventRecord(timing[0], stream));
                checked(hipMemcpyAsync(x, shared->x, request.tokens * a::H * sizeof(float), hipMemcpyHostToDevice, stream));
                checked(hipMemcpyAsync(ptr, pointers.data(), pointers.size() * sizeof(*ptr), hipMemcpyHostToDevice, stream));
                checked(hipMemcpyAsync(start, starts.data(), starts.size() * sizeof(*start), hipMemcpyHostToDevice, stream));
                checked(hipMemcpyAsync(dst, destinations.data(), destinations.size() * sizeof(*dst), hipMemcpyHostToDevice, stream));
                checked(hipMemcpyAsync(token, tokens.data(), tokens.size() * sizeof(*token), hipMemcpyHostToDevice, stream));
                checked(hipMemcpyAsync(groups, &ng, sizeof(ng), hipMemcpyHostToDevice, stream));
                checked(hipEventRecord(timing[1], stream));
                k::quantize_q8_1_rows(x, request.tokens, a::H, q8, stream);
                k::native_expert_grouped(layout, ptr, start, groups, dst, token, ng, request.entries,
                                          q8, scratch, out, stream, ng);
                checked(hipGetLastError());
                checked(hipEventRecord(timing[2], stream));
                checked(hipMemcpyAsync(shared->result, out, request.entries * a::H * sizeof(float),
                                       hipMemcpyDeviceToHost, stream));
                checked(hipEventRecord(timing[3], stream));
                checked(hipStreamSynchronize(stream));
                float input_ms = 0, output_ms = 0, compute_ms = 0;
                checked(hipEventElapsedTime(&input_ms, timing[0], timing[1]));
                checked(hipEventElapsedTime(&compute_ms, timing[1], timing[2]));
                checked(hipEventElapsedTime(&output_ms, timing[2], timing[3]));
                reply.transfer_ms = input_ms + output_ms; reply.compute_ms = compute_ms;
            } else if (request.op == a::Cancel) {
                checked(hipStreamSynchronize(stream));
            } else if (request.op == a::Shutdown) {
                a::send(socket, reply);
                break;
            } else break;
            if (!a::send(socket, reply)) break;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "amd expert worker: %s\n", e.what());
        return 1; // parent invalidates the batch and recomputes on CPU
    }
    hipFree(arena);
    hipFree(x); hipFree(out); hipFree(q8); hipFree(scratch);
    hipFree(ptr); hipFree(start); hipFree(dst); hipFree(token); hipFree(groups);
    if (stream) hipStreamDestroy(stream);
    for (auto event : timing) if (event) hipEventDestroy(event);
    munmap(shared, sizeof(a::Shared));
    close(memory); close(socket);
    return 0;
}
