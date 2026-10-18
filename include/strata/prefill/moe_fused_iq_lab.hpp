// Opt-in laboratory/diagnostic API. Production interfaces stay in moe_fused_iq.hpp.
#pragma once
#include "strata/prefill/moe_fused_iq.hpp"
namespace strata::prefill::fused {
struct NativeLabInfo {
    int requested_variant = 0, variant = 0, phase = 0, format = -1, tile = 0, weight_rows = 0;
    int threads = 512, activation_stages = 4, prefetch = 2;
    int registers = 0, occupancy = 0, sms = 0;
    size_t dynamic_shared = 0;
    unsigned grid = 0;
};
// phase 1 gate/up including SwiGLU and quantization, 2 down; tile 0 automatic, 64 or 128 forced.
NativeLabInfo native_lab_info(const Batch&, const NativeGeom&, int n_expert, int64_t n,
                              int phase, int variant, int tile = 0);
// phase 0 executes both projections. Explicit variant overrides the environment.
void experts_native_lab(const Batch&, const NativeGeom&, int n_expert, int64_t n,
                        const void* scratch, const void* xa, const int32_t* src,
                        void* ha, float* dm, void* stream, int phase, int variant, int tile = 0);
// Allocate the diagnostic-only pinned count pool before prefill starts. No-op when profiling is off.
void native_profile_prepare();
// Call after an existing stream synchronization. Only completed events are collected; never synchronizes.
void native_profile_collect();
}  // namespace strata::prefill::fused
