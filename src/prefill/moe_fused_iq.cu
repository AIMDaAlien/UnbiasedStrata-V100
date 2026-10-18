// src/prefill/moe_fused_iq.cu - see include/strata/prefill/moe_fused_iq.hpp (#136: the native packs' prompt experts on
// the fused int8 kernels of moe_fused.cu).
//
// The arithmetic.  An activation block of 32 values is x = d_x * a, a = round(x / d_x) in -127..127, stored in natural
// order (per 64 values: 64 codes, then {d_0, d_1} and 8 unused bytes - 80 bytes, the Q2_0 path's size).  A weight
// block of 32 (or 16) values is w = d_w * q with q an int8: the i-quants' codebook entries with their signs applied
// (IQ2_XXS / IQ2_XS / IQ2_S: 0..43, IQ3_XXS: 0..62, IQ3_S: 0..15, IQ4_XS / IQ4_NL: -127..113), Q2_0's code - 1.  So
//     sum w x = d_w * d_x * sum q a,     |sum q a| <= 32 * 127 * 127 < 2^22,
// and the int32 dot becomes a float exactly with one float add: the mma starts from C = 0x4B400000 (the bits of
// 1.5 * 2^23), so as_float(D) - 1.5 * 2^23 is the dot.  Per output and 32 values: that add, d_w * d_x, one fma (the
// formats with a scale per 16 values: two dots, each times its d_w, then d_x).  d_w is llama.cpp's own: IQ2_XXS
// d (2s + 1) / 8, IQ2_XS / IQ2_S d (2s + 1) / 8 per 16, IQ3_XXS d (2s + 1) / 4, IQ3_S d (2s + 1), IQ4_XS d (s - 32),
// IQ4_NL and Q2_0 d (dequantize_row_* in ggml-quants.c, mmq-load-tiles.cuh).
//
// The load stage.  Per 64 values of K (a stage), a thread decodes one 32-value sub-block of one of the work item's
// weight rows: its block bytes are read from the blob with 16-bit loads (the GGUF blocks are 2-byte aligned: 66 / 74 /
// 82 / 98 / 110 / 136 / 18 bytes) one stage ahead into registers (and two 256-value blocks ahead into L2), then turned
// into 32 int8 and the sub-block's scales in shared memory, double-buffered.  The codebooks live in shared memory (up
// to 8 KB, IQ2_S).  The activations arrive by cp.async, four stages deep, as in moe_fused.cu; the fragments come by
// ldmatrix.  The products are mma.sync m16n8k32 (m16n8k16 for the formats with a scale per 16 values); the epilogues
// (SwiGLU, H to int8 per 32 features; down into the per-slot rows) are moe_fused.cu's.
#include "strata/prefill/moe_fused_iq.hpp"
#include "strata/prefill/moe_fused_iq_lab.hpp"
#include "strata/prefill/moe_fused_iq_dispatch.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "ggml.h"

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace strata::prefill::fused {
namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "prefill fused experts (native): %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

constexpr int AB = 80;                  // activation bytes per 64 values: 64 codes, {d0, d1}, 8 unused
constexpr int MAGIC = 0x4B400000;       // the bits of 1.5 * 2^23
constexpr float MAGICF = 12582912.0f;
// 16 warps, each 64 weight rows x 16 routed rows; WW of them along the weight rows and 16 / WW along the routed rows: a
// work item is (WR = 64 WW weight rows) x (TR = 256 / WW routed rows of one expert) - 256 x 64 or 128 x 128, chosen
// per launch by the rows per expert (pick_ww)
constexpr int THREADS = 512;
constexpr int WLD = 80;                 // bytes per decoded weight row of a stage: 64 int8 + 16 (no bank conflicts)
__host__ __device__ constexpr int tile_rows(int ww, int threads = THREADS) { return (threads / 32) * 16 / ww; }
__host__ __device__ constexpr int weight_rows(int ww) { return 64 * ww; }
constexpr int ASTAGES = 4;              // cp.async depth of the activations (64 values of K a stage)
constexpr int GU_ROWS_K = 2560, D_ROWS_K = 640;   // K of gate/up (n_embd) and of down (n_ff)

// the formats (ggml type ids)
constexpr int T_IQ2_XXS = GGML_TYPE_IQ2_XXS, T_IQ2_XS = GGML_TYPE_IQ2_XS, T_IQ2_S = GGML_TYPE_IQ2_S,
              T_IQ3_XXS = GGML_TYPE_IQ3_XXS, T_IQ3_S = GGML_TYPE_IQ3_S, T_IQ4_XS = GGML_TYPE_IQ4_XS,
              T_IQ4_NL = GGML_TYPE_IQ4_NL, T_Q2_0 = GGML_TYPE_Q2_0;
static_assert(T_IQ2_XXS == 16 && T_IQ2_XS == 17 && T_IQ3_XXS == 18 && T_IQ4_NL == 20 &&
              T_IQ3_S == 21 && T_IQ2_S == 22 && T_Q2_0 == 42,
              "frozen expert dispatch uses these pack format ids");
static_assert(sizeof(block_iq2_xxs) == 66 && sizeof(block_iq2_xs) == 74 && sizeof(block_iq2_s) == 82 &&
              sizeof(block_iq3_xxs) == 98 && sizeof(block_iq3_s) == 110 && sizeof(block_iq4_xs) == 136 &&
              sizeof(block_iq4_nl) == 18 && sizeof(block_q2_0) == 18, "the block layouts this file decodes");

// block bytes, scale per 16 values, codebook bytes in shared memory
__host__ __device__ constexpr int block_bytes(int t) {
    return t == T_IQ2_XXS ? 66 : t == T_IQ2_XS ? 74 : t == T_IQ2_S ? 82 : t == T_IQ3_XXS ? 98 : t == T_IQ3_S ? 110
         : t == T_IQ4_XS ? 136 : 18;
}
__host__ __device__ constexpr bool per16(int t) { return t == T_IQ2_XS || t == T_IQ2_S; }
__host__ __device__ constexpr int grid_bytes(int t) {
    return t == T_IQ2_XXS ? 256 * 8 : t == T_IQ2_XS ? 512 * 8 : t == T_IQ2_S ? 1024 * 8 : t == T_IQ3_XXS ? 256 * 4
         : t == T_IQ3_S ? 512 * 4 : 0;
}
__host__ __device__ constexpr size_t smem_bytes(int t, int ww, int threads = THREADS, int stages = ASTAGES) {
    return (size_t) 2 * weight_rows(ww) * (WLD + 16) + (size_t) stages * tile_rows(ww, threads) * AB +
           (size_t) tile_rows(ww, threads) * 4 + grid_bytes(t);
}

// moe_fused.cu's grouping tables (the same layout: fused::group writes them)
struct Tables {
    int32_t *cnt, *off, *fill, *ts;
    int2* tiles;
};
size_t tiles_at(int n_expert) { return ((size_t) (4 * n_expert + 2) * 4 + 15) / 16 * 16; }
Tables tables(void* scratch, int n_expert) {
    int32_t* p = (int32_t*) scratch;
    return {p, p + n_expert, p + 2 * n_expert + 1, p + 3 * n_expert + 1,
            (int2*) ((uint8_t*) scratch + tiles_at(n_expert))};
}

// ---- activations in natural order: one warp per 64 values
__global__ void quant_act_nat_kernel(const float* __restrict__ x, int64_t nblk, uint8_t* __restrict__ xa) {
    const int64_t w = (int64_t) blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (w >= nblk) return;
    const float v0 = x[w * 64 + lane], v1 = x[w * 64 + 32 + lane];
    float a0 = fabsf(v0), a1 = fabsf(v1);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        a0 = fmaxf(a0, __shfl_xor_sync(0xffffffffu, a0, o));
        a1 = fmaxf(a1, __shfl_xor_sync(0xffffffffu, a1, o));
    }
    uint8_t* out = xa + w * AB;
    out[lane] = (uint8_t) (int8_t) (a0 > 0.0f ? __float2int_rn(v0 * (127.0f / a0)) : 0);
    out[32 + lane] = (uint8_t) (int8_t) (a1 > 0.0f ? __float2int_rn(v1 * (127.0f / a1)) : 0);
    if (lane == 0) *(float4*) (out + 64) = make_float4(a0 / 127.0f, a1 / 127.0f, 0.0f, 0.0f);
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ void cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned) __cvta_generic_to_shared(dst)),
                 "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void pf_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];\n" ::"l"(p)); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
// d = MAGIC + A (16 x 32 s8, row) * B (32 x 8 s8, col): the int32 dot with the magic bias already added (the C
// operand), so as_float(d) - 1.5 * 2^23 is the dot as a float
__device__ __forceinline__ void mma32(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(MAGIC));
}
// the same at m16n8k16 (A 16 x 16, B 16 x 8)
__device__ __forceinline__ void mma16(int (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a0), "r"(a1), "r"(b0), "r"(MAGIC));
}
__device__ __forceinline__ float dotf(int d) { return __int_as_float(d) - MAGICF; }
// four 8 x 16-byte matrices, row addresses from lanes 0-7, 8-15, 16-23, 24-31: lane (g, t) gets bytes 4t..4t+3 of row g
// of each - an mma fragment
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"((unsigned) __cvta_generic_to_shared(p))
                 : "memory");
}

// ---- the load stage: a 32-value sub-block's bytes into registers, then int8 and scales
__device__ __forceinline__ uint32_t ld16(const uint8_t* p) { return *(const uint16_t*) p; }
__device__ __forceinline__ uint32_t ld32(const uint8_t* p) { return ld16(p) | (ld16(p + 2) << 16); }
__device__ __forceinline__ float half_at(uint32_t w) { return __half2float(__ushort_as_half((unsigned short) w)); }
// llama.cpp's sign unpacking: 7 bits of signs, the 8th their parity (bit 7 of v may be anything)
__device__ __forceinline__ uint32_t unpack_ksigns(uint32_t v) {
    v &= 0xFF;
    const uint32_t p = __popc(v) & 1;
    return (v ^ p << 7) * 0x01010101u;
}
// 8 bytes of a codebook entry with the sign byte `s` (broadcast) applied: bits 0-3 to .x, 4-7 to .y
__device__ __forceinline__ void signed8(uint32_t gx, uint32_t gy, uint32_t s, uint32_t& qx, uint32_t& qy) {
    const uint32_t m0 = __vcmpne4(s & 0x08040201u, 0), m1 = __vcmpne4(s & 0x80402010u, 0);
    qx = __vsub4(gx ^ m0, m0);
    qy = __vsub4(gy ^ m1, m1);
}
// 8 nibbles of q4 through a 16-entry int8 table (4 words): the low nibbles' values in .x, the high ones' in .y
__device__ __forceinline__ void table16(uint32_t q4, const uint32_t (&t)[4], uint32_t& lo, uint32_t& hi) {
    uint32_t tmp[2];
    const uint32_t sel = 0x32103210u | ((q4 & 0x88888888u) >> 1);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t sh = 16 * i;
        const uint32_t l = __byte_perm(t[0], t[1], q4 >> sh), h = __byte_perm(t[2], t[3], q4 >> sh);
        tmp[i] = __byte_perm(l, h, sel >> sh);
    }
    lo = __byte_perm(tmp[0], tmp[1], 0x6420);
    hi = __byte_perm(tmp[0], tmp[1], 0x7531);
}

// Raw bytes of sub-block `ib` of the block at `bp` (IQ: the 256-value super-block, ib 0..7; Q2_0: the 64-value block,
// ib 0..1; IQ4_NL: the 32-value block).
template <int T> __device__ __forceinline__ void load_unit(const uint8_t* bp, int ib, uint32_t (&w)[5]) {
    if constexpr (T == T_IQ2_XXS) {
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld16(bp);
    } else if constexpr (T == T_IQ2_XS) {
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld16(bp) | ((uint32_t) bp[66 + ib] << 16);
    } else if constexpr (T == T_IQ2_S) {
        w[0] = ld32(bp + 2 + 4 * ib); w[1] = ld32(bp + 34 + 4 * ib);
        w[2] = ld16(bp) | ((uint32_t) bp[66 + ib] << 16) | ((uint32_t) bp[74 + ib] << 24);
    } else if constexpr (T == T_IQ3_XXS) {
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld32(bp + 66 + 4 * ib); w[3] = ld16(bp);
    } else if constexpr (T == T_IQ3_S) {
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld32(bp + 74 + 4 * ib);
        w[3] = ld16(bp) | ((uint32_t) bp[66 + ib] << 16) | ((uint32_t) ((bp[106 + ib / 2] >> (4 * (ib & 1))) & 15) << 24);
    } else if constexpr (T == T_IQ4_XS) {
#pragma unroll
        for (int k = 0; k < 4; ++k) w[k] = ld32(bp + 8 + 16 * ib + 4 * k);
        const uint32_t ls = ((bp[4 + ib / 2] >> (4 * (ib & 1))) & 15) | (((ld16(bp + 2) >> (2 * ib)) & 3) << 4);
        w[4] = ld16(bp) | (ls << 16);
    } else if constexpr (T == T_IQ4_NL) {
#pragma unroll
        for (int k = 0; k < 4; ++k) w[k] = ld32(bp + 2 + 4 * k);
        w[4] = ld16(bp);
    } else {   // Q2_0
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld16(bp);
    }
}

// The sub-block as 32 int8 (q[0..7], natural order) and its scales (s0: values 0-15, s1: 16-31).
template <int T>
__device__ __forceinline__ void convert(const uint32_t (&w)[5], const uint8_t* grid, const uint32_t (&kv)[4],
                                        uint32_t (&q)[8], float& s0, float& s1) {
    if constexpr (T == T_IQ2_XXS) {
        const uint2* g = (const uint2*) grid;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 e = g[(w[0] >> (8 * l)) & 255];
            signed8(e.x, e.y, unpack_ksigns(w[1] >> (7 * l)), q[2 * l], q[2 * l + 1]);
        }
        s0 = s1 = half_at(w[2]) * (float) ((w[1] >> 27) | 1) * 0.125f;
    } else if constexpr (T == T_IQ2_XS) {
        const uint2* g = (const uint2*) grid;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t c = (w[l >> 1] >> (16 * (l & 1))) & 0xFFFF;
            const uint2 e = g[c & 511];
            signed8(e.x, e.y, unpack_ksigns(c >> 9), q[2 * l], q[2 * l + 1]);
        }
        const float d = half_at(w[2]);
        const uint32_t sc = w[2] >> 16;
        s0 = d * (float) (2 * (sc & 15) + 1) * 0.125f;
        s1 = d * (float) (2 * ((sc >> 4) & 15) + 1) * 0.125f;
    } else if constexpr (T == T_IQ2_S) {
        const uint2* g = (const uint2*) grid;
        const uint32_t qh = (w[2] >> 16) & 255;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 e = g[((w[0] >> (8 * l)) & 255) | ((qh << (8 - 2 * l)) & 0x300)];
            signed8(e.x, e.y, ((w[1] >> (8 * l)) & 255) * 0x01010101u, q[2 * l], q[2 * l + 1]);
        }
        const float d = half_at(w[2]);
        const uint32_t sc = w[2] >> 24;
        s0 = d * (float) (2 * (sc & 15) + 1) * 0.125f;
        s1 = d * (float) (2 * (sc >> 4) + 1) * 0.125f;
    } else if constexpr (T == T_IQ3_XXS) {
        const uint32_t* g = (const uint32_t*) grid;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t i0 = (w[l >> 1] >> (16 * (l & 1))) & 255, i1 = (w[l >> 1] >> (16 * (l & 1) + 8)) & 255;
            signed8(g[i0], g[i1], unpack_ksigns(w[2] >> (7 * l)), q[2 * l], q[2 * l + 1]);
        }
        s0 = s1 = half_at(w[3]) * (float) (2 * (w[2] >> 28) + 1) * 0.25f;
    } else if constexpr (T == T_IQ3_S) {
        const uint32_t* g = (const uint32_t*) grid;
        const uint32_t qh = (w[3] >> 16) & 255;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t i0 = (w[l >> 1] >> (16 * (l & 1))) & 255, i1 = (w[l >> 1] >> (16 * (l & 1) + 8)) & 255;
            signed8(g[i0 | ((qh << (8 - 2 * l)) & 256)], g[i1 | ((qh << (7 - 2 * l)) & 256)],
                    ((w[2] >> (8 * l)) & 255) * 0x01010101u, q[2 * l], q[2 * l + 1]);
        }
        s0 = s1 = half_at(w[3]) * (float) (1 + 2 * (w[3] >> 24));
    } else if constexpr (T == T_IQ4_XS || T == T_IQ4_NL) {
#pragma unroll
        for (int k = 0; k < 4; ++k) table16(w[k], kv, q[k], q[4 + k]);
        s0 = s1 = T == T_IQ4_XS ? half_at(w[4]) * (float) ((int) (w[4] >> 16) - 32) : half_at(w[4]);
    } else {   // Q2_0: code - 1 via a byte table {-1, 0, 1, 2}
#pragma unroll
        for (int h = 0; h < 4; ++h) {
            const uint32_t c = (w[h >> 1] >> (16 * (h & 1))) & 0xFFFF;
            const uint32_t qe = __byte_perm(0x020100FFu, 0x020100FFu, c & 0x7777);
            const uint32_t qo = __byte_perm(0x020100FFu, 0x020100FFu, (c >> 2) & 0x7777);
            q[2 * h] = __byte_perm(qe, qo, 0x5140);
            q[2 * h + 1] = __byte_perm(qe, qo, 0x7362);
        }
        s0 = s1 = half_at(w[2]);
    }
}

template <int T> __device__ __forceinline__ const void* grid_src() {
    if constexpr (T == T_IQ2_XXS) return iq2xxs_grid;
    else if constexpr (T == T_IQ2_XS) return iq2xs_grid;
    else if constexpr (T == T_IQ2_S) return iq2s_grid;
    else if constexpr (T == T_IQ3_XXS) return iq3xxs_grid;
    else if constexpr (T == T_IQ3_S) return iq3s_grid;
    else return nullptr;
}
#endif

// One work item = (TR routed rows of expert e - one or two of group()'s 64-row tiles -, a block of WR weight rows);
// a persistent grid walks the batch's items.  GU: gate/up rows of WR / 2 features (format WT) against the rows'
// tokens' activations, SwiGLU, H to int8 per 32 features into `out`.  !GU: down rows (format WT) against H, FP32 into
// `dm`.  Warp (wf, wt): weight rows 64 wf.. as four m16 tiles (GU: a tile = 8 features, gate rows as mma rows 0-7 and
// up rows as 8-15, so a lane holds gate and up of one feature), routed rows 16 wt.. as two n8.
template <int WT, bool GU, int WW, int NT = THREADS, int AS = ASTAGES, int PREFETCH = 2>
__global__ void __launch_bounds__(NT, 1)
native_kernel(const Batch b, const NativeGeom geo, const Tables tb, const uint8_t* __restrict__ act,
              const int32_t* __restrict__ src, uint8_t* __restrict__ out, float* __restrict__ dm) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int TR = tile_rows(WW, NT), WR = weight_rows(WW);
    constexpr int WT_BYTES = WR * WLD, WS_FLOATS = WR * 4, ACT_STAGE = TR * AB;
    constexpr int NS = (GU ? GU_ROWS_K : D_ROWS_K) / 64;   // 64-value stages along K
    constexpr int NFB = (GU ? 1280 : 2560) / WR;          // weight-row blocks per tile
    constexpr int ACT_LD = NS * AB;                       // bytes per activation row: 3200 (a token), 800 (a row's H)
    constexpr int BS = block_bytes(WT);
    constexpr bool K16 = per16(WT);
    extern __shared__ __align__(16) uint8_t smem[];
    uint8_t* wt = smem;                                               // [2][WR][WLD] decoded int8 weights
    float* ws = (float*) (smem + 2 * WT_BYTES);                       // [2][WR][4] their scales per 16 values
    uint8_t* stages = (uint8_t*) (ws + 2 * WS_FLOATS);                // [AS][TR][AB] activations
    int* srow = (int*) (stages + AS * ACT_STAGE);                // [TR] the activation row of each tile row
    uint8_t* sgrid = (uint8_t*) (srow + TR);                          // the codebook

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, tig = lane & 3;
    const int wf = warp % WW, nb0 = 16 * (warp / WW);
    if constexpr (grid_bytes(WT) > 0) {
        const uint32_t* gs = (const uint32_t*) grid_src<WT>();
        for (int i = tid; i < grid_bytes(WT) / 4; i += NT) ((uint32_t*) sgrid)[i] = gs[i];
    }
    uint32_t kv[4] = {0, 0, 0, 0};
    if constexpr (WT == T_IQ4_XS || WT == T_IQ4_NL) {
#pragma unroll
        for (int k = 0; k < 16; ++k) kv[k >> 2] |= (uint32_t) (uint8_t) kvalues_iq4nl[k] << (8 * (k & 3));
    }
    // Local weight rows: m tile i of warp wf is rows 64 wf + 16 i.. (mma rows g and g + 8: rows +g and +8+g).  For
    // gate/up a tile is 8 features, its gate rows then its up rows, so a lane holds gate and up of one feature.  The
    // fragments come by ldmatrix: lane l gives the address of row (l & 7) of matrix l >> 3 - A: {rows 0-7, rows 8-15}
    // x {bytes 0-15, 16-31} of a 32-value chunk; B: {tokens nb0 + 0-7, + 8-15} x {bytes 0-15, 16-31}.
    const int lm = lane >> 3, l8 = lane & 7;
    const int a_off = (64 * wf + 8 * (lm & 1) + l8) * WLD + 16 * (lm >> 1);
    const int b_off = (nb0 + 8 * (lm >> 1) + l8) * AB + 16 * (lm & 1);
    // this thread's decode unit (the first 2 WR threads): local weight row ur, sub-block uj of each stage
    const bool dec = tid < 2 * WR;
    const int ur = dec ? tid >> 1 : 0, uj = tid & 1;
    const int t0 = tb.ts[b.e0], nwork = (tb.ts[b.e1] - t0) * NFB;
    for (int w = blockIdx.x; w < nwork; w += gridDim.x) {
        const int2 tl = tb.tiles[t0 + w / NFB];
        const int fb = w % NFB, e = tl.x, row0 = tl.y;
        if ((row0 - tb.off[e]) % TR != 0) continue;              // a 64-row tile inside an earlier item's TR rows
        const int nrows = min(TR, tb.off[e + 1] - row0);
        const uint8_t* blob = b.blob[e - b.e0];
        const int rbase = fb * WR;
        // local row ur: gate/up - feature (WR / 2) fb + 8 (ur >> 4) + (ur & 7), its up row when bit 3 is set
        const uint8_t* wrow = GU ? blob + ((ur & 8) ? geo.up_off : 0) +
                                       (size_t) (fb * (WR / 2) + 8 * (ur >> 4) + (ur & 7)) * geo.gu_row
                                 : blob + geo.down_off + (size_t) (rbase + ur) * geo.d_row;
        auto unit = [&](int s) -> const uint8_t* {                  // the block of stage s's sub-block
            if (GU) return wrow + (s >> 2) * BS;
            return WT == T_IQ4_NL ? wrow + (2 * s + uj) * BS : wrow + s * BS;
        };
        auto sub = [&](int s) { return GU ? 2 * (s & 3) + uj : uj; };
        // the weight bytes into L2 ahead of the register loads (one stage ahead only - less than a DRAM round trip):
        // gate/up PF super-blocks ahead (the row's two threads take a block's first and last line), down the whole row
        // slice at once (180 or 360 bytes a row)
        constexpr int PF = PREFETCH;
        auto prefetch_sb = [&](int sb) {
            if (dec && sb < GU_ROWS_K / 256) pf_l2(wrow + sb * BS + (uj ? BS - 1 : 0));
        };
        if (GU) {
#pragma unroll
            for (int sb = 0; sb < PF; ++sb) prefetch_sb(sb);
        } else if (dec) {
            for (size_t o = 128 * (size_t) uj; o < geo.d_row + 127; o += 256) pf_l2(wrow + min(o, geo.d_row - 1));
        }
        __syncthreads();                                          // the previous item is done with the buffers
        if (tid < TR) srow[tid] = tid < nrows ? (GU ? src[row0 + tid] : row0 + tid) : -1;
        __syncthreads();                                          // srow
        // the rows past the item's end are not loaded: a warp entirely past it skips the products, and the columns
        // of a partial one are never written (no reduction mixes columns)
        auto load_act = [&](int s) {
            uint8_t* st = stages + (s % AS) * ACT_STAGE;
            for (int c = tid; c < TR * 5; c += NT) {
                const int r = c / 5, q = c % 5;
                if (r < nrows) cp16(st + r * AB + q * 16, act + (size_t) srow[r] * ACT_LD + s * AB + q * 16);
            }
        };
        auto put = [&](const uint32_t (&raw)[5], int buf) {
            if (!dec) return;
            uint32_t q[8];
            float s0, s1;
            convert<WT>(raw, sgrid, kv, q, s0, s1);
            uint4* d = (uint4*) (wt + buf * WT_BYTES + ur * WLD + 32 * uj);
            d[0] = make_uint4(q[0], q[1], q[2], q[3]);
            d[1] = make_uint4(q[4], q[5], q[6], q[7]);
            *(float2*) (ws + buf * WS_FLOATS + ur * 4 + 2 * uj) = make_float2(s0, s1);
        };
#pragma unroll
        for (int s = 0; s < AS - 1; ++s) {
            if (s < NS) load_act(s);
            cp_commit();
        }
        uint32_t raw[5] = {0, 0, 0, 0, 0};
        if (dec) load_unit<WT>(unit(0), sub(0), raw);
        put(raw, 0);
        if (dec) load_unit<WT>(unit(1), sub(1), raw);
        // a warp whose rows are all past the tile's end only takes part in the loads
        const bool on0 = nb0 < nrows, on1 = nb0 + 8 < nrows;
        float acc[4][2][4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int n = 0; n < 2; ++n)
#pragma unroll
                for (int q = 0; q < 4; ++q) acc[i][n][q] = 0.0f;
        for (int s = 0; s < NS; ++s) {
            cp_wait<AS - 2>();
            __syncthreads();
            if (s + AS - 1 < NS) load_act(s + AS - 1);
            cp_commit();
            if (GU && (s & 3) == 0) prefetch_sb((s >> 2) + PF);
            if (on0) {
                const uint8_t* W = wt + (s & 1) * WT_BYTES;
                const float* S = ws + (s & 1) * WS_FLOATS;
                const uint8_t* sa = stages + (s % AS) * ACT_STAGE;
                // the stage's scales: per weight row 4 (per 16 values), per routed row 2 (per 32)
                float2 dx[2][2];                                  // [n8 tile][column 2 tig + cc]
#pragma unroll
                for (int n = 0; n < 2; ++n)
#pragma unroll
                    for (int cc = 0; cc < 2; ++cc) dx[n][cc] = *(const float2*) (sa + (nb0 + 8 * n + 2 * tig + cc) * AB + 64);
#pragma unroll
                for (int h = 0; h < 2; ++h) {                     // the stage's two 32-value halves
                    uint32_t bq[4];                               // {n0 bytes 0-15, n0 16-31, n1 0-15, n1 16-31}
                    ldsm4(bq, sa + b_off + 32 * h);
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        uint32_t a[4];                            // {rows g, g + 8} x {bytes 0-15, 16-31}
                        ldsm4(a, W + a_off + 16 * i * WLD + 32 * h);
                        const float2 swa = *(const float2*) (S + (64 * wf + 16 * i + g) * 4 + 2 * h);
                        const float2 swb = *(const float2*) (S + (64 * wf + 16 * i + 8 + g) * 4 + 2 * h);
                        const float wa0 = swa.x, wa1 = swa.y, wb0 = swb.x, wb1 = swb.y;
#pragma unroll
                        for (int n = 0; n < 2; ++n) {
                            if (n == 1 && !on1) break;
                            if constexpr (K16) {
                                // a scale per 16 weights: the two halves' dots, each times its scale, then d_x
                                int d0[4], d1[4];
                                mma16(d0, a[0], a[1], bq[2 * n]);
                                mma16(d1, a[2], a[3], bq[2 * n + 1]);
#pragma unroll
                                for (int q = 0; q < 4; ++q) {
                                    const float v = fmaf(q < 2 ? wa1 : wb1, dotf(d1[q]), (q < 2 ? wa0 : wb0) * dotf(d0[q]));
                                    acc[i][n][q] = fmaf(h ? dx[n][q & 1].y : dx[n][q & 1].x, v, acc[i][n][q]);
                                }
                            } else {
                                int d[4];
                                mma32(d, a, bq[2 * n], bq[2 * n + 1]);
#pragma unroll
                                for (int q = 0; q < 4; ++q) {
                                    const float p = (q < 2 ? wa0 : wb0) * (h ? dx[n][q & 1].y : dx[n][q & 1].x);
                                    acc[i][n][q] = fmaf(p, dotf(d[q]), acc[i][n][q]);
                                }
                            }
                        }
                    }
                }
            }
            // the next stage's weights (into the other buffer, read after the next barrier).  (Decoding them before
            // this stage's products instead was slower on the RTX 5070: a layer of 2048 / 3584 / 8192 tokens, IQ2_S
            // 5.2 / 9.5 / 17.5 ms -> 6.0 / 10.2 / 18.3.)
            if (s + 1 < NS) {
                put(raw, (s + 1) & 1);
                if (dec && s + 2 < NS) load_unit<WT>(unit(s + 2), sub(s + 2), raw);
            }
        }
        if (!on0) continue;
        if (GU) {
            // SwiGLU of feature (WR / 2) fb + 32 wf + 8 i + g (lane g of m tile i: rows 64 wf + 16 i + g and + 8), tile
            // rows nb0 + 8 n + 2 tig + c; H to int8 per row over the warp's 32 features (one 32-value half-block of the
            // down product's K)
            const int blk = WW * fb + wf;
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                if (n == 1 && !on1) break;
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    float h[4], am = 0.0f;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float gt = acc[i][n][c], up = acc[i][n][2 + c];
                        h[i] = gt / (1.0f + __expf(-gt)) * up;
                        am = fmaxf(am, fabsf(h[i]));
                    }
#pragma unroll
                    for (int o = 4; o < 32; o <<= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
                    const float inv = am > 0.0f ? 127.0f / am : 0.0f;
                    const int r = nb0 + 8 * n + 2 * tig + c;
                    if (r < nrows) {
                        uint8_t* o = out + (size_t) (row0 + r) * (10 * AB) + (blk >> 1) * AB;
                        const int hh = blk & 1;
#pragma unroll
                        for (int i = 0; i < 4; ++i) o[32 * hh + 8 * i + g] = (uint8_t) (int8_t) __float2int_rn(h[i] * inv);
                        if (g == 0) *(float*) (o + 64 + 4 * hh) = am / 127.0f;
                    }
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    if (n == 1 && !on1) break;
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const int r = nb0 + 8 * n + 2 * tig + (q & 1);
                        if (r < nrows) dm[(size_t) (row0 + r) * 2560 + rbase + 64 * wf + 16 * i + 8 * (q >> 1) + g] = acc[i][n][q];
                    }
                }
        }
    }
#endif
}

unsigned blocks(int64_t n, int per) { return (unsigned) ((n + per - 1) / per); }

// per device: whether every kernel here runs (sm_80+, device code in this build, fits), and their occupancy
struct DevInfo {
    bool done = false, ok = false;
    int sms = 0, occ = 1;
    bool sm86 = false;
};
std::mutex g_mu;
DevInfo g_dev[32];

#if !defined(STRATA_HIP_GFX906)
template <int T, bool GU, int WW> bool setup_ww(int& occ) {
    cudaFuncAttributes fa{};
    if (cudaFuncGetAttributes(&fa, native_kernel<T, GU, WW>) != cudaSuccess || fa.ptxVersion < 80) return false;
    if (cudaFuncSetAttribute(native_kernel<T, GU, WW>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int) smem_bytes(T, WW)) != cudaSuccess)
        return false;
    int o = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o, native_kernel<T, GU, WW>, THREADS, smem_bytes(T, WW)) !=
            cudaSuccess || o < 1)
        return false;
    occ = std::min(occ, o);
    return true;
}
template <int T, bool GU> bool setup_one(int& occ) { return setup_ww<T, GU, 4>(occ) && setup_ww<T, GU, 2>(occ); }
#endif

const DevInfo& dev_info() {
    int dev = 0;
    cudaGetDevice(&dev);
    std::lock_guard<std::mutex> lk(g_mu);
    DevInfo& d = g_dev[dev & 31];
    if (d.done) return d;
    d.done = true;
#if defined(STRATA_HIP_GFX906)
    // gfx906 reports compute capability 9.0 through HIP, but the mma.sync bodies above are CUDA sm_80+ only and
    // empty in a hipcc build: never available here (the prompt path keeps its own expert GEMMs)
    return d;
#else
    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    d.sm86 = major == 8 && minor == 6;
    cudaDeviceGetAttribute(&d.sms, cudaDevAttrMultiProcessorCount, dev);
    if (major < 8) return d;
    int occ = 1 << 20;
    d.ok = setup_one<T_IQ2_XXS, true>(occ) && setup_one<T_IQ2_XS, true>(occ) && setup_one<T_IQ2_S, true>(occ) &&
           setup_one<T_IQ3_XXS, true>(occ) && setup_one<T_IQ3_S, true>(occ) && setup_one<T_IQ4_XS, true>(occ) &&
           setup_one<T_Q2_0, false>(occ) && setup_one<T_IQ4_NL, false>(occ);
    d.occ = d.ok ? occ : 1;
    cudaGetLastError();
#endif
    return d;
}

bool gu_covered(int t) {
    return t == T_IQ2_XXS || t == T_IQ2_XS || t == T_IQ2_S || t == T_IQ3_XXS || t == T_IQ3_S || t == T_IQ4_XS;
}
bool d_covered(int t) { return t == T_Q2_0 || t == T_IQ4_NL; }

template <int T, bool GU>
void launch(int ww, unsigned grid, const Batch& b, const NativeGeom& g, const Tables& tb, const void* act,
            const int32_t* src, void* out, float* dm, cudaStream_t s) {
    const uint8_t* a = (const uint8_t*) act;
    uint8_t* o = (uint8_t*) out;
    if (ww == 4) native_kernel<T, GU, 4><<<grid, THREADS, smem_bytes(T, 4), s>>>(b, g, tb, a, src, o, dm);
    else native_kernel<T, GU, 2><<<grid, THREADS, smem_bytes(T, 2), s>>>(b, g, tb, a, src, o, dm);
}

// The work item's shape for a layer of `n` routed rows over `n_expert` experts: 128 routed rows when an expert has
// about one such tile (56 to 112 rows on average), else 64.  Measured on the RTX 5070 (tests/cuda/prefill_fused_iq_test,
// a layer of 2048 / 3584 / 8192 tokens, ms, 64 -> 128): IQ2_S 5.2 -> 6.7, 9.5 -> 9.3, 17.5 -> 19.5; IQ2_XXS 4.2 ->
// 5.8, 7.4 -> 7.6, 13.6 -> 16.5; IQ3_S 5.6 -> 6.6, 10.2 -> 8.8, 18.4 -> 19.0; IQ3_XXS 5.1 -> 6.3, 9.1 -> 8.3, 16.5 ->
// 17.5 (256 rows was slower still).  STRATA_PF_FUSED_TILE=64|128 forces one.
int pick_ww(int64_t n, int n_expert) {
    static const int forced = [] {
        const char* v = std::getenv("STRATA_PF_FUSED_TILE");
        const int t = v ? std::atoi(v) : 0;
        return t == 64 ? 4 : t == 128 ? 2 : 0;
    }();
    if (forced) return forced;
    const double avg = (double) n / std::max(n_expert, 1);
    return avg > 56.0 && avg <= 112.0 ? 2 : 4;
}


int requested_variant() {
    const char* v = std::getenv("STRATA_PF_NATIVE_VARIANT");
    if (!v) return 0;
    char* end = nullptr;
    const long mode = std::strtol(v, &end, 10);
    return end != v && *end == '\0' && mode >= 0 && mode <= 5 ? (int) mode : 0;
}

// Candidates are configured per actual specialization and device; the control retains its historical
// minimum-across-formats occupancy. No failed candidate setup can affect the original dispatch.
template <int T, bool GU, int WW, int NT, int AS, int PF>
NativeLabInfo specialization_info() {
    NativeLabInfo info;
    info.phase = GU ? 1 : 2;
    info.format = T;
    info.tile = tile_rows(WW, NT);
    info.weight_rows = weight_rows(WW);
    info.threads = NT;
    info.activation_stages = AS;
    info.prefetch = PF;
    info.dynamic_shared = smem_bytes(T, WW, NT, AS);
#if !defined(STRATA_HIP_GFX906)
    struct Cached { bool done = false; NativeLabInfo info; };
    static Cached cache[32];
    int dev = 0;
    ck(cudaGetDevice(&dev), "lab device");
    std::lock_guard<std::mutex> lk(g_mu);
    Cached& c = cache[dev & 31];
    if (c.done) return c.info;
    c.done = true;
    cudaFuncAttributes attr{};
    auto kernel = native_kernel<T, GU, WW, NT, AS, PF>;
    if (cudaFuncGetAttributes(&attr, kernel) == cudaSuccess && attr.ptxVersion >= 80 &&
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int) info.dynamic_shared) == cudaSuccess) {
        int occ = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kernel, NT, info.dynamic_shared) == cudaSuccess)
            info.occupancy = occ;
        info.registers = attr.numRegs;
    }
    cudaGetLastError(); // unavailable experimental specialization falls back to control
    c.info = info;
#endif
    return info;
}

template <int T, bool GU>
NativeLabInfo phase_info(int ww, int variant) {
    NativeLabInfo info;
    const DevInfo& d = dev_info();
    if (!d.sm86 || variant < 0 || variant > 4) variant = 0;
    int base = 2;
    if (const char* v = std::getenv("STRATA_PF_NATIVE_BASE")) {
        const int requested = std::atoi(v);
        if (requested == 0 || requested == 1 || requested == 2 || (variant == 4 && requested == 3)) base = requested;
    }
    int prefetch_base = 2;
    if (const char* v = std::getenv("STRATA_PF_NATIVE_PREFETCH_BASE")) {
        const int requested = std::atoi(v);
        if (requested >= 0 && requested <= 2) prefetch_base = requested;
    }
    const bool control_layout = (variant == 3 || variant == 4) &&
                                (base == 0 || (base == 3 && prefetch_base == 0));
    const int stages = control_layout || variant == 1 || ((variant == 3 || variant == 4) &&
                       (base == 1 || (base == 3 && prefetch_base == 1))) ? 4 : 3;
    const int prefetch = GU && (variant == 3 || (variant == 4 && base == 3)) ? 4 : 2;
    if (variant != 0) {
        if (control_layout && prefetch == 4)
            info = ww == 4 ? specialization_info<T, GU, 4, 512, 4, 4>()
                           : specialization_info<T, GU, 2, 512, 4, 4>();
        else if (control_layout)
            info = ww == 4 ? specialization_info<T, GU, 4, 512, 4, 2>()
                           : specialization_info<T, GU, 2, 512, 4, 2>();
        else if (stages == 4 && prefetch == 4)
            info = ww == 4 ? specialization_info<T, GU, 2, 256, 4, 4>()
                           : specialization_info<T, GU, 1, 256, 4, 4>();
        else if (stages == 4)
            info = ww == 4 ? specialization_info<T, GU, 2, 256, 4, 2>()
                           : specialization_info<T, GU, 1, 256, 4, 2>();
        else if (prefetch == 4)
            info = ww == 4 ? specialization_info<T, GU, 2, 256, 3, 4>()
                           : specialization_info<T, GU, 1, 256, 3, 4>();
        else
            info = ww == 4 ? specialization_info<T, GU, 2, 256, 3, 2>()
                           : specialization_info<T, GU, 1, 256, 3, 2>();
    }
    if (variant == 0 || info.occupancy < 1) {
        variant = 0;
        info = ww == 4 ? specialization_info<T, GU, 4, 512, 4, 2>()
                       : specialization_info<T, GU, 2, 512, 4, 2>();
    }
    info.variant = variant;
    info.sms = d.sms;
    return info;
}

template <int T, bool GU, int WW, int AS, int PF, int NT = 256>
void launch_candidate(unsigned grid, const Batch& b, const NativeGeom& g, const Tables& tb,
                      const void* act, const int32_t* src, void* out, float* dm, cudaStream_t stream) {
    native_kernel<T, GU, WW, NT, AS, PF><<<grid, NT, smem_bytes(T, WW, NT, AS), stream>>>(
        b, g, tb, (const uint8_t*) act, src, (uint8_t*) out, dm);
}

template <int T, bool GU>
void phase_launch(const NativeLabInfo& i, const Batch& b, const NativeGeom& g, const Tables& tb,
                  const void* act, const int32_t* src, void* out, float* dm, cudaStream_t stream) {
    if (i.variant == 0) {
        launch<T, GU>(i.tile == 64 ? 4 : 2, i.grid, b, g, tb, act, src, out, dm, stream);
    } else if (i.threads == 512 && i.prefetch == 4) {
        if (i.tile == 64) launch_candidate<T, GU, 4, 4, 4, 512>(i.grid, b, g, tb, act, src, out, dm, stream);
        else launch_candidate<T, GU, 2, 4, 4, 512>(i.grid, b, g, tb, act, src, out, dm, stream);
    } else if (i.threads == 512) {
        launch<T, GU>(i.tile == 64 ? 4 : 2, i.grid, b, g, tb, act, src, out, dm, stream);
    } else if (i.activation_stages == 4 && i.prefetch == 4) {
        if (i.tile == 64) launch_candidate<T, GU, 2, 4, 4>(i.grid, b, g, tb, act, src, out, dm, stream);
        else launch_candidate<T, GU, 1, 4, 4>(i.grid, b, g, tb, act, src, out, dm, stream);
    } else if (i.activation_stages == 4) {
        if (i.tile == 64) launch_candidate<T, GU, 2, 4, 2>(i.grid, b, g, tb, act, src, out, dm, stream);
        else launch_candidate<T, GU, 1, 4, 2>(i.grid, b, g, tb, act, src, out, dm, stream);
    } else if (i.prefetch == 4) {
        if (i.tile == 64) launch_candidate<T, GU, 2, 3, 4>(i.grid, b, g, tb, act, src, out, dm, stream);
        else launch_candidate<T, GU, 1, 3, 4>(i.grid, b, g, tb, act, src, out, dm, stream);
    } else {
        if (i.tile == 64) launch_candidate<T, GU, 2, 3, 2>(i.grid, b, g, tb, act, src, out, dm, stream);
        else launch_candidate<T, GU, 1, 3, 2>(i.grid, b, g, tb, act, src, out, dm, stream);
    }
}

struct ProfileRecord {
    cudaEvent_t begin{}, end{}, ready{};
    int32_t* counts = nullptr;
    NativeLabInfo info;
    int e0 = 0, e1 = 0, n_expert = 0;
    int64_t rows = 0;
};
std::mutex g_profile_mu;
std::vector<ProfileRecord> g_profile;
constexpr size_t PROFILE_RECORD_CAPACITY = 65536;
constexpr size_t PROFILE_COUNT_CAPACITY = PROFILE_RECORD_CAPACITY * 512; // diagnostic-only pinned pool
std::vector<ProfileRecord> g_profile_slots;
size_t g_profile_slot_used = 0;
int32_t* g_profile_counts = nullptr;
size_t g_profile_count_used = 0;

bool profiling_requested() {
    const char* v = std::getenv("STRATA_PF_NATIVE_PROFILE");
    return v && v[0] == '1';
}

void dispatch_phase(const NativeLabInfo& info, const Batch& b, const NativeGeom& g, int n_expert, int64_t n,
                    const Tables& tb, const void* xa, const int32_t* src, void* ha, float* dm, cudaStream_t stream) {
    const bool profile = profiling_requested();
    ProfileRecord record;
    if (profile) {
        record.info = info;
        record.e0 = b.e0; record.e1 = b.e1; record.n_expert = n_expert; record.rows = n;
        {
            std::lock_guard<std::mutex> lk(g_profile_mu);
            if (!g_profile_counts || g_profile_slot_used >= PROFILE_RECORD_CAPACITY ||
                g_profile_count_used + (size_t) (b.e1 - b.e0) > PROFILE_COUNT_CAPACITY) {
                std::fprintf(stderr, "native expert profiling pool unavailable/exhausted; call native_profile_prepare before prefill and collect after existing synchronization\n");
                std::exit(1);
            }
            const ProfileRecord& slot = g_profile_slots[g_profile_slot_used++];
            record.begin = slot.begin; record.end = slot.end; record.ready = slot.ready;
            record.counts = g_profile_counts + g_profile_count_used;
            g_profile_count_used += b.e1 - b.e0;
        }
        ck(cudaEventRecord(record.begin, stream), "profile begin record");
    }
    if (info.phase == 1) {
        switch (g.gu_type) {
            case T_IQ2_XXS: phase_launch<T_IQ2_XXS, true>(info, b, g, tb, xa, src, ha, nullptr, stream); break;
            case T_IQ2_XS: phase_launch<T_IQ2_XS, true>(info, b, g, tb, xa, src, ha, nullptr, stream); break;
            case T_IQ2_S: phase_launch<T_IQ2_S, true>(info, b, g, tb, xa, src, ha, nullptr, stream); break;
            case T_IQ3_XXS: phase_launch<T_IQ3_XXS, true>(info, b, g, tb, xa, src, ha, nullptr, stream); break;
            case T_IQ3_S: phase_launch<T_IQ3_S, true>(info, b, g, tb, xa, src, ha, nullptr, stream); break;
            default: phase_launch<T_IQ4_XS, true>(info, b, g, tb, xa, src, ha, nullptr, stream); break;
        }
    } else {
        if (g.d_type == T_Q2_0) phase_launch<T_Q2_0, false>(info, b, g, tb, ha, src, nullptr, dm, stream);
        else phase_launch<T_IQ4_NL, false>(info, b, g, tb, ha, src, nullptr, dm, stream);
    }
    if (profile) {
        ck(cudaEventRecord(record.end, stream), "profile end record");
        ck(cudaMemcpyAsync(record.counts, tb.cnt + b.e0, (size_t) (b.e1 - b.e0) * sizeof(int32_t),
                           cudaMemcpyDeviceToHost, stream), "profile counts copy");
        ck(cudaEventRecord(record.ready, stream), "profile ready record");
        std::lock_guard<std::mutex> lk(g_profile_mu);
        g_profile.push_back(record);
    }
}

}  // namespace

bool native_supported(int gu_type, int d_type) {
    static const bool off = [] {   // STRATA_PF_FUSED_NATIVE=0: the native packs keep MMQ under STRATA_PF_FUSED=1 (A/B)
        const char* v = std::getenv("STRATA_PF_FUSED_NATIVE");
        return v != nullptr && v[0] == '0';
    }();
    const bool supported = !off && requested() && gu_covered(gu_type) && d_covered(d_type) && dev_info().ok;
    if (supported && profiling_requested()) native_profile_prepare();
    return supported;   // opt-in (=1)
}

void quantize_act_native(const float* x, int64_t rows, int64_t cols, void* xa, void* stream) {
    if (rows <= 0) return;
    const int64_t nblk = rows * (cols / 64);
    quant_act_nat_kernel<<<blocks(nblk, 8), 256, 0, (cudaStream_t) stream>>>(x, nblk, (uint8_t*) xa);
    ck(cudaGetLastError(), "quantize_act_native");
}


NativeLabInfo native_lab_info(const Batch& b, const NativeGeom& g, int n_expert, int64_t n,
                              int phase, int variant, int tile) {
    const int ww = tile == 64 ? 4 : tile == 128 ? 2 : pick_ww(n, n_expert);
    const int requested = variant;
    if (variant == 5) {
        const int64_t experts = std::max(n_expert, 1);
        const int row_class = n <= 56 * experts ? 0 : n <= 112 * experts ? 1 : 2;
        variant = dev_info().sm86 ? native_lab_dispatch::mode(phase, phase == 1 ? g.gu_type : g.d_type, row_class) : 0;
    }
    NativeLabInfo info;
    if (phase == 1) {
        switch (g.gu_type) {
            case T_IQ2_XXS: info = phase_info<T_IQ2_XXS, true>(ww, variant); break;
            case T_IQ2_XS: info = phase_info<T_IQ2_XS, true>(ww, variant); break;
            case T_IQ2_S: info = phase_info<T_IQ2_S, true>(ww, variant); break;
            case T_IQ3_XXS: info = phase_info<T_IQ3_XXS, true>(ww, variant); break;
            case T_IQ3_S: info = phase_info<T_IQ3_S, true>(ww, variant); break;
            default: info = phase_info<T_IQ4_XS, true>(ww, variant); break;
        }
    } else {
        info = g.d_type == T_Q2_0 ? phase_info<T_Q2_0, false>(ww, variant)
                                 : phase_info<T_IQ4_NL, false>(ww, variant);
    }
    info.requested_variant = requested;
    const int64_t tiles = (n + kTileRows - 1) / kTileRows + (b.e1 - b.e0);
    const int occ = info.variant ? info.occupancy : dev_info().occ;
    const int multiple = info.variant == 4 ? 2 : 1;
    info.grid = (unsigned) std::min<int64_t>(tiles * ((phase == 1 ? 1280 : 2560) / info.weight_rows),
                                           (int64_t) info.sms * occ * multiple);
    return info;
}

void experts_native_lab(const Batch& b, const NativeGeom& g, int n_expert, int64_t n, const void* scratch,
                        const void* xa, const int32_t* src, void* ha, float* dm, void* stream,
                        int phase, int variant, int tile) {
    if (b.e1 <= b.e0 || n <= 0) return;
    if (!dev_info().ok || !gu_covered(g.gu_type) || !d_covered(g.d_type) || phase < 0 || phase > 2) {
        std::fprintf(stderr, "prefill fused experts (native): types %d / %d or phase %d are not covered here\n",
                     g.gu_type, g.d_type, phase);
        std::exit(1);
    }
    const Tables tb = tables(const_cast<void*>(scratch), n_expert);
    const cudaStream_t s = (cudaStream_t) stream;
    if (phase != 2) dispatch_phase(native_lab_info(b, g, n_expert, n, 1, variant, tile),
                                  b, g, n_expert, n, tb, xa, src, ha, dm, s);
    if (phase != 1) dispatch_phase(native_lab_info(b, g, n_expert, n, 2, variant, tile),
                                  b, g, n_expert, n, tb, xa, src, ha, dm, s);
    ck(cudaGetLastError(), "experts_native");
}

static void experts_native_control(const Batch& b, const NativeGeom& g, int n_expert, int64_t n, const void* scratch, const void* xa,
                    const int32_t* src, void* ha, float* dm, void* stream) {
    if (b.e1 <= b.e0 || n <= 0) return;
    const DevInfo& d = dev_info();
    if (!d.ok || !gu_covered(g.gu_type) || !d_covered(g.d_type)) {
        std::fprintf(stderr, "prefill fused experts (native): types %d / %d are not covered here\n", g.gu_type, g.d_type);
        std::exit(1);
    }
    const cudaStream_t s = (cudaStream_t) stream;
    const Tables tb = tables(const_cast<void*>(scratch), n_expert);
    const int ww = pick_ww(n, n_expert);
    // the most 64-row tiles the batch can have (every row in it, plus a partial tile per expert) - the items of the
    // ones inside a larger item end at once
    const int64_t tiles = (n + kTileRows - 1) / kTileRows + (b.e1 - b.e0);
    const unsigned g_gu = (unsigned) std::min<int64_t>(tiles * (1280 / weight_rows(ww)), (int64_t) d.sms * d.occ);
    const unsigned g_d = (unsigned) std::min<int64_t>(tiles * (2560 / weight_rows(ww)), (int64_t) d.sms * d.occ);
    switch (g.gu_type) {
        case T_IQ2_XXS: launch<T_IQ2_XXS, true>(ww, g_gu, b, g, tb, xa, src, ha, nullptr, s); break;
        case T_IQ2_XS: launch<T_IQ2_XS, true>(ww, g_gu, b, g, tb, xa, src, ha, nullptr, s); break;
        case T_IQ2_S: launch<T_IQ2_S, true>(ww, g_gu, b, g, tb, xa, src, ha, nullptr, s); break;
        case T_IQ3_XXS: launch<T_IQ3_XXS, true>(ww, g_gu, b, g, tb, xa, src, ha, nullptr, s); break;
        case T_IQ3_S: launch<T_IQ3_S, true>(ww, g_gu, b, g, tb, xa, src, ha, nullptr, s); break;
        default: launch<T_IQ4_XS, true>(ww, g_gu, b, g, tb, xa, src, ha, nullptr, s); break;
    }
    if (g.d_type == T_Q2_0) launch<T_Q2_0, false>(ww, g_d, b, g, tb, ha, src, nullptr, dm, s);
    else launch<T_IQ4_NL, false>(ww, g_d, b, g, tb, ha, src, nullptr, dm, s);
    ck(cudaGetLastError(), "experts_native");
}


void experts_native(const Batch& b, const NativeGeom& g, int n_expert, int64_t n, const void* scratch, const void* xa,
                    const int32_t* src, void* ha, float* dm, void* stream) {
    const int variant = requested_variant();
    if (variant == 0 && !profiling_requested()) {
        experts_native_control(b, g, n_expert, n, scratch, xa, src, ha, dm, stream);
        return;
    }
    experts_native_lab(b, g, n_expert, n, scratch, xa, src, ha, dm, stream, 0, variant);
}

void native_profile_prepare() {
    if (!profiling_requested()) return;
    std::lock_guard<std::mutex> lk(g_profile_mu);
    if (g_profile_counts) return;
    ck(cudaMallocHost((void**) &g_profile_counts, PROFILE_COUNT_CAPACITY * sizeof(int32_t)), "profile pool allocate");
    g_profile_slots.resize(PROFILE_RECORD_CAPACITY);
    g_profile.reserve(PROFILE_RECORD_CAPACITY);
    for (ProfileRecord& slot : g_profile_slots) {
        ck(cudaEventCreate(&slot.begin), "profile begin create");
        ck(cudaEventCreate(&slot.end), "profile end create");
        ck(cudaEventCreateWithFlags(&slot.ready, cudaEventDisableTiming), "profile ready create");
    }
}

void native_profile_collect() {
    std::lock_guard<std::mutex> lk(g_profile_mu);
    auto it = g_profile.begin();
    while (it != g_profile.end()) {
        const cudaError_t status = cudaEventQuery(it->ready);
        if (status == cudaErrorNotReady) break;
        ck(status, "profile query");
        float ms = 0.0f;
        ck(cudaEventElapsedTime(&ms, it->begin, it->end), "profile elapsed");
        const NativeLabInfo& i = it->info;
        int histogram[4] = {0, 0, 0, 0};
        int64_t routed = 0, tile_count = 0, partial_tiles = 0;
        for (int e = 0; e < it->e1 - it->e0; ++e) {
            const int rows = it->counts[e];
            ++histogram[rows == 0 ? 0 : rows < 64 ? 1 : rows < 128 ? 2 : 3];
            routed += rows;
            tile_count += (rows + i.tile - 1) / i.tile;
            partial_tiles += rows % i.tile != 0;
        }
        std::fprintf(stderr, "native_expert_profile {\"phase\":\"%s\",\"format\":%d,\"requested_variant\":%d,\"variant\":%d,"
                     "\"tile\":%d,\"weight_rows\":%d,\"threads\":%d,\"stages\":%d,\"prefetch\":%d,"
                     "\"grid\":%u,\"registers\":%d,\"dynamic_shared\":%zu,\"occupancy\":%d,"
                     "\"batch_start\":%d,\"batch_experts\":%d,\"total_experts\":%d,\"total_routed_rows\":%lld,"
                     "\"batch_routed_rows\":%lld,\"tile_count\":%lld,\"partial_tiles\":%lld,"
                     "\"row_histogram_0_1to63_64to127_128plus\":[%d,%d,%d,%d],\"elapsed_ms\":%.6f,\"expert_rows\":[",
                     i.phase == 1 ? "gate_up_epilogue" : "down", i.format, i.requested_variant, i.variant, i.tile, i.weight_rows,
                     i.threads, i.activation_stages, i.prefetch, i.grid, i.registers, i.dynamic_shared, i.occupancy,
                     it->e0, it->e1 - it->e0, it->n_expert, (long long) it->rows, (long long) routed,
                     (long long) tile_count, (long long) partial_tiles,
                     histogram[0], histogram[1], histogram[2], histogram[3], ms);
        for (int e = 0; e < it->e1 - it->e0; ++e) std::fprintf(stderr, "%s%d", e ? "," : "", it->counts[e]);
        std::fprintf(stderr, "]}\n");
        ++it;
    }
    g_profile.erase(g_profile.begin(), it);
    if (g_profile.empty()) { g_profile_count_used = 0; g_profile_slot_used = 0; }
}

}  // namespace strata::prefill::fused
