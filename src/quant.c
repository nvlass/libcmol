/*
 * quant.c — quantisation kernels and SIMD dispatch
 * Included by src/cmol.c (unity build); do not compile standalone.
 *
 * Supported quantised formats:
 *   Q8_0 : 32 values / block,  34 bytes/block  (float16 scale + 32 × int8)
 *   Q4_K : 256 values / block, 144 bytes/block (Q4_K_M super-block)
 *   Q6_K : 256 values / block, 210 bytes/block (Q6_K super-block)
 *   F16  : passthrough, no block structure
 *   F32  : passthrough, no block structure
 *
 * SIMD dispatch:
 *   cmol_kernels_select() detects CPU features at run time (via platform.c)
 *   and returns a kernel table pointing to the fastest available matmul.
 *   Each SIMD kernel is annotated with __attribute__((target(...))) so it
 *   can be compiled in the same translation unit as the scalar fallback even
 *   when the global -march flag does not enable the target ISA.
 */

#include "quant.h"

#include <string.h>  /* memcpy */
#include <stdint.h>

/* SIMD headers — guarded so the unity build compiles everywhere */
#if defined(__x86_64__) || defined(_M_X64)
#  include <immintrin.h>   /* AVX2, AVX-512 */
#endif

#if defined(__ARM_NEON) || defined(__aarch64__)
#  include <arm_neon.h>
#endif

/* =========================================================================
 * Float16 → Float32 (scalar, portable)
 * ====================================================================== */

static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1FU;
    uint32_t mant = h & 0x3FFU;
    uint32_t f32;

    if (exp == 0u) {
        if (mant == 0u) {
            f32 = sign;                        /* ±0 */
        } else {
            /* Subnormal: normalise into the f32 range */
            exp = 1u;
            while (!(mant & 0x400U)) { mant <<= 1; exp--; }
            mant &= 0x3FFU;
            f32 = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
        }
    } else if (exp == 31u) {
        f32 = sign | (0xFFu << 23) | (mant << 13); /* Inf / NaN */
    } else {
        f32 = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
    }

    float r;
    memcpy(&r, &f32, sizeof r);
    return r;
}

/* =========================================================================
 * Quantised block layouts
 *
 * __attribute__((packed)) ensures no implicit padding between fields.
 * The compile-time assertions below verify the expected sizes.
 * ====================================================================== */

#if defined(__GNUC__) || defined(__clang__)
#  define CMOL_PACKED __attribute__((packed))
#else
#  define CMOL_PACKED  /* MSVC: use #pragma pack(push,1) around structs */
#endif

/* Q8_0 — 34 bytes */
typedef struct CMOL_PACKED {
    uint16_t d;       /* float16 scale                  */
    int8_t   qs[32];  /* 32 × int8 quantised values     */
} q8_0_block_t;

/* Q4_K — 144 bytes (super-block of 8 × 32-value sub-blocks) */
typedef struct CMOL_PACKED {
    uint16_t d;          /* float16 super-block scale for scales  */
    uint16_t dmin;       /* float16 super-block scale for mins    */
    uint8_t  scales[12]; /* 8×6-bit scale values + 8×6-bit mins  */
    uint8_t  qs[128];    /* 256 × 4-bit quantised values          */
} q4_k_block_t;

/* Q6_K — 210 bytes (super-block of 16 × 16-value sub-blocks) */
typedef struct CMOL_PACKED {
    uint8_t  ql[128];    /* lower 4 bits of each of 256 values   */
    uint8_t  qh[64];     /* upper 2 bits of each of 256 values   */
    int8_t   scales[16]; /* per-16-value int8 scales              */
    uint16_t d;          /* float16 super-block scale             */
} q6_k_block_t;

/* Compile-time size checks */
typedef char chk_q8_0[(sizeof(q8_0_block_t)  == 34)  ? 1 : -1];
typedef char chk_q4_k[(sizeof(q4_k_block_t)  == 144) ? 1 : -1];
typedef char chk_q6_k[(sizeof(q6_k_block_t)  == 210) ? 1 : -1];

/* =========================================================================
 * Q4_K scale/min extraction
 *
 * The 12 bytes of `scales` pack 8 × 6-bit scale values and 8 × 6-bit min
 * values using the following layout (j = sub-block index 0..7):
 *
 *   j = 0..3:  sc = scales[j]   & 0x3F
 *              mn = scales[j+4] & 0x3F
 *   j = 4..7:  sc = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4)
 *              mn = (scales[j+4] >> 4)   | ((scales[j-0] >> 6) << 4)
 * ====================================================================== */

static inline void q4k_get_scale_min(int j, const uint8_t *s,
                                      uint8_t *sc, uint8_t *mn) {
    if (j < 4) {
        *sc = s[j]   & 0x3F;
        *mn = s[j+4] & 0x3F;
    } else {
        *sc = (s[j+4] & 0x0F) | ((s[j-4] >> 6) << 4);
        *mn = (s[j+4] >> 4)   | ((s[j-0] >> 6) << 4);
    }
}

/* =========================================================================
 * Per-block dequantisation helpers
 *
 * Each writes exactly block_size floats to dst (32 for Q8_0, 256 for Q4/Q6).
 * These are small enough that the compiler inlines / auto-vectorises them
 * in the SIMD kernel functions.
 * ====================================================================== */

static void dequant_block_q8_0(const q8_0_block_t *b, float *dst) {
    float d = f16_to_f32(b->d);
    int i;
    for (i = 0; i < 32; i++) dst[i] = d * (float)b->qs[i];
}

static void dequant_block_q4_k(const q4_k_block_t *b, float *dst) {
    float d    = f16_to_f32(b->d);
    float dmin = f16_to_f32(b->dmin);
    int sub, k;

    for (sub = 0; sub < 8; sub++) {
        uint8_t sc, mn;
        q4k_get_scale_min(sub, b->scales, &sc, &mn);
        float db = d    * (float)sc;
        float mb = dmin * (float)mn;

        const uint8_t *q = b->qs + sub * 16;
        for (k = 0; k < 16; k++) {
            dst[sub * 32 + k * 2 + 0] = db * (float)(q[k] & 0xF) - mb;
            dst[sub * 32 + k * 2 + 1] = db * (float)(q[k] >> 4)  - mb;
        }
    }
}

static void dequant_block_q6_k(const q6_k_block_t *b, float *dst) {
    float d = f16_to_f32(b->d);
    int i;

    for (i = 0; i < 256; i++) {
        uint8_t lo = (b->ql[i / 2] >> ((i & 1) * 4)) & 0xFu;
        uint8_t hi = (b->qh[i / 4] >> ((i & 3) * 2)) & 0x3u;
        int8_t  q  = (int8_t)((int)(lo | ((unsigned)hi << 4)) - 32);
        dst[i] = d * (float)b->scales[i / 16] * (float)q;
    }
}

/* =========================================================================
 * cmol_dequant_row
 *
 * Public entry point: dequantise `n` values from a packed quantised row.
 * `n` must be a multiple of the format's block size.
 * ====================================================================== */

void cmol_dequant_row(const void *src, float *dst, int n, cmol_dtype_t dtype) {
    if (!src || !dst || n <= 0) return;

    switch (dtype) {

    case CMOL_DTYPE_F32: {
        memcpy(dst, src, (size_t)n * sizeof(float));
        break;
    }

    case CMOL_DTYPE_F16: {
        const uint16_t *s = (const uint16_t *)src;
        int i;
        for (i = 0; i < n; i++) dst[i] = f16_to_f32(s[i]);
        break;
    }

    case CMOL_DTYPE_Q8_0: {
        const q8_0_block_t *b = (const q8_0_block_t *)src;
        int bi, nb = n / 32;
        for (bi = 0; bi < nb; bi++) dequant_block_q8_0(&b[bi], dst + bi * 32);
        break;
    }

    case CMOL_DTYPE_Q4_K: {
        const q4_k_block_t *b = (const q4_k_block_t *)src;
        int bi, nb = n / 256;
        for (bi = 0; bi < nb; bi++) dequant_block_q4_k(&b[bi], dst + bi * 256);
        break;
    }

    case CMOL_DTYPE_Q6_K: {
        const q6_k_block_t *b = (const q6_k_block_t *)src;
        int bi, nb = n / 256;
        for (bi = 0; bi < nb; bi++) dequant_block_q6_k(&b[bi], dst + bi * 256);
        break;
    }

    default:
        break;
    }
}

/* =========================================================================
 * Block-level dispatch helpers (used by all matmul kernels)
 *
 * Returns the block size in values and populates `bsz` with the block
 * size in bytes.  Returns 0 for unknown types.
 * ====================================================================== */

static inline int block_params(cmol_dtype_t dtype, size_t *bsz) {
    switch (dtype) {
    case CMOL_DTYPE_F32:  *bsz = sizeof(float);          return 1;
    case CMOL_DTYPE_F16:  *bsz = sizeof(uint16_t);       return 1;
    case CMOL_DTYPE_Q8_0: *bsz = sizeof(q8_0_block_t);   return 32;
    case CMOL_DTYPE_Q4_K: *bsz = sizeof(q4_k_block_t);   return 256;
    case CMOL_DTYPE_Q6_K: *bsz = sizeof(q6_k_block_t);   return 256;
    default:              *bsz = 0;                       return 0;
    }
}

/* Dequantise one block at address `blk` of `dtype` into `tmp`.
 * Assumes `blk` is correctly aligned (it comes from the GGUF mmap). */
static inline void dequant_block(const void *blk, float *tmp, cmol_dtype_t dtype) {
    switch (dtype) {
    case CMOL_DTYPE_Q8_0: dequant_block_q8_0((const q8_0_block_t *)blk, tmp); break;
    case CMOL_DTYPE_Q4_K: dequant_block_q4_k((const q4_k_block_t *)blk, tmp); break;
    case CMOL_DTYPE_Q6_K: dequant_block_q6_k((const q6_k_block_t *)blk, tmp); break;
    default: break;
    }
}

/* =========================================================================
 * Scalar matmul
 *
 *   out[m × n] = A[m × k]  ·  B[k × n]
 *
 * A is a quantised weight matrix (row-major, packed blocks).
 * B is a float32 activation matrix (row-major: B[ki][ni] = B[ki*n + ni]).
 *
 * Processed one block of A at a time — no k-sized scratch buffer on the
 * heap; the local `tmp[256]` covers the largest block size.
 * ====================================================================== */

void cmol_matmul_scalar(float *out, const void *a, const float *b,
                         int m, int k, int n, cmol_dtype_t dtype) {
    size_t bsz;
    int    bs = block_params(dtype, &bsz);
    if (!bs) return;

    const uint8_t *abytes   = (const uint8_t *)a;
    size_t         row_bytes = (size_t)(k / bs) * bsz;
    float          tmp[256]; /* max block size */
    int mi, ni, bi;

    for (mi = 0; mi < m; mi++) {
        const uint8_t *row_a = abytes + (size_t)mi * row_bytes;

        for (ni = 0; ni < n; ni++) {
            float acc = 0.0f;

            if (dtype == CMOL_DTYPE_F32) {
                const float *af = (const float *)row_a;
                int ki;
                for (ki = 0; ki < k; ki++) acc += af[ki] * b[ki * n + ni];

            } else if (dtype == CMOL_DTYPE_F16) {
                const uint16_t *ah = (const uint16_t *)row_a;
                int ki;
                for (ki = 0; ki < k; ki++)
                    acc += f16_to_f32(ah[ki]) * b[ki * n + ni];

            } else {
                int nb = k / bs;
                for (bi = 0; bi < nb; bi++) {
                    int   base = bi * bs;
                    int   j;
                    dequant_block(row_a + (size_t)bi * bsz, tmp, dtype);
                    for (j = 0; j < bs; j++)
                        acc += tmp[j] * b[(base + j) * n + ni];
                }
            }

            out[mi * n + ni] = acc;
        }
    }
}

/* =========================================================================
 * AVX2 matmul  (x86-64, compiled with target("avx2,fma"))
 *
 * Dequantises one block at a time into a 256-element float buffer, then
 * accumulates the dot product using 256-bit FMA instructions (8 f32/cycle).
 *
 * For n == 1 (the common autoregressive inference case) B is loaded with
 * a contiguous 256-bit load; for n > 1 we scalar-gather into a small
 * local array to keep the inner loop clean.
 * ====================================================================== */

#if defined(__x86_64__) || defined(_M_X64)

/* Horizontal sum of a __m256 */
static __attribute__((target("avx,avx2"))) inline float hsum_m256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

__attribute__((target("avx2,fma")))
void cmol_matmul_avx2(float *out, const void *a, const float *b,
                       int m, int k, int n, cmol_dtype_t dtype) {
    size_t bsz;
    int    bs = block_params(dtype, &bsz);
    if (!bs) return;

    const uint8_t *abytes   = (const uint8_t *)a;
    size_t         row_bytes = (size_t)(k / bs) * bsz;
    float          tmp[256];
    int mi, ni, bi;

    for (mi = 0; mi < m; mi++) {
        const uint8_t *row_a = abytes + (size_t)mi * row_bytes;

        for (ni = 0; ni < n; ni++) {
            __m256 acc = _mm256_setzero_ps();

            if (dtype == CMOL_DTYPE_F32) {
                /* F32: pure AVX2 dot product, 8 floats/iter */
                const float *af = (const float *)row_a;
                int ki;
                for (ki = 0; ki + 7 < k; ki += 8) {
                    __m256 av = _mm256_loadu_ps(af + ki);
                    __m256 bv = (n == 1)
                        ? _mm256_loadu_ps(b + ki)
                        : _mm256_set_ps(b[(ki+7)*n+ni], b[(ki+6)*n+ni],
                                        b[(ki+5)*n+ni], b[(ki+4)*n+ni],
                                        b[(ki+3)*n+ni], b[(ki+2)*n+ni],
                                        b[(ki+1)*n+ni], b[(ki+0)*n+ni]);
                    acc = _mm256_fmadd_ps(av, bv, acc);
                }
                float tail = hsum_m256(acc);
                /* scalar tail for k not a multiple of 8 */
                for (; ki < k; ki++) tail += af[ki] * b[ki * n + ni];
                out[mi * n + ni] = tail;
                continue; /* next ni */

            } else if (dtype == CMOL_DTYPE_F16) {
                /* F16: scalar path — rarely a bottleneck */
                const uint16_t *ah = (const uint16_t *)row_a;
                float s = 0.0f;
                int ki;
                for (ki = 0; ki < k; ki++)
                    s += f16_to_f32(ah[ki]) * b[ki * n + ni];
                out[mi * n + ni] = s;
                continue;
            }

            /* Quantised path: dequant one block, AVX2 dot product */
            int nb = k / bs;
            for (bi = 0; bi < nb; bi++) {
                int base = bi * bs;
                int j;
                dequant_block(row_a + (size_t)bi * bsz, tmp, dtype);

                for (j = 0; j + 7 < bs; j += 8) {
                    __m256 av = _mm256_loadu_ps(tmp + j);
                    __m256 bv;
                    if (n == 1) {
                        bv = _mm256_loadu_ps(b + base + j);
                    } else {
                        bv = _mm256_set_ps(
                            b[(base+j+7)*n+ni], b[(base+j+6)*n+ni],
                            b[(base+j+5)*n+ni], b[(base+j+4)*n+ni],
                            b[(base+j+3)*n+ni], b[(base+j+2)*n+ni],
                            b[(base+j+1)*n+ni], b[(base+j+0)*n+ni]);
                    }
                    acc = _mm256_fmadd_ps(av, bv, acc);
                }
                /* tail for bs not a multiple of 8 (Q8_0: bs=32, always ×8) */
                for (; j < bs; j++)
                    acc = _mm256_add_ps(acc,
                          _mm256_set1_ps(tmp[j] * b[(base+j)*n+ni]));
            }

            out[mi * n + ni] = hsum_m256(acc);
        }
    }
}

/* =========================================================================
 * AVX-512 matmul  (x86-64, target("avx512f"))
 *
 * Same structure as AVX2 but uses 512-bit ZMM registers (16 f32/cycle).
 * ====================================================================== */

__attribute__((target("avx512f")))
void cmol_matmul_avx512(float *out, const void *a, const float *b,
                          int m, int k, int n, cmol_dtype_t dtype) {
    size_t bsz;
    int    bs = block_params(dtype, &bsz);
    if (!bs) return;

    const uint8_t *abytes   = (const uint8_t *)a;
    size_t         row_bytes = (size_t)(k / bs) * bsz;
    float          tmp[256];
    int mi, ni, bi;

    for (mi = 0; mi < m; mi++) {
        const uint8_t *row_a = abytes + (size_t)mi * row_bytes;

        for (ni = 0; ni < n; ni++) {
            __m512 acc = _mm512_setzero_ps();

            if (dtype == CMOL_DTYPE_F32) {
                const float *af = (const float *)row_a;
                int ki;
                for (ki = 0; ki + 15 < k; ki += 16) {
                    __m512 av = _mm512_loadu_ps(af + ki);
                    __m512 bv;
                    if (n == 1) {
                        bv = _mm512_loadu_ps(b + ki);
                    } else {
                        float vals[16];
                        int jj;
                        for (jj = 0; jj < 16; jj++) vals[jj] = b[(ki+jj)*n+ni];
                        bv = _mm512_loadu_ps(vals);
                    }
                    acc = _mm512_fmadd_ps(av, bv, acc);
                }
                float tail = _mm512_reduce_add_ps(acc);
                for (; ki < k; ki++) tail += af[ki] * b[ki * n + ni];
                out[mi * n + ni] = tail;
                continue;

            } else if (dtype == CMOL_DTYPE_F16) {
                const uint16_t *ah = (const uint16_t *)row_a;
                float s = 0.0f;
                int ki;
                for (ki = 0; ki < k; ki++)
                    s += f16_to_f32(ah[ki]) * b[ki * n + ni];
                out[mi * n + ni] = s;
                continue;
            }

            int nb = k / bs;
            for (bi = 0; bi < nb; bi++) {
                int base = bi * bs;
                int j;
                dequant_block(row_a + (size_t)bi * bsz, tmp, dtype);

                for (j = 0; j + 15 < bs; j += 16) {
                    __m512 av = _mm512_loadu_ps(tmp + j);
                    __m512 bv;
                    if (n == 1) {
                        bv = _mm512_loadu_ps(b + base + j);
                    } else {
                        float vals[16];
                        int jj;
                        for (jj = 0; jj < 16; jj++)
                            vals[jj] = b[(base+j+jj)*n+ni];
                        bv = _mm512_loadu_ps(vals);
                    }
                    acc = _mm512_fmadd_ps(av, bv, acc);
                }
                /* tail (Q8_0 bs=32: 32/16=2 iterations, no tail) */
                for (; j < bs; j++)
                    acc = _mm512_add_ps(acc,
                          _mm512_set1_ps(tmp[j] * b[(base+j)*n+ni]));
            }

            out[mi * n + ni] = _mm512_reduce_add_ps(acc);
        }
    }
}

#endif /* x86_64 */

/* =========================================================================
 * NEON matmul  (AArch64 always; ARMv7 when compiled with -mfpu=neon)
 *
 * 4 × float32 lanes.  Same block-at-a-time structure as AVX2.
 * ====================================================================== */

#if defined(__ARM_NEON) || defined(__aarch64__)

void cmol_matmul_neon(float *out, const void *a, const float *b,
                       int m, int k, int n, cmol_dtype_t dtype) {
    size_t bsz;
    int    bs = block_params(dtype, &bsz);
    if (!bs) return;

    const uint8_t *abytes   = (const uint8_t *)a;
    size_t         row_bytes = (size_t)(k / bs) * bsz;
    float          tmp[256];
    int mi, ni, bi;

    for (mi = 0; mi < m; mi++) {
        const uint8_t *row_a = abytes + (size_t)mi * row_bytes;

        for (ni = 0; ni < n; ni++) {
            float32x4_t acc = vdupq_n_f32(0.0f);

            if (dtype == CMOL_DTYPE_F32) {
                const float *af = (const float *)row_a;
                int ki;
                for (ki = 0; ki + 3 < k; ki += 4) {
                    float32x4_t av = vld1q_f32(af + ki);
                    float32x4_t bv;
                    if (n == 1) {
                        bv = vld1q_f32(b + ki);
                    } else {
                        float vals[4] = {
                            b[(ki+0)*n+ni], b[(ki+1)*n+ni],
                            b[(ki+2)*n+ni], b[(ki+3)*n+ni]
                        };
                        bv = vld1q_f32(vals);
                    }
                    acc = vmlaq_f32(acc, av, bv);
                }
                /* horizontal sum */
                float32x2_t lo = vget_low_f32(acc);
                float32x2_t hi = vget_high_f32(acc);
                float32x2_t s  = vadd_f32(lo, hi);
                float tail = vget_lane_f32(vpadd_f32(s, s), 0);
                for (; ki < k; ki++) tail += af[ki] * b[ki * n + ni];
                out[mi * n + ni] = tail;
                continue;

            } else if (dtype == CMOL_DTYPE_F16) {
                const uint16_t *ah = (const uint16_t *)row_a;
                float s = 0.0f;
                int ki;
                for (ki = 0; ki < k; ki++)
                    s += f16_to_f32(ah[ki]) * b[ki * n + ni];
                out[mi * n + ni] = s;
                continue;
            }

            int nb = k / bs;
            for (bi = 0; bi < nb; bi++) {
                int base = bi * bs;
                int j;
                dequant_block(row_a + (size_t)bi * bsz, tmp, dtype);

                for (j = 0; j + 3 < bs; j += 4) {
                    float32x4_t av = vld1q_f32(tmp + j);
                    float32x4_t bv;
                    if (n == 1) {
                        bv = vld1q_f32(b + base + j);
                    } else {
                        float vals[4] = {
                            b[(base+j+0)*n+ni], b[(base+j+1)*n+ni],
                            b[(base+j+2)*n+ni], b[(base+j+3)*n+ni]
                        };
                        bv = vld1q_f32(vals);
                    }
                    acc = vmlaq_f32(acc, av, bv);
                }
                /* tail (Q8_0 bs=32, Q4_K/Q6_K bs=256: all multiples of 4) */
                for (; j < bs; j++)
                    acc = vaddq_f32(acc,
                          vdupq_n_f32(tmp[j] * b[(base+j)*n+ni]));
            }

            {
                float32x2_t lo = vget_low_f32(acc);
                float32x2_t hi = vget_high_f32(acc);
                float32x2_t s  = vadd_f32(lo, hi);
                out[mi * n + ni] = vget_lane_f32(vpadd_f32(s, s), 0);
            }
        }
    }
}

#endif /* ARM_NEON */

/* =========================================================================
 * cmol_kernels_select
 *
 * Detects the CPU at run time and returns the fastest available kernel.
 * Called once from cmol_load().
 * ====================================================================== */

cmol_kernels_t cmol_kernels_select(void) {
    cmol_kernels_t kn;
    cmol_cpu_t     cpu = cmol_detect_cpu();

#if defined(__x86_64__) || defined(_M_X64)
    if (cpu.avx512f) {
        kn.matmul = cmol_matmul_avx512;
        kn.name   = "avx512";
        return kn;
    }
    if (cpu.avx2) {
        kn.matmul = cmol_matmul_avx2;
        kn.name   = "avx2";
        return kn;
    }
#endif

#if defined(__ARM_NEON) || defined(__aarch64__)
    if (cpu.neon) {
        kn.matmul = cmol_matmul_neon;
        kn.name   = "neon";
        return kn;
    }
#endif

    (void)cpu;
    kn.matmul = cmol_matmul_scalar;
    kn.name   = "scalar";
    return kn;
}
