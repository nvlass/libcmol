/*
 * quant.h — quantisation kernels and SIMD dispatch
 * Implemented in Phase 4.
 * Internal header — not part of the public API.
 */

#ifndef CMOL_QUANT_H
#define CMOL_QUANT_H

#include "cmol_internal.h"

/*
 * cmol_kernels_select — detect CPU capabilities and return the fastest
 * available kernel set.  Called once in cmol_load().
 */
cmol_kernels_t cmol_kernels_select(void);

/*
 * cmol_dequant_row — dequantise `n` values from a quantised row into
 * float32.  `n` must be a multiple of the block size for `dtype`.
 *
 * Q8_0  block size: 32  values
 * Q4_K  block size: 256 values
 */
void cmol_dequant_row(const void *src, float *dst, int n, cmol_dtype_t dtype);

/* ---- Kernel implementations (one per SIMD level) ---------------------- */

void cmol_matmul_scalar(float *out, const void *a, const float *b,
                         int m, int k, int n, cmol_dtype_t dtype);

#if defined(__x86_64__) || defined(_M_X64)
void cmol_matmul_avx2  (float *out, const void *a, const float *b,
                         int m, int k, int n, cmol_dtype_t dtype);
void cmol_matmul_avx512(float *out, const void *a, const float *b,
                         int m, int k, int n, cmol_dtype_t dtype);
#endif

#if defined(__ARM_NEON) || defined(__aarch64__)
void cmol_matmul_neon  (float *out, const void *a, const float *b,
                         int m, int k, int n, cmol_dtype_t dtype);
#endif

#endif /* CMOL_QUANT_H */
