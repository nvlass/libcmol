/*
 * quant.c — quantisation kernels and SIMD dispatch
 * Implemented in Phase 4.
 * Included by src/cmol.c (unity build); do not compile standalone.
 */

#include "quant.h"

/* Phase 4 — TODO: Q8_0 and Q4_K_M dequant + matmul kernels, SIMD variants */

void cmol_dequant_row(const void *src, float *dst, int n, cmol_dtype_t dtype) {
    (void)src; (void)dst; (void)n; (void)dtype;
}

void cmol_matmul_scalar(float *out, const void *a, const float *b,
                         int m, int k, int n, cmol_dtype_t dtype) {
    (void)out; (void)a; (void)b; (void)m; (void)k; (void)n; (void)dtype;
}

cmol_kernels_t cmol_kernels_select(void) {
    cmol_kernels_t kn;
    kn.matmul = cmol_matmul_scalar;
    kn.name   = "scalar";
    /* Phase 4: detect CPU features and select AVX-512 / AVX2 / NEON */
    return kn;
}
