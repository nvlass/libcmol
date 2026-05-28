/*
 * test_quant.c — Phase 4 tests for quantisation kernels
 *
 * Tests are built against build/cmol_d.o (the full unity build); they only
 * need to include headers for declarations.  The implementations live in
 * src/quant.c (included by src/cmol.c).
 *
 * Coverage:
 *   1. cmol_dequant_row() — Q8_0, Q4_K, Q6_K, F32 passthrough
 *   2. cmol_matmul_scalar() — F32 weights (exact), Q8_0 weights (exact)
 *   3. cmol_kernels_select() — returns valid function pointer + name;
 *                              selected kernel matches scalar result
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>   /* fabsf */

#include "quant.h"  /* cmol_dequant_row, cmol_matmul_scalar, cmol_kernels_select */

/* =========================================================================
 * Minimal test harness
 * ====================================================================== */

static int g_tests = 0, g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do {                                           \
    g_tests++;                                                          \
    if (cond) { g_pass++; printf("  PASS  %s\n", msg); }              \
    else       { g_fail++; printf("  FAIL  %s\n", msg); }             \
} while (0)

#define NEAR(a, b)  (fabsf((a) - (b)) < 1e-4f)

#define SECTION(name) printf("\n[%s]\n", name)

/* =========================================================================
 * Float16 helpers  (must match quant.c's f16_to_f32)
 *
 * Inline encoder — only used to build test blocks; not tested separately.
 * ====================================================================== */

/* Encode a small positive float as float16 (no subnormal / Inf / NaN) */
static uint16_t f32_to_f16_test(float v) {
    uint32_t u;
    memcpy(&u, &v, 4);
    uint16_t sign = (uint16_t)((u >> 16) & 0x8000u);
    int      exp  = (int)((u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (u & 0x7FFFFFu) >> 13;
    if (exp <= 0)  return sign;              /* flush to ±0 */
    if (exp >= 31) return sign | 0x7C00u;    /* clamp to Inf */
    return sign | (uint16_t)((unsigned)exp << 10) | (uint16_t)mant;
}

/* =========================================================================
 * Block layout mirrors (must match quant.c; packed structs)
 * ====================================================================== */

#if defined(__GNUC__) || defined(__clang__)
#  define PACKED __attribute__((packed))
#else
#  define PACKED
#endif

typedef struct PACKED { uint16_t d; int8_t qs[32]; }                 tq8_0;
typedef struct PACKED { uint16_t d; uint16_t dmin;
                         uint8_t scales[12]; uint8_t qs[128]; }       tq4_k;
typedef struct PACKED { uint8_t ql[128]; uint8_t qh[64];
                         int8_t scales[16]; uint16_t d; }             tq6_k;

/* =========================================================================
 * 1. cmol_dequant_row — F32 passthrough
 * ====================================================================== */

static void test_dequant_f32(void) {
    SECTION("dequant_row F32 passthrough");

    float src[4] = { 1.0f, -2.5f, 0.0f, 100.0f };
    float dst[4] = { 0 };
    cmol_dequant_row(src, dst, 4, CMOL_DTYPE_F32);

    CHECK(NEAR(dst[0],   1.0f), "dst[0] = 1.0");
    CHECK(NEAR(dst[1],  -2.5f), "dst[1] = -2.5");
    CHECK(NEAR(dst[2],   0.0f), "dst[2] = 0.0");
    CHECK(NEAR(dst[3], 100.0f), "dst[3] = 100.0");
}

/* =========================================================================
 * 2. cmol_dequant_row — Q8_0
 *
 * Block: d = 2.0 (f16), qs = {3, -1, 0, 0, ...}
 * Expected: {6.0, -2.0, 0.0, ...}
 * ====================================================================== */

static void test_dequant_q8_0(void) {
    SECTION("dequant_row Q8_0");

    tq8_0 blk;
    memset(&blk, 0, sizeof blk);
    blk.d    = f32_to_f16_test(2.0f);
    blk.qs[0] =  3;
    blk.qs[1] = -1;
    /* rest stay 0 */

    float dst[32] = { 0 };
    cmol_dequant_row(&blk, dst, 32, CMOL_DTYPE_Q8_0);

    CHECK(NEAR(dst[0],  6.0f), "Q8_0 dst[0] = 6.0");
    CHECK(NEAR(dst[1], -2.0f), "Q8_0 dst[1] = -2.0");
    CHECK(NEAR(dst[2],  0.0f), "Q8_0 dst[2] = 0.0");
    CHECK(NEAR(dst[31], 0.0f), "Q8_0 dst[31]= 0.0");

    /* Two consecutive blocks */
    tq8_0 blk2[2];
    memset(blk2, 0, sizeof blk2);
    blk2[0].d    = f32_to_f16_test(1.0f);
    blk2[0].qs[0] = 7;
    blk2[1].d    = f32_to_f16_test(3.0f);
    blk2[1].qs[0] = 2;

    float dst2[64] = { 0 };
    cmol_dequant_row(blk2, dst2, 64, CMOL_DTYPE_Q8_0);
    CHECK(NEAR(dst2[0],  7.0f), "Q8_0 two-block dst[0] = 7.0");
    CHECK(NEAR(dst2[32], 6.0f), "Q8_0 two-block dst[32]= 6.0");
}

/* =========================================================================
 * 3. cmol_dequant_row — Q4_K
 *
 * Correct llama.cpp layout: sub-blocks 0 and 1 both consume qs[0..31],
 * sub 0 → lower nibbles, sub 1 → upper nibbles.
 *
 * Super-block: d=1.0, dmin=0.0
 * scales = {1,1,0,...,0} → sub-block 0: sc=1; sub-block 1: sc=1; rest: sc=0
 * qs[0] = 0xA3 → lower nibble = 3, upper nibble = 10 (0xA)
 *
 * Expected:
 *   dst[0]  = sc[0]*lower(qs[0]) = 1*3 = 3.0  (sub-block 0, pos 0)
 *   dst[1]  = sc[0]*lower(qs[1]) = 1*0 = 0.0  (sub-block 0, pos 1; qs[1]=0)
 *   dst[2]  = 0.0 (qs[2]=0)
 *   dst[32] = sc[1]*upper(qs[0]) = 1*10 = 10.0 (sub-block 1, pos 0)
 *   dst[255]= 0.0 (sub-blocks 2-7: sc=0)
 * ====================================================================== */

static void test_dequant_q4_k(void) {
    SECTION("dequant_row Q4_K");

    tq4_k blk;
    memset(&blk, 0, sizeof blk);
    blk.d    = f32_to_f16_test(1.0f);
    blk.dmin = f32_to_f16_test(0.0f);

    /* sub-block 0 (lower nibbles of qs[0..31]): sc=1, mn=0 */
    blk.scales[0] = 1;
    /* sub-block 1 (upper nibbles of qs[0..31]): sc=1, mn=0 */
    blk.scales[1] = 1;

    /* qs[0]: lower nibble=3, upper nibble=10 */
    blk.qs[0] = (uint8_t)((10u << 4) | 3u);

    float dst[256] = { 0 };
    cmol_dequant_row(&blk, dst, 256, CMOL_DTYPE_Q4_K);

    CHECK(NEAR(dst[0],  3.0f),  "Q4_K dst[0] = 3.0");
    CHECK(NEAR(dst[1],  0.0f),  "Q4_K dst[1] = 0.0 (qs[1]=0, lower nibble)");
    CHECK(NEAR(dst[2],  0.0f),  "Q4_K dst[2] = 0.0 (qs[1]=0)");
    CHECK(NEAR(dst[32],10.0f),  "Q4_K dst[32]= 10.0 (upper nibble of qs[0], sub 1)");
    CHECK(NEAR(dst[255],0.0f),  "Q4_K dst[255]=0.0");
}

/* =========================================================================
 * 4. cmol_dequant_row — Q6_K
 *
 * Super-block: d=1.0, scales[0]=2 (first 16 values use scale 2)
 * i=0: ql[0] low nibble = 10 (0xA), qh[0] bits 1:0 = 2
 *      lo=10, hi=2, combined = 10 | (2<<4) = 42, q = 42-32 = 10
 *      dst[0] = 1.0 * 2 * 10 = 20.0
 * i=1: ql[0] high nibble = 0, qh[0] bits 3:2 = 0
 *      lo=0, hi=0, combined=0, q=0-32=-32
 *      dst[1] = 1.0 * 2 * (-32) = -64.0
 * ====================================================================== */

static void test_dequant_q6_k(void) {
    SECTION("dequant_row Q6_K");

    tq6_k blk;
    memset(&blk, 0, sizeof blk);
    blk.d        = f32_to_f16_test(1.0f);
    blk.scales[0] = 2;   /* scale for values 0..15 */

    /* i=0: low nibble of ql[0] = 0xA (10), hi 2 bits of qh[0] bits 1:0 = 2 */
    blk.ql[0] = 0x0Au;  /* high nibble 0 for i=1, low nibble 10 for i=0 */
    blk.qh[0] = 0x02u;  /* bits 1:0=2 for i=0, bits 3:2=0 for i=1       */

    float dst[256] = { 0 };
    cmol_dequant_row(&blk, dst, 256, CMOL_DTYPE_Q6_K);

    /* i=0: d=1.0, scales[0]=2, q=10 → 1.0*2*10=20 */
    CHECK(NEAR(dst[0], 20.0f),  "Q6_K dst[0] = 20.0");
    /* i=1: d=1.0, scales[0]=2, q=-32 → 1.0*2*(-32)=-64 */
    CHECK(NEAR(dst[1], -64.0f), "Q6_K dst[1] = -64.0");
    /* i=16: scales[1]=0 → dst[16] = 0 */
    CHECK(NEAR(dst[16], 0.0f),  "Q6_K dst[16]= 0.0 (scale 0)");
}

/* =========================================================================
 * 5. cmol_matmul_scalar — F32 weights
 *
 * m=3, k=4, n=1
 * A (3×4 f32): rows = {1,0,0,0}, {0,1,0,0}, {1,1,0,0}
 * B (4×1 f32): {5.0, 7.0, 0.0, 0.0}
 * Expected out: {5.0, 7.0, 12.0}
 * ====================================================================== */

static void test_matmul_f32(void) {
    SECTION("matmul_scalar F32");

    float A[12] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        1.0f, 1.0f, 0.0f, 0.0f,
    };
    float B[4] = { 5.0f, 7.0f, 0.0f, 0.0f };
    float out[3] = { 0 };

    cmol_matmul_scalar(out, A, B, 3, 4, 1, CMOL_DTYPE_F32);

    CHECK(NEAR(out[0],  5.0f), "F32 matmul out[0] = 5.0");
    CHECK(NEAR(out[1],  7.0f), "F32 matmul out[1] = 7.0");
    CHECK(NEAR(out[2], 12.0f), "F32 matmul out[2] = 12.0");
}

/* =========================================================================
 * 6. cmol_matmul_scalar — F32 weights, n=2 (batch)
 *
 * m=2, k=4, n=2
 * A: rows = {1,0,0,0}, {0,1,0,0}
 * B (4×2): {{1,2},{3,4},{0,0},{0,0}} stored row-major: {1,2,3,4,0,0,0,0}
 * out[0][0] = 1, out[0][1] = 2, out[1][0] = 3, out[1][1] = 4
 * ====================================================================== */

static void test_matmul_f32_batch(void) {
    SECTION("matmul_scalar F32 n=2");

    float A[8] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
    };
    /* B[k][n] row-major: k=0→{1,2}, k=1→{3,4}, k=2→{0,0}, k=3→{0,0} */
    float B[8] = { 1.0f, 2.0f, 3.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    float out[4] = { 0 };

    cmol_matmul_scalar(out, A, B, 2, 4, 2, CMOL_DTYPE_F32);

    CHECK(NEAR(out[0], 1.0f), "F32 batch out[0][0] = 1.0");
    CHECK(NEAR(out[1], 2.0f), "F32 batch out[0][1] = 2.0");
    CHECK(NEAR(out[2], 3.0f), "F32 batch out[1][0] = 3.0");
    CHECK(NEAR(out[3], 4.0f), "F32 batch out[1][1] = 4.0");
}

/* =========================================================================
 * 7. cmol_matmul_scalar — Q8_0 weights
 *
 * m=2, k=32, n=1
 * Row 0: d=1.0, qs={1,0,...} → dot with B[0]=1 only → 1.0
 * Row 1: d=3.0, qs={0,1,...} → dot with B[1]=2    → 3.0*1*2 = 6.0
 * B = {1.0, 2.0, 0.0, ..., 0.0}
 * ====================================================================== */

static void test_matmul_q8_0(void) {
    SECTION("matmul_scalar Q8_0");

    tq8_0 A[2];
    memset(A, 0, sizeof A);
    A[0].d    = f32_to_f16_test(1.0f);
    A[0].qs[0] = 1;
    A[1].d    = f32_to_f16_test(3.0f);
    A[1].qs[1] = 1;

    float B[32] = { 0 };
    B[0] = 1.0f;
    B[1] = 2.0f;

    float out[2] = { 0 };
    cmol_matmul_scalar(out, A, B, 2, 32, 1, CMOL_DTYPE_Q8_0);

    CHECK(NEAR(out[0], 1.0f), "Q8_0 matmul out[0] = 1.0");
    CHECK(NEAR(out[1], 6.0f), "Q8_0 matmul out[1] = 6.0 (3*1*2)");
}

/* =========================================================================
 * 8. cmol_matmul_scalar — Q8_0, larger k=64 (2 blocks)
 *
 * m=1, k=64 (2 Q8_0 blocks), n=1
 * Block 0: d=1.0, qs[0]=5  → contributes 5.0 (B[0]=1.0)
 * Block 1: d=2.0, qs[0]=3  → contributes 6.0 (B[32]=1.0)
 * Expected: 11.0
 * ====================================================================== */

static void test_matmul_q8_0_two_blocks(void) {
    SECTION("matmul_scalar Q8_0 k=64 (2 blocks)");

    tq8_0 A[2];
    memset(A, 0, sizeof A);
    A[0].d    = f32_to_f16_test(1.0f);
    A[0].qs[0] = 5;
    A[1].d    = f32_to_f16_test(2.0f);
    A[1].qs[0] = 3;

    float B[64] = { 0 };
    B[0]  = 1.0f;
    B[32] = 1.0f;

    float out[1] = { 0 };
    cmol_matmul_scalar(out, A, B, 1, 64, 1, CMOL_DTYPE_Q8_0);

    CHECK(NEAR(out[0], 11.0f), "Q8_0 two-block dot = 11.0");
}

/* =========================================================================
 * 9. cmol_kernels_select — validity and correctness
 *
 * Build a small Q8_0 matmul with the scalar kernel and the selected kernel;
 * results must match.
 * ====================================================================== */

static void test_kernels_select(void) {
    SECTION("cmol_kernels_select");

    cmol_kernels_t kn = cmol_kernels_select();
    CHECK(kn.matmul != NULL, "selected matmul fn non-NULL");
    CHECK(kn.name   != NULL, "selected kernel name non-NULL");
    if (kn.name) printf("         selected kernel: %s\n", kn.name);

    /* Run both the scalar kernel and the selected kernel on identical data
     * and compare results.  Use m=4, k=32, n=1. */
    tq8_0 A[4];
    memset(A, 0, sizeof A);
    A[0].d = f32_to_f16_test(1.0f); A[0].qs[0] = 1;  A[0].qs[1] = 2;
    A[1].d = f32_to_f16_test(2.0f); A[1].qs[0] = -1; A[1].qs[2] = 3;
    A[2].d = f32_to_f16_test(0.5f); A[2].qs[4] = 4;
    A[3].d = f32_to_f16_test(1.0f); A[3].qs[31]= 7;

    float B[32] = { 0 };
    int i;
    for (i = 0; i < 32; i++) B[i] = (float)(i + 1);

    float ref[4] = { 0 };
    float got[4] = { 0 };

    cmol_matmul_scalar(ref, A, B, 4, 32, 1, CMOL_DTYPE_Q8_0);
    kn.matmul(got, A, B, 4, 32, 1, CMOL_DTYPE_Q8_0);

    CHECK(NEAR(got[0], ref[0]), "kernel matches scalar for row 0");
    CHECK(NEAR(got[1], ref[1]), "kernel matches scalar for row 1");
    CHECK(NEAR(got[2], ref[2]), "kernel matches scalar for row 2");
    CHECK(NEAR(got[3], ref[3]), "kernel matches scalar for row 3");
}

/* =========================================================================
 * 10. Edge cases: NULL / zero inputs
 * ====================================================================== */

static void test_edge_cases(void) {
    SECTION("Edge cases (NULL / no-op)");

    float dst[32] = { 9.9f };
    /* NULL src → no crash, dst unchanged */
    cmol_dequant_row(NULL, dst, 32, CMOL_DTYPE_Q8_0);
    CHECK(dst[0] == 9.9f, "NULL src: dst unchanged");

    /* NULL dst → no crash */
    tq8_0 blk; memset(&blk, 0, sizeof blk);
    cmol_dequant_row(&blk, NULL, 32, CMOL_DTYPE_Q8_0);
    CHECK(1, "NULL dst: no crash");

    /* n=0 → no crash */
    cmol_dequant_row(&blk, dst, 0, CMOL_DTYPE_Q8_0);
    CHECK(dst[0] == 9.9f, "n=0: dst unchanged");

    /* unknown dtype → no crash */
    cmol_dequant_row(&blk, dst, 32, (cmol_dtype_t)99);
    CHECK(dst[0] == 9.9f, "unknown dtype: dst unchanged");
}

/* =========================================================================
 * main
 * ====================================================================== */

int main(void) {
    printf("=== test_quant ===\n");

    test_dequant_f32();
    test_dequant_q8_0();
    test_dequant_q4_k();
    test_dequant_q6_k();
    test_matmul_f32();
    test_matmul_f32_batch();
    test_matmul_q8_0();
    test_matmul_q8_0_two_blocks();
    test_kernels_select();
    test_edge_cases();

    printf("\n=== %d/%d passed", g_pass, g_tests);
    if (g_fail) printf(", %d FAILED", g_fail);
    printf(" ===\n");

    return g_fail ? 1 : 0;
}
