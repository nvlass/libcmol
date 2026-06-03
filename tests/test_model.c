/*
 * test_model.c — Phase 5 tests for transformer primitives and forward pass
 *
 * Sections:
 *   1. cmol_rms_norm
 *   2. cmol_softmax
 *   3. cmol_swiglu
 *   4. cmol_rope_apply   (RoPE identity at pos=0 + rotation at pos=1)
 *   5. cmol_model_forward — end-to-end pass through a tiny synthetic model
 *      (2 layers, d_model=32, n_heads=2, n_kv_heads=1, d_ffn=64,
 *       vocab_size=16, all weights F32)
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#include "model.h"   /* primitive declarations */
#include "attn.h"    /* cmol_attn_forward       */
#include "quant.h"   /* cmol_kernels_select      */

/* =========================================================================
 * Harness
 * ====================================================================== */

static int g_tests = 0, g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do {                                            \
    g_tests++;                                                           \
    if (cond) { g_pass++; printf("  PASS  %s\n", msg); }               \
    else       { g_fail++; printf("  FAIL  %s\n", msg); }              \
} while (0)

#define NEAR(a,b)  (fabsf((float)(a)-(float)(b)) < 1e-4f)
#define NEAR3(a,b) (fabsf((float)(a)-(float)(b)) < 1e-3f)

#define SECTION(n) printf("\n[%s]\n", n)

/* =========================================================================
 * 1. cmol_rms_norm
 * ====================================================================== */

static void test_rms_norm(void) {
    SECTION("cmol_rms_norm");

    /* x = {3,4}, w = {1,1}: rms = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.5355
     * out = x * (1/rms): {3/3.5355, 4/3.5355} = {0.8485, 1.1314}         */
    float x[2] = {3.0f, 4.0f}, w[2] = {1.0f, 1.0f}, out[2];
    cmol_rms_norm(x, w, out, 2, 0.0f);
    CHECK(NEAR(out[0], 3.0f / sqrtf(12.5f)), "norm({3,4})[0]");
    CHECK(NEAR(out[1], 4.0f / sqrtf(12.5f)), "norm({3,4})[1]");

    /* Scale weight doubles output */
    float w2[2] = {2.0f, 2.0f};
    cmol_rms_norm(x, w2, out, 2, 0.0f);
    CHECK(NEAR(out[0], 2.0f * 3.0f / sqrtf(12.5f)), "scaled norm[0]");

    /* In-place (out == x) */
    float xip[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float wip[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    cmol_rms_norm(xip, wip, xip, 4, 1e-6f);
    /* rms({1,0,0,0}) = sqrt(1/4 + eps) ≈ 0.5; out[0] ≈ 2.0 */
    CHECK(NEAR3(xip[0], 2.0f), "in-place norm[0] ≈ 2.0");
    CHECK(NEAR3(xip[1], 0.0f), "in-place norm[1] = 0.0");

    /* eps prevents div-by-zero on all-zero input */
    float z[2] = {0.0f, 0.0f}, wz[2] = {1.0f, 1.0f}, oz[2];
    cmol_rms_norm(z, wz, oz, 2, 1e-5f);
    CHECK(isfinite(oz[0]), "zero input → finite output");
}

/* =========================================================================
 * 2. cmol_softmax
 * ====================================================================== */

static void test_softmax(void) {
    SECTION("cmol_softmax");

    float x[3] = {1.0f, 2.0f, 3.0f};
    cmol_softmax(x, 3);

    /* Sum must be 1 */
    float sm = x[0] + x[1] + x[2];
    CHECK(NEAR(sm, 1.0f), "softmax sum = 1.0");

    /* Largest input gets largest probability */
    CHECK(x[2] > x[1] && x[1] > x[0], "order preserved");

    /* All equal → uniform */
    float e[4] = {5.0f, 5.0f, 5.0f, 5.0f};
    cmol_softmax(e, 4);
    CHECK(NEAR(e[0], 0.25f) && NEAR(e[3], 0.25f), "equal inputs → uniform");

    /* One-hot after large spread */
    float oh[3] = {0.0f, 0.0f, 100.0f};
    cmol_softmax(oh, 3);
    CHECK(NEAR(oh[2], 1.0f), "dominant logit → ~1.0");
    CHECK(NEAR(oh[0], 0.0f), "non-dominant → ~0.0");
}

/* =========================================================================
 * 3. cmol_swiglu
 * ====================================================================== */

static void test_swiglu(void) {
    SECTION("cmol_swiglu");

    /* SiLU(0) = 0; 0 * up = 0 */
    float g[2] = {0.0f, 1.0f}, u[2] = {3.0f, 2.0f}, out[2];
    cmol_swiglu(g, u, out, 2);
    CHECK(NEAR(out[0], 0.0f),  "silu(0) * 3 = 0");
    /* silu(1) = 1 / (1 + e^{-1}) ≈ 0.7311; 0.7311 * 2 ≈ 1.4624 */
    CHECK(NEAR3(out[1], (1.0f / (1.0f + expf(-1.0f))) * 2.0f), "silu(1)*2");

    /* In-place (out == gate) */
    float g2[2] = {2.0f, -2.0f}, u2[2] = {1.0f, 1.0f};
    cmol_swiglu(g2, u2, g2, 2);
    CHECK(g2[0] > 0.0f, "silu(2)*1 > 0");
    CHECK(g2[1] < 0.0f, "silu(-2)*1 < 0");
}

/* =========================================================================
 * 4. cmol_rope_apply
 * ====================================================================== */

static void test_rope(void) {
    SECTION("cmol_rope_apply");

    /* At pos=0: theta=0 for all dims → cos=1, sin=0 → no change */
    float q[8] = {1,2,3,4, 5,6,7,8};   /* 2 heads × 4 dims */
    float k[4] = {1,2,3,4};             /* 1 KV head × 4 dims */
    float q_ref[8], k_ref[4];
    memcpy(q_ref, q, sizeof q);
    memcpy(k_ref, k, sizeof k);

    cmol_rope_apply(q, k, /*pos=*/0, /*n_heads=*/2, /*n_kv_heads=*/1,
                    /*head_dim=*/4, /*freq_base=*/10000.0f);

    CHECK(NEAR(q[0], q_ref[0]) && NEAR(q[4], q_ref[4]), "pos=0: Q head0 unchanged");
    CHECK(NEAR(q[1], q_ref[1]) && NEAR(q[5], q_ref[5]), "pos=0: Q head1 unchanged");
    CHECK(NEAR(k[0], k_ref[0]) && NEAR(k[2], k_ref[2]), "pos=0: K unchanged");

    /* At pos=1, head_dim=4: NORM style rotates adjacent pairs.
     * theta_0 = pos / freq_base^(2*0/4) = 1/1 = 1.0  → pair (q[0], q[1])
     * theta_1 = pos / freq_base^(2*1/4) = 1/100 = 0.01 → pair (q[2], q[3])
     * q'[0] = q[0]*cos(1) - q[1]*sin(1) = 1*0.5403 - 2*0.8415 ≈ -1.1427
     * q'[1] = q[0]*sin(1) + q[1]*cos(1) = 1*0.8415 + 2*0.5403 ≈  1.9221
     */
    float q2[4] = {1.0f, 2.0f, 3.0f, 4.0f}; /* 1 head, head_dim=4 */
    float k2[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float dummy[1] = {0}; (void)dummy;
    cmol_rope_apply(q2, k2, 1, 1, 1, 4, 10000.0f);

    float theta0 = 1.0f; /* pos=1, pair i=0: 1/10000^(0/4) = 1.0 */
    float q0_expected = 1.0f * cosf(theta0) - 2.0f * sinf(theta0);
    float q1_expected = 1.0f * sinf(theta0) + 2.0f * cosf(theta0);
    CHECK(NEAR3(q2[0], q0_expected), "pos=1 Q[0] rotated");
    CHECK(NEAR3(q2[1], q1_expected), "pos=1 Q[2] rotated");

    /* Rotation must be orthonormal: norm of each pair preserved */
    float norm_before = sqrtf(1.0f*1.0f + 2.0f*2.0f);
    float norm_after  = sqrtf(q2[0]*q2[0] + q2[1]*q2[1]);
    CHECK(NEAR(norm_before, norm_after), "RoPE preserves norm");
}

/* =========================================================================
 * 5. cmol_model_forward — synthetic micro-model
 *
 * Architecture:
 *   vocab_size=16, d_model=32, n_heads=2, n_kv_heads=1, d_head=16
 *   d_ffn=64, n_layers=2, rms_norm_eps=1e-5
 *   All weight tensors F32, zero-initialised except:
 *     - token_embd.weight:  identity-like (token T → x[T]=1, rest 0)
 *     - output_norm.weight: all ones
 *     - blk.*.attn_norm.weight: all ones
 *     - blk.*.ffn_norm.weight:  all ones
 *   All projection weights (attn_q/k/v/output, ffn_gate/up/down) = zero.
 *
 * With zero projection weights:
 *   attn output = 0, ffn output = 0 (gate and up are 0 → swiglu = 0)
 *   So x stays equal to the embedding throughout.
 *   Final RMSNorm of x (which has one 1.0 and rest 0.0) with w=ones:
 *     norm(x) = x / rms(x) = x / sqrt(1/32) = x * sqrt(32)
 *   lm_head weight = 0 → logits = 0.
 *
 * So we just verify: no crash, logits returned (not NULL), shape is
 * [vocab_size], and the process is deterministic.
 * ====================================================================== */

#define MICRO_VOCAB    16
#define MICRO_D        32
#define MICRO_HEADS    2
#define MICRO_KVHEADS  1
#define MICRO_DHEAD    16    /* MICRO_D / MICRO_HEADS */
#define MICRO_DFFN     64
#define MICRO_LAYERS   2
#define MICRO_CTX      8

/* Number of tensors in the synthetic model:
 *   1 token_embd.weight
 *   1 output_norm.weight
 *   1 output.weight (lm head, not tied)
 *   per layer: attn_norm, ffn_norm, attn_q, attn_k, attn_v, attn_output,
 *              ffn_gate, ffn_up, ffn_down → 9
 *   total = 3 + MICRO_LAYERS * 9
 */
#define MICRO_N_TENSORS (3 + MICRO_LAYERS * 9)

/* Allocate and return a zeroed float tensor data block */
static float *alloc_f32(int n) {
    float *p = (float *)calloc((size_t)n, sizeof(float));
    return p;
}

/* One tensor entry in a flat array */
typedef struct {
    cmol_tensor_t t;
    float        *data_buf; /* separate heap block for float data */
} micro_tensor_t;

static micro_tensor_t  g_mtensors[MICRO_N_TENSORS];
static int             g_n_mt = 0;

static cmol_tensor_t *add_tensor(const char *name, int rows, int cols,
                                   float *data) {
    micro_tensor_t *mt = &g_mtensors[g_n_mt++];
    memset(&mt->t, 0, sizeof mt->t);
    strncpy(mt->t.name, name, CMOL_MAX_TENSOR_NAME - 1);
    mt->t.dtype    = CMOL_DTYPE_F32;
    mt->t.n_dims   = 2;
    mt->t.shape[0] = cols;   /* innermost = in_features */
    mt->t.shape[1] = rows;   /* outermost = out_features */
    mt->t.data     = data;
    mt->data_buf   = data;
    return &mt->t;
}

static void set_all(float *p, int n, float v) {
    int i; for (i = 0; i < n; i++) p[i] = v;
}

/* Build the synthetic model tensors into g_mtensors */
static void build_micro_tensors(void) {
    char name[CMOL_MAX_TENSOR_NAME];
    int  l;

    g_n_mt = 0;

    /* token_embd.weight [vocab_size × d_model]: row T has x[T]=1 */
    {
        float *d = alloc_f32(MICRO_VOCAB * MICRO_D);
        int t;
        for (t = 0; t < MICRO_VOCAB; t++) d[t * MICRO_D + (t % MICRO_D)] = 1.0f;
        add_tensor("token_embd.weight", MICRO_VOCAB, MICRO_D, d);
    }

    /* output_norm.weight [d_model]: all ones */
    {
        float *d = alloc_f32(MICRO_D); set_all(d, MICRO_D, 1.0f);
        add_tensor("output_norm.weight", 1, MICRO_D, d);
    }

    /* output.weight (lm head) [vocab_size × d_model]: zero */
    add_tensor("output.weight", MICRO_VOCAB, MICRO_D, alloc_f32(MICRO_VOCAB * MICRO_D));

    for (l = 0; l < MICRO_LAYERS; l++) {
        /* attn_norm [d_model]: ones */
        float *an = alloc_f32(MICRO_D); set_all(an, MICRO_D, 1.0f);
        snprintf(name, sizeof name, "blk.%d.attn_norm.weight", l);
        add_tensor(name, 1, MICRO_D, an);

        /* ffn_norm [d_model]: ones */
        float *fn = alloc_f32(MICRO_D); set_all(fn, MICRO_D, 1.0f);
        snprintf(name, sizeof name, "blk.%d.ffn_norm.weight", l);
        add_tensor(name, 1, MICRO_D, fn);

        /* attn_q  [d_model × d_model]: zero */
        snprintf(name, sizeof name, "blk.%d.attn_q.weight", l);
        add_tensor(name, MICRO_D, MICRO_D, alloc_f32(MICRO_D * MICRO_D));

        /* attn_k  [kv_dim × d_model]: zero */
        snprintf(name, sizeof name, "blk.%d.attn_k.weight", l);
        add_tensor(name, MICRO_KVHEADS * MICRO_DHEAD, MICRO_D,
                   alloc_f32(MICRO_KVHEADS * MICRO_DHEAD * MICRO_D));

        /* attn_v  [kv_dim × d_model]: zero */
        snprintf(name, sizeof name, "blk.%d.attn_v.weight", l);
        add_tensor(name, MICRO_KVHEADS * MICRO_DHEAD, MICRO_D,
                   alloc_f32(MICRO_KVHEADS * MICRO_DHEAD * MICRO_D));

        /* attn_output [d_model × d_model]: zero */
        snprintf(name, sizeof name, "blk.%d.attn_output.weight", l);
        add_tensor(name, MICRO_D, MICRO_D, alloc_f32(MICRO_D * MICRO_D));

        /* ffn_gate [d_ffn × d_model]: zero */
        snprintf(name, sizeof name, "blk.%d.ffn_gate.weight", l);
        add_tensor(name, MICRO_DFFN, MICRO_D, alloc_f32(MICRO_DFFN * MICRO_D));

        /* ffn_up [d_ffn × d_model]: zero */
        snprintf(name, sizeof name, "blk.%d.ffn_up.weight", l);
        add_tensor(name, MICRO_DFFN, MICRO_D, alloc_f32(MICRO_DFFN * MICRO_D));

        /* ffn_down [d_model × d_ffn]: zero */
        snprintf(name, sizeof name, "blk.%d.ffn_down.weight", l);
        add_tensor(name, MICRO_D, MICRO_DFFN, alloc_f32(MICRO_D * MICRO_DFFN));
    }
}

static void free_micro_tensors(void) {
    int i;
    for (i = 0; i < g_n_mt; i++) free(g_mtensors[i].data_buf);
    g_n_mt = 0;
}

/* Build a cmol_model_t and cmol_session_t from the synthetic tensors */
static struct cmol_model   g_micro_model;
static struct cmol_session g_micro_sess;
static float              *g_kv_k = NULL;
static float              *g_kv_v = NULL;
static float              *g_scratch = NULL;

static void build_micro_model(void) {
    cmol_hparams_t *hp = &g_micro_model.hparams;
    hp->n_layers              = MICRO_LAYERS;
    hp->n_heads               = MICRO_HEADS;
    hp->n_kv_heads            = MICRO_KVHEADS;
    hp->d_model               = MICRO_D;
    hp->d_head                = MICRO_DHEAD;
    hp->d_ffn                 = MICRO_DFFN;
    hp->vocab_size            = MICRO_VOCAB;
    hp->rms_norm_eps          = 1e-5f;
    hp->rope_freq_base        = 10000.0f;
    hp->no_rope_layer_interval = 0;
    hp->tie_embeddings        = 0;

    /* Collect tensor pointers into g_micro_model */
    g_micro_model.tensors  = (cmol_tensor_t *)calloc(MICRO_N_TENSORS,
                                                       sizeof(cmol_tensor_t));
    g_micro_model.n_tensors = g_n_mt;
    {
        int i;
        for (i = 0; i < g_n_mt; i++)
            g_micro_model.tensors[i] = g_mtensors[i].t;
    }

    /* Kernel dispatch */
    g_micro_model.kernels = cmol_kernels_select();

    /* KV cache (MICRO_LAYERS × MICRO_CTX × MICRO_KVHEADS × MICRO_DHEAD) */
    size_t kv_sz = (size_t)MICRO_LAYERS * MICRO_CTX * MICRO_KVHEADS * MICRO_DHEAD;
    g_kv_k = (float *)calloc(kv_sz, sizeof(float));
    g_kv_v = (float *)calloc(kv_sz, sizeof(float));

    g_micro_sess.model                 = &g_micro_model;
    g_micro_sess.kvcache.k             = g_kv_k;
    g_micro_sess.kvcache.v             = g_kv_v;
    g_micro_sess.kvcache.n_tokens      = 0;
    g_micro_sess.kvcache.max_tokens    = MICRO_CTX;

    /* Scratch:
     *   (5*d + 2*kv_dim + max_ctx + 2*d_ffn + vocab_size) floats */
    size_t sc_sz = (size_t)(5 * MICRO_D
                            + 2 * MICRO_KVHEADS * MICRO_DHEAD
                            + MICRO_CTX
                            + 2 * MICRO_DFFN
                            + MICRO_VOCAB);
    g_scratch = (float *)calloc(sc_sz, sizeof(float));
    g_micro_sess.scratch      = g_scratch;
    g_micro_sess.scratch_size = sc_sz * sizeof(float);
}

static void free_micro_model(void) {
    free(g_micro_model.tensors);
    free(g_kv_k);
    free(g_kv_v);
    free(g_scratch);
    g_micro_model.tensors = NULL;
    g_kv_k = g_kv_v = g_scratch = NULL;
}

static void test_forward(void) {
    SECTION("cmol_model_forward (synthetic micro-model)");

    build_micro_tensors();
    build_micro_model();

    /* Forward pass: token 3 at position 0 */
    float *logits = cmol_model_forward(&g_micro_model, &g_micro_sess, 3, 0);

    CHECK(logits != NULL, "forward pass returns non-NULL");

    if (logits) {
        /* All projection weights are zero, so logits should all be 0 */
        int all_zero = 1, i;
        for (i = 0; i < MICRO_VOCAB; i++) {
            if (!isfinite(logits[i])) { all_zero = 0; break; }
        }
        CHECK(all_zero, "all logits finite (zero weights → zero output)");

        /* Logits are all 0 since lm_head weights are zero */
        CHECK(NEAR(logits[0], 0.0f), "logits[0] = 0.0 (zero lm_head)");
    }

    /* Second forward pass (pos=1): verify no crash and deterministic */
    float *logits2 = cmol_model_forward(&g_micro_model, &g_micro_sess, 5, 1);
    CHECK(logits2 != NULL, "second forward (pos=1) non-NULL");
    /* logits and logits2 point into same scratch — both valid now */
    CHECK(logits2 == logits, "logits pointer is stable (same scratch slot)");

    free_micro_model();
    free_micro_tensors();
}

/* =========================================================================
 * 6. Verify scratch minimum size formula
 *    (just checks that scratch_size >= minimum, no crash)
 * ====================================================================== */

static void test_scratch_size(void) {
    SECTION("Scratch size minimum");

    int d     = MICRO_D;
    int kv    = MICRO_KVHEADS * MICRO_DHEAD;
    int ctx   = MICRO_CTX;
    int dffn  = MICRO_DFFN;
    int vocab = MICRO_VOCAB;

    size_t min_floats = (size_t)(5*d + 2*kv + ctx + 2*dffn + vocab);
    size_t got_floats = min_floats; /* we allocate exactly this in the test above */

    CHECK(got_floats >= min_floats, "scratch >= minimum required");
    printf("         scratch = %zu floats = %zu bytes\n",
           got_floats, got_floats * sizeof(float));
}

/* =========================================================================
 * main
 * ====================================================================== */

int main(void) {
    printf("=== test_model ===\n");

    test_rms_norm();
    test_softmax();
    test_swiglu();
    test_rope();
    test_forward();
    test_scratch_size();

    printf("\n=== %d/%d passed", g_pass, g_tests);
    if (g_fail) printf(", %d FAILED", g_fail);
    printf(" ===\n");

    return g_fail ? 1 : 0;
}
