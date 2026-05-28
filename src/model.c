/*
 * model.c — transformer forward pass
 * Included by src/cmol.c (unity build); do not compile standalone.
 *
 * Implements:
 *   cmol_rms_norm    — in-place RMSNorm
 *   cmol_softmax     — numerically stable softmax
 *   cmol_swiglu      — SiLU(gate) * up  (SwiGLU activation)
 *   cmol_rope_apply  — rotary position encoding (LLaMA half-rotation style)
 *   cmol_model_forward — full N-layer transformer pass for one token
 *
 * Tensor naming convention (matches llama.cpp GGUF output):
 *   token_embd.weight          blk.{i}.attn_norm.weight
 *   output_norm.weight         blk.{i}.ffn_norm.weight
 *   output.weight              blk.{i}.attn_q.weight / attn_k / attn_v
 *                              blk.{i}.attn_output.weight
 *                              blk.{i}.attn_q_norm.weight (optional, SmolLM3)
 *                              blk.{i}.attn_k_norm.weight (optional, SmolLM3)
 *                              blk.{i}.ffn_gate.weight / ffn_up / ffn_down
 *
 * Scratch layout (in floats) — set up by cmol_model_forward:
 *   [0]                     x[d_model]               residual stream
 *   [d]                     xnorm[d_model]            post-RMSNorm temp
 *   [2d]                    q[d_model]                Q projection / attn-concat temp
 *   [3d]                    k_buf[kv_dim]             K projection
 *   [3d+kv]                 v_buf[kv_dim]             V projection
 *   [3d+2*kv]               scores[max_ctx]           attention score scratch
 *   [3d+2*kv+ctx]           attn_out[d_model]         attn output (reused for ffn down)
 *   [4d+2*kv+ctx]           ffn_gate[d_ffn]           gate activation
 *   [4d+2*kv+ctx+ffn]       ffn_up[d_ffn]             up   activation
 *   [4d+2*kv+ctx+2*ffn]     logits[vocab_size]        output logits
 *
 * Minimum scratch_size (bytes):
 *   (5*d_model + 2*n_kv_heads*d_head + max_ctx + 2*d_ffn + vocab_size) * sizeof(float)
 */

#include "model.h"
#include "attn.h"     /* cmol_attn_forward */
#include "quant.h"    /* cmol_dequant_row  */

#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>    /* snprintf */

/* =========================================================================
 * Tensor lookup helpers  (also used by attn.c, which is included after us)
 * ====================================================================== */

static cmol_tensor_t *cmol__find_tensor(cmol_tensor_t *tensors, int n,
                                         const char *name) {
    int i;
    for (i = 0; i < n; i++)
        if (strcmp(tensors[i].name, name) == 0)
            return &tensors[i];
    return NULL;
}

/* Build "blk.{layer}.{suffix}" and look it up */
static cmol_tensor_t *cmol__find_blk(cmol_tensor_t *tensors, int n,
                                      int layer, const char *suffix) {
    char name[CMOL_MAX_TENSOR_NAME];
    snprintf(name, sizeof name, "blk.%d.%s", layer, suffix);
    return cmol__find_tensor(tensors, n, name);
}

/* Bytes per row of `k` values packed with `dtype`. */
static size_t cmol__row_bytes(int k, cmol_dtype_t dtype) {
    switch (dtype) {
    case CMOL_DTYPE_F32:  return (size_t)k * 4u;
    case CMOL_DTYPE_F16:  return (size_t)k * 2u;
    case CMOL_DTYPE_Q5_0: return (size_t)(k / 32)  * 22u;
    case CMOL_DTYPE_Q8_0: return (size_t)(k / 32)  * 34u;
    case CMOL_DTYPE_Q4_K: return (size_t)(k / 256) * 144u;
    case CMOL_DTYPE_Q6_K: return (size_t)(k / 256) * 210u;
    default:              return 0u;
    }
}

/* =========================================================================
 * Primitive ops
 * ====================================================================== */

/*
 * cmol_rms_norm — RMSNorm
 *   out[i] = w[i] * x[i] / sqrt(mean(x^2) + eps)
 *
 * x and out may alias (safe: ss is computed before the write loop).
 */
void cmol_rms_norm(const float *x, const float *w, float *out,
                   int n, float eps) {
    float ss = 0.0f;
    int i;
    for (i = 0; i < n; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / (float)n + eps);
    for (i = 0; i < n; i++) out[i] = w[i] * (x[i] * ss);
}

/*
 * cmol_softmax — numerically stable in-place softmax
 */
void cmol_softmax(float *x, int n) {
    float mx = x[0];
    float sm = 0.0f;
    int i;
    for (i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    for (i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sm += x[i]; }
    sm = 1.0f / sm;
    for (i = 0; i < n; i++) x[i] *= sm;
}

/*
 * cmol_swiglu — SiLU(gate) * up element-wise
 *   out[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i]
 * out may alias gate.
 */
void cmol_swiglu(const float *gate, const float *up, float *out, int n) {
    int i;
    for (i = 0; i < n; i++) {
        float g = gate[i];
        out[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

/*
 * cmol_rope_apply — LLaMA half-rotation RoPE
 *
 * For each head h and dimension i in [0, head_dim/2):
 *   θ_i = pos / freq_base^(2i / head_dim)
 *   q[h][i]           ← q[h][i] * cos(θ) − q[h][i+half] * sin(θ)
 *   q[h][i + half]    ← q[h][i] * sin(θ) + q[h][i+half] * cos(θ)
 *   (same for k, up to n_kv_heads)
 */
void cmol_rope_apply(float *q, float *k,
                     int pos, int n_heads, int n_kv_heads,
                     int head_dim, float freq_base) {
    int half = head_dim / 2;
    int h, i;

    for (h = 0; h < n_heads; h++) {
        float *qh = q + h * head_dim;
        for (i = 0; i < half; i++) {
            float theta = (float)pos /
                          powf(freq_base, (float)(2 * i) / (float)head_dim);
            float cs = cosf(theta), sn = sinf(theta);
            float q0 = qh[i], q1 = qh[i + half];
            qh[i]        = q0 * cs - q1 * sn;
            qh[i + half] = q0 * sn + q1 * cs;
        }
    }

    for (h = 0; h < n_kv_heads; h++) {
        float *kh = k + h * head_dim;
        for (i = 0; i < half; i++) {
            float theta = (float)pos /
                          powf(freq_base, (float)(2 * i) / (float)head_dim);
            float cs = cosf(theta), sn = sinf(theta);
            float k0 = kh[i], k1 = kh[i + half];
            kh[i]        = k0 * cs - k1 * sn;
            kh[i + half] = k0 * sn + k1 * cs;
        }
    }
}

/* =========================================================================
 * cmol_model_forward
 *
 * Full transformer forward pass for one token at position `pos`.
 * Returns a pointer into session->scratch holding logits[vocab_size], or
 * NULL if any required tensor is missing.
 * ====================================================================== */

float *cmol_model_forward(const cmol_model_t  *model,
                           struct cmol_session *session,
                           int32_t              token,
                           int                  pos) {
    if (!model || !session) return NULL;

    const cmol_hparams_t *hp       = &model->hparams;
    const cmol_kernels_t *kn       = &model->kernels;
    cmol_tensor_t        *tensors  = model->tensors;
    int                   n_tens   = model->n_tensors;
    cmol_kvcache_t       *kvcache  = &session->kvcache;

    int d     = hp->d_model;
    int kv    = hp->n_kv_heads * hp->d_head;
    int ctx   = kvcache->max_tokens;
    int d_ffn = hp->d_ffn;

    /* Scratch layout (see file-top comment for the full table) */
    float *scratch   = session->scratch;
    float *x         = scratch;                           /* [d]       */
    float *xnorm     = x       + d;                      /* [d]       */
    float *q_buf     = xnorm   + d;                      /* [d]       */
    float *k_buf     = q_buf   + d;                      /* [kv]      */
    float *v_buf     = k_buf   + kv;                     /* [kv]      */
    float *scores    = v_buf   + kv;                     /* [ctx]     */
    float *attn_out  = scores  + ctx;                    /* [d]       */
    float *ffn_gate  = attn_out + d;                     /* [d_ffn]   */
    float *ffn_up    = ffn_gate + d_ffn;                 /* [d_ffn]   */
    float *logits    = ffn_up  + d_ffn;                  /* [vocab_size] */

    /* ── 1. Token embedding lookup ──────────────────────────────────── */
    cmol_tensor_t *embd = cmol__find_tensor(tensors, n_tens, "token_embd.weight");
    if (!embd) return NULL;

    {
        size_t row = cmol__row_bytes(d, embd->dtype);
        cmol_dequant_row((const uint8_t *)embd->data + (size_t)token * row,
                         x, d, embd->dtype);
    }

    /* ── 2. Transformer layers ──────────────────────────────────────── */
    int layer;
    for (layer = 0; layer < hp->n_layers; layer++) {

        /* ---- 2a. Attention sub-layer -------------------------------- */
        cmol_tensor_t *attn_norm =
            cmol__find_blk(tensors, n_tens, layer, "attn_norm.weight");
        if (!attn_norm) return NULL;

        cmol_rms_norm(x, (float *)attn_norm->data, xnorm, d, hp->rms_norm_eps);

        cmol_attn_forward(hp, kn, layer, pos,
                          xnorm, q_buf, k_buf, v_buf, scores, attn_out,
                          kvcache, tensors, n_tens);

        /* Residual */
        { int i; for (i = 0; i < d; i++) x[i] += attn_out[i]; }

        /* ---- 2b. FFN sub-layer (SwiGLU) ----------------------------- */
        cmol_tensor_t *ffn_norm =
            cmol__find_blk(tensors, n_tens, layer, "ffn_norm.weight");
        cmol_tensor_t *ffn_gate_w =
            cmol__find_blk(tensors, n_tens, layer, "ffn_gate.weight");
        cmol_tensor_t *ffn_up_w =
            cmol__find_blk(tensors, n_tens, layer, "ffn_up.weight");
        cmol_tensor_t *ffn_down_w =
            cmol__find_blk(tensors, n_tens, layer, "ffn_down.weight");
        if (!ffn_norm || !ffn_gate_w || !ffn_up_w || !ffn_down_w) return NULL;

        cmol_rms_norm(x, (float *)ffn_norm->data, xnorm, d, hp->rms_norm_eps);

        kn->matmul(ffn_gate, ffn_gate_w->data, xnorm,
                   d_ffn, d, 1, ffn_gate_w->dtype);
        kn->matmul(ffn_up,   ffn_up_w->data,   xnorm,
                   d_ffn, d, 1, ffn_up_w->dtype);

        cmol_swiglu(ffn_gate, ffn_up, ffn_gate, d_ffn); /* result in ffn_gate */

        /* ffn_down: [d_model × d_ffn] · ffn_gate → attn_out (reused buf) */
        kn->matmul(attn_out, ffn_down_w->data, ffn_gate,
                   d, d_ffn, 1, ffn_down_w->dtype);

        /* Residual */
        { int i; for (i = 0; i < d; i++) x[i] += attn_out[i]; }
    }

    /* ── 3. Final RMSNorm + LM head ─────────────────────────────────── */
    cmol_tensor_t *output_norm =
        cmol__find_tensor(tensors, n_tens, "output_norm.weight");
    if (!output_norm) return NULL;
    cmol_rms_norm(x, (float *)output_norm->data, xnorm, d, hp->rms_norm_eps);

    cmol_tensor_t *lm_head = cmol__find_tensor(tensors, n_tens, "output.weight");
    if (!lm_head && hp->tie_embeddings) lm_head = embd; /* tied embeddings */
    if (!lm_head) return NULL;

    kn->matmul(logits, lm_head->data, xnorm,
               hp->vocab_size, d, 1, lm_head->dtype);

    return logits;
}
