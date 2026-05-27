/*
 * attn.c — grouped-query attention with KV cache
 * Included by src/cmol.c (unity build); do not compile standalone.
 *
 * Depends on helpers defined in model.c (included before this file in
 * the unity build):
 *   cmol__find_blk, cmol__row_bytes
 *   cmol_rms_norm, cmol_rope_apply, cmol_softmax
 */

#include "attn.h"
#include "quant.h"  /* cmol_dequant_row (not used directly here but quant.h
                       is needed for block size constants exposed via quant.h) */

#include <string.h>  /* memcpy, memset */
#include <math.h>    /* sqrtf          */

/*
 * cmol_attn_forward
 *
 * See attn.h for the full contract.
 *
 * KV cache layout (k and v are flat arrays):
 *   k[layer][pos][kv_head][dim]
 *   index = ((layer * max_tokens + pos) * n_kv_heads + h_kv) * d_head
 */
void cmol_attn_forward(const cmol_hparams_t *hp,
                        const cmol_kernels_t *kn,
                        int                   layer,
                        int                   pos,
                        const float          *xnorm,
                        float                *q,
                        float                *k_buf,
                        float                *v_buf,
                        float                *scores,
                        float                *out,
                        cmol_kvcache_t       *kvcache,
                        cmol_tensor_t        *tensors,
                        int                   n_tensors) {

    int n_heads    = hp->n_heads;
    int n_kv_heads = hp->n_kv_heads;
    int d_head     = hp->d_head;
    int d_model    = hp->d_model;
    int kv_dim     = n_kv_heads * d_head;

    /* ── 1. Project Q, K, V ─────────────────────────────────────────── */
    cmol_tensor_t *wq = cmol__find_blk(tensors, n_tensors, layer, "attn_q.weight");
    cmol_tensor_t *wk = cmol__find_blk(tensors, n_tensors, layer, "attn_k.weight");
    cmol_tensor_t *wv = cmol__find_blk(tensors, n_tensors, layer, "attn_v.weight");
    cmol_tensor_t *wo = cmol__find_blk(tensors, n_tensors, layer, "attn_output.weight");
    if (!wq || !wk || !wv || !wo) {
        memset(out, 0, (size_t)d_model * sizeof(float));
        return;
    }

    kn->matmul(q,     wq->data, xnorm, d_model, d_model, 1, wq->dtype);
    kn->matmul(k_buf, wk->data, xnorm, kv_dim,  d_model, 1, wk->dtype);
    kn->matmul(v_buf, wv->data, xnorm, kv_dim,  d_model, 1, wv->dtype);

    /* ── 2. Optional QK normalization (SmolLM3 / Falcon3 style) ─────── */
    {
        cmol_tensor_t *q_norm =
            cmol__find_blk(tensors, n_tensors, layer, "attn_q_norm.weight");
        cmol_tensor_t *k_norm =
            cmol__find_blk(tensors, n_tensors, layer, "attn_k_norm.weight");
        int h;
        if (q_norm) {
            float *w = (float *)q_norm->data; /* shape [d_head], shared over heads */
            for (h = 0; h < n_heads; h++)
                cmol_rms_norm(q + h * d_head, w,
                              q + h * d_head, d_head, hp->rms_norm_eps);
        }
        if (k_norm) {
            float *w = (float *)k_norm->data;
            for (h = 0; h < n_kv_heads; h++)
                cmol_rms_norm(k_buf + h * d_head, w,
                              k_buf + h * d_head, d_head, hp->rms_norm_eps);
        }
    }

    /* ── 3. RoPE (skipped for NoPE layers) ──────────────────────────── */
    {
        int nope = hp->no_rope_layer_interval > 0
                   && (layer + 1) % hp->no_rope_layer_interval == 0;
        if (!nope)
            cmol_rope_apply(q, k_buf, pos,
                            n_heads, n_kv_heads, d_head,
                            hp->rope_freq_base);
    }

    /* ── 4. Write K, V into cache at position `pos` ──────────────────── */
    {
        int stride = n_kv_heads * d_head; /* floats per position per layer */
        float *kc = kvcache->k +
                    ((size_t)layer * kvcache->max_tokens + pos) * stride;
        float *vc = kvcache->v +
                    ((size_t)layer * kvcache->max_tokens + pos) * stride;
        memcpy(kc, k_buf, (size_t)kv_dim * sizeof(float));
        memcpy(vc, v_buf, (size_t)kv_dim * sizeof(float));
    }

    /* ── 5. GQA scaled dot-product attention ────────────────────────── */
    float scale  = 1.0f / sqrtf((float)d_head);
    int   seqlen = pos + 1;   /* positions 0 .. pos inclusive */
    int   stride = n_kv_heads * d_head;
    int   h;

    memset(out, 0, (size_t)d_model * sizeof(float));

    for (h = 0; h < n_heads; h++) {
        int    h_kv  = h * n_kv_heads / n_heads;   /* GQA head mapping */
        float *qh    = q + h * d_head;
        float *out_h = out + h * d_head;
        int    t, i;

        /* Compute attention scores for this head over all past tokens */
        for (t = 0; t < seqlen; t++) {
            const float *kh = kvcache->k
                + ((size_t)layer * kvcache->max_tokens + t) * stride
                + h_kv * d_head;
            float dot = 0.0f;
            for (i = 0; i < d_head; i++) dot += qh[i] * kh[i];
            scores[t] = dot * scale;
        }

        cmol_softmax(scores, seqlen);

        /* Weighted sum of cached V vectors */
        for (t = 0; t < seqlen; t++) {
            const float *vh = kvcache->v
                + ((size_t)layer * kvcache->max_tokens + t) * stride
                + h_kv * d_head;
            float s = scores[t];
            for (i = 0; i < d_head; i++) out_h[i] += s * vh[i];
        }
    }

    /* ── 6. Output projection: wo @ out → (via q as temp) → out ──────── */
    /*
     * q is no longer needed (RoPE and KV write are done), so we copy
     * the attention result there and matmul into `out`.
     */
    memcpy(q, out, (size_t)d_model * sizeof(float));
    kn->matmul(out, wo->data, q, d_model, d_model, 1, wo->dtype);
}
