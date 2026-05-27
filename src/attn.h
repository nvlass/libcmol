/*
 * attn.h — grouped-query attention with KV cache
 * Internal header — not part of the public API.
 *
 * Callers (model.c) are responsible for allocating the scratch buffers
 * and arranging the scratch layout described in model.c.
 */

#ifndef CMOL_ATTN_H
#define CMOL_ATTN_H

#include "cmol_internal.h"

/*
 * cmol_attn_forward — one attention sub-layer for token at position `pos`.
 *
 * Steps performed:
 *   1. Q, K, V projections (quantised matmul via kn->matmul)
 *   2. Optional QK-norm (blk.{layer}.attn_q_norm / attn_k_norm, SmolLM3)
 *   3. RoPE — skipped for NoPE layers (no_rope_layer_interval)
 *   4. Write K, V into kvcache at position `pos`
 *   5. GQA scaled dot-product attention over [0 .. pos]
 *   6. Output projection (attn_output.weight) into `out`
 *
 * Buffer sizes (in floats):
 *   q       [hp->d_model]                  = n_heads * d_head
 *   k_buf   [hp->n_kv_heads * hp->d_head]
 *   v_buf   [hp->n_kv_heads * hp->d_head]
 *   scores  [kvcache->max_tokens]
 *   out     [hp->d_model]
 */
void cmol_attn_forward(const cmol_hparams_t *hp,
                        const cmol_kernels_t *kn,
                        int                   layer,
                        int                   pos,
                        const float          *xnorm,  /* [d_model] RMSNorm'd */
                        float                *q,      /* [d_model] scratch   */
                        float                *k_buf,  /* [kv_dim]  scratch   */
                        float                *v_buf,  /* [kv_dim]  scratch   */
                        float                *scores, /* [max_ctx] scratch   */
                        float                *out,    /* [d_model] result    */
                        cmol_kvcache_t       *kvcache,
                        cmol_tensor_t        *tensors,
                        int                   n_tensors);

#endif /* CMOL_ATTN_H */
