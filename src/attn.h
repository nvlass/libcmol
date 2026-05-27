/*
 * attn.h — multi-head / grouped-query attention with KV cache
 * Implemented in Phase 5.
 * Internal header — not part of the public API.
 */

#ifndef CMOL_ATTN_H
#define CMOL_ATTN_H

#include "cmol_internal.h"

/*
 * cmol_attn_forward — run one attention sub-layer for position `pos`.
 *
 * Reads Q/K/V weight tensors by name lookup, projects `x_in`, applies RoPE,
 * writes K and V into the session KV cache, computes GQA attention, projects
 * output into `x_out`.
 *
 *   layer   — transformer block index (0-based)
 *   pos     — current token position in the context
 *   x_in    — input activation  [d_model], float32
 *   x_out   — output activation [d_model], float32
 *   scratch — temporary buffer (>= d_model * n_heads * sizeof(float))
 */
void cmol_attn_forward(const cmol_hparams_t *hp,
                        const cmol_kernels_t *kn,
                        int                   layer,
                        int                   pos,
                        const float          *x_in,
                        float                *x_out,
                        float                *scratch,
                        cmol_kvcache_t       *kvcache,
                        cmol_tensor_t        *tensors,
                        int                   n_tensors);

#endif /* CMOL_ATTN_H */
