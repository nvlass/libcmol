/*
 * attn.c — grouped-query attention with KV cache
 * Implemented in Phase 5.
 * Included by src/cmol.c (unity build); do not compile standalone.
 */

#include "attn.h"

/* Phase 5 — TODO */

void cmol_attn_forward(const cmol_hparams_t *hp,
                        const cmol_kernels_t *kn,
                        int                   layer,
                        int                   pos,
                        const float          *x_in,
                        float                *x_out,
                        float                *scratch,
                        cmol_kvcache_t       *kvcache,
                        cmol_tensor_t        *tensors,
                        int                   n_tensors) {
    (void)hp; (void)kn; (void)layer; (void)pos;
    (void)x_in; (void)x_out; (void)scratch;
    (void)kvcache; (void)tensors; (void)n_tensors;
}
