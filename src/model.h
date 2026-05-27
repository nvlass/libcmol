/*
 * model.h — transformer forward pass
 * Implemented in Phase 5.
 * Internal header — not part of the public API.
 */

#ifndef CMOL_MODEL_H
#define CMOL_MODEL_H

#include "cmol_internal.h"

/*
 * cmol_model_forward — full forward pass for one token at position `pos`.
 *
 * Embedding lookup → N × (RMSNorm + Attention + RMSNorm + SwiGLU FFN)
 *                  → final RMSNorm → LM head
 *
 * Returns a pointer into `session->scratch` holding the logit vector
 * [vocab_size].  Valid until the next call with the same session.
 */
float *cmol_model_forward(const cmol_model_t  *model,
                           struct cmol_session *session,
                           int32_t              token,
                           int                  pos);

/* ---- Primitive ops exposed for unit testing --------------------------- */

/* in-place RMSNorm: out[i] = x[i] / rms(x) * w[i] */
void cmol_rms_norm(const float *x, const float *w, float *out,
                   int n, float eps);

/* in-place rotary position embedding applied to Q and K buffers */
void cmol_rope_apply(float *q, float *k,
                     int pos, int n_heads, int n_kv_heads,
                     int head_dim, float freq_base);

/* SwiGLU: out[i] = silu(gate[i]) * up[i]  where silu(x) = x * sigmoid(x) */
void cmol_swiglu(const float *gate, const float *up, float *out, int n);

/* softmax in-place over `n` elements */
void cmol_softmax(float *x, int n);

#endif /* CMOL_MODEL_H */
