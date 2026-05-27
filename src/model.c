/*
 * model.c — transformer forward pass
 * Implemented in Phase 5.
 * Included by src/cmol.c (unity build); do not compile standalone.
 */

#include "model.h"
/* attn.h will be used in Phase 5 */

/* Phase 5 — TODO */

float *cmol_model_forward(const cmol_model_t  *model,
                           struct cmol_session *session,
                           int32_t              token,
                           int                  pos) {
    (void)model; (void)session; (void)token; (void)pos;
    return NULL;
}

void cmol_rms_norm(const float *x, const float *w, float *out,
                   int n, float eps) {
    (void)x; (void)w; (void)out; (void)n; (void)eps;
}

void cmol_rope_apply(float *q, float *k,
                     int pos, int n_heads, int n_kv_heads,
                     int head_dim, float freq_base) {
    (void)q; (void)k; (void)pos; (void)n_heads;
    (void)n_kv_heads; (void)head_dim; (void)freq_base;
}

void cmol_swiglu(const float *gate, const float *up, float *out, int n) {
    (void)gate; (void)up; (void)out; (void)n;
}

void cmol_softmax(float *x, int n) {
    (void)x; (void)n;
}
