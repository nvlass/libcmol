/*
 * api.c — public API implementation
 * Phases 2 and 7 (lifecycle, session management, generate).
 * Included by src/cmol.c (unity build); do not compile standalone.
 */

#include "../include/cmol.h"
#include "cmol_internal.h"
#include "gguf.h"        /* cmol_gguf_peek, cmol_gguf_parse             */
#include "tokenizer.h"   /* cmol_tokenizer_encode, _decode_token        */
/* Phase 7 will also use: arena.h, quant.h, model.h, sampler.h */

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* =========================================================================
 * Utilities  (available immediately)
 * ====================================================================== */

const char *cmol_strerror(cmol_err_t err) {
    switch (err) {
        case CMOL_OK:              return "success";
        case CMOL_ERR_OOM:         return "out of memory";
        case CMOL_ERR_IO:          return "I/O error";
        case CMOL_ERR_INVALID:     return "invalid or corrupt GGUF file";
        case CMOL_ERR_UNSUPPORTED: return "unsupported quantisation type or architecture";
        case CMOL_ERR_NO_SESSION:  return "session pool exhausted — retry later";
        case CMOL_ERR_ARGS:        return "invalid argument (NULL or out of range)";
        case CMOL_ERR_CTX_FULL:    return "KV cache full — call cmol_session_reset()";
        case CMOL_ERR_TRUNC:       return "output buffer too small (result truncated)";
        default:                   return "unknown error";
    }
}

const char *cmol_version(void) {
    return CMOL_VERSION_STRING;
}

void cmol_log(const cmol_model_t *m, int level, const char *fmt, ...) {
    if (!m || !m->cfg.log_fn) return;
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    m->cfg.log_fn(level, buf, m->cfg.log_ud);
}

/* =========================================================================
 * Arena size calculation
 *
 * Computes the total arena bytes needed for a given model + config.
 * All sizes are conservative (round up).
 * ====================================================================== */

static size_t cmol__arena_size(const cmol_hparams_t *hp,
                                const cmol_config_t  *cfg) {
    int ns  = cfg->max_sessions;
    int ctx = cfg->max_ctx;

    /* KV cache: 2 buffers (K and V) × layers × ctx × kv_heads × d_head × float */
    size_t kv_per_session = 2u
        * (size_t)hp->n_layers
        * (size_t)ctx
        * (size_t)hp->n_kv_heads
        * (size_t)hp->d_head
        * sizeof(float);

    /* Scratch: activation buffer sized for one prefill chunk.
     * Needs to hold at least: d_model + d_ffn*2 + n_heads*d_head
     * (conservative: 4 × d_model as a safe upper bound) */
    size_t scratch_per_session = 4u * (size_t)hp->d_model * sizeof(float)
                               + (size_t)hp->vocab_size   * sizeof(float);

    /* Tensor descriptor array */
    size_t tensor_descs = 512u * sizeof(cmol_tensor_t); /* 512 tensors max */

    /* Tokenizer strings, merge rules (rough upper bound) */
    size_t tokenizer = (size_t)hp->vocab_size * 32u    /* avg token len   */
                     + (size_t)hp->vocab_size * 8u;    /* merge pairs     */

    /* Session slot structs themselves */
    size_t session_structs = (size_t)ns * sizeof(struct cmol_session);

    /* Alignment padding: 64 bytes per allocation × generous factor */
    size_t padding = 4096u;

    return  tensor_descs
          + tokenizer
          + session_structs
          + (size_t)ns * kv_per_session
          + (size_t)ns * scratch_per_session
          + padding;
}

/* =========================================================================
 * cmol_arena_estimate  (public utility)
 * ====================================================================== */

size_t cmol_arena_estimate(const char *gguf_path, const cmol_config_t *cfg) {
    if (!gguf_path || !cfg) return 0;

    cmol_hparams_t hp;
    size_t         n_tensors;
    if (cmol_gguf_peek(gguf_path, &hp, &n_tensors) != CMOL_OK) return 0;

    /* Clamp max_ctx to what the model supports */
    cmol_config_t c = *cfg;
    if (c.max_ctx > hp.model_max_ctx) c.max_ctx = hp.model_max_ctx;

    return cmol__arena_size(&hp, &c);
}

/* =========================================================================
 * cmol_load / cmol_free
 * Phase 7 — TODO: full implementation
 * ====================================================================== */

cmol_model_t *cmol_load(const char          *gguf_path,
                         const cmol_config_t *cfg,
                         cmol_err_t          *err) {
    (void)gguf_path; (void)cfg;
    if (err) *err = CMOL_ERR_UNSUPPORTED;
    return NULL; /* Phase 7 */
}

void cmol_free(cmol_model_t *m) {
    if (!m) return;
    cmol_mmap_close(&m->mmap);
    pthread_mutex_destroy(&m->pool_lock);
    free(m->arena_buf);
    /* m itself lives in arena_buf, so it is freed above */
}

/* =========================================================================
 * Session management
 * Phase 7 — TODO: full implementation
 * ====================================================================== */

cmol_session_t *cmol_session_acquire(cmol_model_t *m) {
    if (!m) return NULL;
    pthread_mutex_lock(&m->pool_lock);
    if (!m->pool_free) {
        pthread_mutex_unlock(&m->pool_lock);
        return NULL; /* CMOL_ERR_NO_SESSION — caller retries */
    }
    /* Find lowest free slot via bit scan */
    int slot = __builtin_ctz(m->pool_free); /* GCC/Clang; Phase 7: add MSVC fallback */
    m->pool_free &= ~(1u << slot);
    pthread_mutex_unlock(&m->pool_lock);
    return &m->session_slots[slot];
}

void cmol_session_release(cmol_session_t *s) {
    if (!s) return;
    cmol_model_t *m = s->model;
    pthread_mutex_lock(&m->pool_lock);
    m->pool_free |= (1u << s->slot);
    pthread_mutex_unlock(&m->pool_lock);
}

void cmol_session_reset(cmol_session_t *s) {
    if (!s) return;
    s->kvcache.n_tokens = 0;
    /* scratch buffer intentionally not zeroed — will be overwritten */
}

/* =========================================================================
 * Tokenizer (public wrappers)
 * ====================================================================== */

int cmol_encode(cmol_model_t *m, const char *text,
                int32_t *out, int out_cap) {
    if (!m || !text || !out || out_cap <= 0) return CMOL_ERR_ARGS;
    return cmol_tokenizer_encode(&m->tokenizer, text, out, out_cap, /*add_bos=*/1);
}

const char *cmol_decode_token(cmol_model_t *m, int32_t token_id) {
    if (!m) return NULL;
    return cmol_tokenizer_decode_token(&m->tokenizer, token_id);
}

/* =========================================================================
 * cmol_generate
 * Phase 7 — TODO: prefill loop + generation loop
 * ====================================================================== */

cmol_err_t cmol_generate(cmol_session_t          *s,
                          const char              *prompt,
                          const cmol_gen_params_t *params,
                          cmol_token_cb_t          on_token,
                          void                    *userdata) {
    (void)s; (void)prompt; (void)params; (void)on_token; (void)userdata;
    return CMOL_ERR_UNSUPPORTED; /* Phase 7 */
}
