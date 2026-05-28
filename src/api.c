/*
 * api.c — public API implementation
 * Included by src/cmol.c (unity build); do not compile standalone.
 */

#include "../include/cmol.h"
#include "cmol_internal.h"
#include "arena.h"
#include "gguf.h"
#include "tokenizer.h"
#include "quant.h"    /* cmol_kernels_select */
#include "model.h"    /* cmol_model_forward  */
#include "sampler.h"  /* cmol_sample, cmol_rng_seed */

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* =========================================================================
 * Utilities
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
 * Returns the number of bytes needed for the arena (everything AFTER the
 * model struct itself, which is placed at the start of arena_buf).
 * ====================================================================== */

static size_t cmol__arena_size(const cmol_hparams_t *hp,
                                const cmol_config_t  *cfg,
                                size_t                n_tensors) {
    int ns    = cfg->max_sessions;
    int ctx   = cfg->max_ctx;
    int d     = hp->d_model;
    int kv    = hp->n_kv_heads * hp->d_head;
    int d_ffn = hp->d_ffn;
    int vocab = hp->vocab_size;
    int nl    = hp->n_layers;

    /* Tensor descriptors */
    size_t tensor_descs = n_tensors * sizeof(cmol_tensor_t);

    /* Tokenizer raw data (vocab strings, scores, token_type, merge arrays)
     * + tokenizer_build tables (decoded_vocab, vocab_sort_idx, merge_result,
     * msort_idx, decoded strings).
     * Upper bound: vocab_size * 80 covers all arenas for SmolLM2/3. */
    size_t tokenizer = (size_t)vocab * 80u;

    /* Session slot structs */
    size_t session_structs = (size_t)ns * sizeof(struct cmol_session);

    /* Per-session KV cache: K array + V array */
    size_t kv_per_session = 2u
        * (size_t)nl * (size_t)ctx * (size_t)kv * sizeof(float);

    /* Per-session scratch:
     *   x[d] + xnorm[d] + q[d] + k_buf[kv] + v_buf[kv] +
     *   scores[ctx] + attn_out[d] + ffn_gate[d_ffn] + ffn_up[d_ffn] +
     *   logits[vocab]                                                  */
    size_t scratch_per_session = (
        5u*(size_t)d + 2u*(size_t)kv + (size_t)ctx
        + 2u*(size_t)d_ffn + (size_t)vocab
    ) * sizeof(float);

    /* Per-session token buffer for encode during generate */
    size_t tokbuf_per_session = (size_t)ctx * sizeof(int32_t);

    /* Alignment padding: 64 bytes × generous factor */
    size_t padding = (size_t)(ns + 1) * 64u + 512u;

    return tensor_descs
         + tokenizer
         + session_structs
         + (size_t)ns * kv_per_session
         + (size_t)ns * scratch_per_session
         + (size_t)ns * tokbuf_per_session
         + padding;
}

/* =========================================================================
 * cmol_arena_estimate  (public utility)
 * ====================================================================== */

size_t cmol_arena_estimate(const char *gguf_path, const cmol_config_t *cfg) {
    static const cmol_config_t def = CMOL_DEFAULT_CONFIG;
    if (!gguf_path) return 0;
    if (!cfg) cfg = &def;

    cmol_hparams_t hp;
    size_t         n_tensors;
    if (cmol_gguf_peek(gguf_path, &hp, &n_tensors) != CMOL_OK) return 0;

    cmol_config_t c = *cfg;
    if (c.max_ctx <= 0 || c.max_ctx > hp.model_max_ctx) c.max_ctx = hp.model_max_ctx;
    if (c.max_sessions <= 0) c.max_sessions = 4;
    if (c.max_sessions > CMOL_MAX_SESSIONS) c.max_sessions = CMOL_MAX_SESSIONS;

    /* Add sizeof(model) to the estimate since the caller may use it as a
     * total-budget check against available RAM. */
    return sizeof(struct cmol_model)
         + cmol__arena_size(&hp, &c, n_tensors);
}

/* =========================================================================
 * cmol_load
 * ====================================================================== */

cmol_model_t *cmol_load(const char          *gguf_path,
                         const cmol_config_t *cfg,
                         cmol_err_t          *err) {
    static const cmol_config_t def = CMOL_DEFAULT_CONFIG;
    cmol_err_t rc;

    if (!gguf_path) { if (err) *err = CMOL_ERR_ARGS; return NULL; }
    if (!cfg) cfg = &def;

    /* ── 1. Peek: get hparams + tensor count without full parse ─────────── */
    cmol_hparams_t hp;
    size_t         n_tensors_peek;
    rc = cmol_gguf_peek(gguf_path, &hp, &n_tensors_peek);
    if (rc != CMOL_OK) { if (err) *err = rc; return NULL; }

    /* ── 2. Resolve / clamp config ───────────────────────────────────────── */
    cmol_config_t c = *cfg;
    if (c.max_ctx <= 0 || c.max_ctx > hp.model_max_ctx)
        c.max_ctx = hp.model_max_ctx;
    if (c.max_sessions <= 0)             c.max_sessions  = 4;
    if (c.max_sessions > CMOL_MAX_SESSIONS) c.max_sessions = CMOL_MAX_SESSIONS;
    if (c.prefill_chunk <= 0)            c.prefill_chunk = 512;

    /* ── 3. Compute allocation sizes ─────────────────────────────────────── */
    /* The model struct is placed at buf[0]; the arena starts immediately after
     * (aligned to 16 bytes). */
    size_t model_hdr  = (sizeof(struct cmol_model) + 15u) & ~15u;
    size_t arena_need = cmol__arena_size(&hp, &c, n_tensors_peek);
    size_t total_buf  = model_hdr + arena_need;

    /* ── 4. Single malloc ────────────────────────────────────────────────── */
    uint8_t *buf = (uint8_t *)malloc(total_buf);
    if (!buf) { if (err) *err = CMOL_ERR_OOM; return NULL; }
    memset(buf, 0, total_buf);

    /* ── 5. Model struct at buf[0]; arena starts after it ───────────────── */
    cmol_model_t *m = (cmol_model_t *)(void *)buf;
    m->arena_buf = buf;
    cmol_arena_init(&m->arena, buf + model_hdr, arena_need);
    m->cfg = c;

    /* ── 6. mmap the GGUF file ───────────────────────────────────────────── */
    rc = cmol_mmap_open(gguf_path, &m->mmap);
    if (rc != CMOL_OK) {
        free(buf);
        if (err) *err = rc;
        return NULL;
    }

    /* ── 7. Full GGUF parse ─────────────────────────────────────────────── */
    rc = cmol_gguf_parse(&m->mmap, &m->arena,
                          &m->hparams, &m->tensors, &m->n_tensors,
                          &m->tokenizer);
    if (rc != CMOL_OK) {
        cmol_mmap_close(&m->mmap);
        free(buf);
        if (err) *err = rc;
        return NULL;
    }

    /* ── 8. Build tokenizer runtime lookup tables ────────────────────────── */
    rc = cmol_tokenizer_build(&m->tokenizer, &m->arena);
    if (rc != CMOL_OK) {
        cmol_mmap_close(&m->mmap);
        free(buf);
        if (err) *err = rc;
        return NULL;
    }

    /* ── 9. SIMD kernel selection ────────────────────────────────────────── */
    m->kernels = cmol_kernels_select();

    /* ── 10. Allocate session pool ───────────────────────────────────────── */
    m->session_slots = (struct cmol_session *)cmol_arena_alloc_n(
            &m->arena, (size_t)c.max_sessions, sizeof(struct cmol_session));
    if (!m->session_slots) {
        cmol_mmap_close(&m->mmap);
        free(buf);
        if (err) *err = CMOL_ERR_OOM;
        return NULL;
    }

    {
        const cmol_hparams_t *fhp = &m->hparams;
        int d     = fhp->d_model;
        int kv    = fhp->n_kv_heads * fhp->d_head;
        int ctx   = c.max_ctx;
        int d_ffn = fhp->d_ffn;
        int vocab = fhp->vocab_size;
        int nl    = fhp->n_layers;
        int i;

        size_t kv_floats = (size_t)nl * (size_t)ctx * (size_t)kv;
        size_t scratch_floats = 5u*(size_t)d + 2u*(size_t)kv
                              + (size_t)ctx + 2u*(size_t)d_ffn + (size_t)vocab;

        for (i = 0; i < c.max_sessions; i++) {
            struct cmol_session *s = &m->session_slots[i];
            s->model = m;
            s->slot  = i;

            /* K cache */
            s->kvcache.k = (float *)cmol_arena_alloc_n(
                    &m->arena, kv_floats, sizeof(float));
            /* V cache */
            s->kvcache.v = (float *)cmol_arena_alloc_n(
                    &m->arena, kv_floats, sizeof(float));
            /* Scratch */
            s->scratch = (float *)cmol_arena_alloc_n(
                    &m->arena, scratch_floats, sizeof(float));
            /* Prompt token buffer */
            s->token_buf = (int32_t *)cmol_arena_alloc_n(
                    &m->arena, (size_t)ctx, sizeof(int32_t));

            if (!s->kvcache.k || !s->kvcache.v || !s->scratch || !s->token_buf) {
                cmol_mmap_close(&m->mmap);
                free(buf);
                if (err) *err = CMOL_ERR_OOM;
                return NULL;
            }

            s->kvcache.n_tokens  = 0;
            s->kvcache.max_tokens = ctx;
            s->scratch_size      = scratch_floats * sizeof(float);
            s->token_buf_cap     = ctx;
        }
    }

    /* ── 11. Pool bitmask: all slots free ────────────────────────────────── */
    m->pool_free = (c.max_sessions == 32)
                 ? 0xFFFFFFFFu
                 : (1u << c.max_sessions) - 1u;

    /* ── 12. Mutex ───────────────────────────────────────────────────────── */
    if (pthread_mutex_init(&m->pool_lock, NULL) != 0) {
        cmol_mmap_close(&m->mmap);
        free(buf);
        if (err) *err = CMOL_ERR_OOM;
        return NULL;
    }

    CMOL_LOGI(m,
        "cmol_load: arch=%s layers=%d heads=%d/%d d=%d ffn=%d "
        "vocab=%d ctx=%d sessions=%d kernel=%s",
        m->hparams.arch, m->hparams.n_layers,
        m->hparams.n_heads, m->hparams.n_kv_heads,
        m->hparams.d_model, m->hparams.d_ffn,
        m->hparams.vocab_size, c.max_ctx, c.max_sessions,
        m->kernels.name);

    if (err) *err = CMOL_OK;
    return m;
}

/* =========================================================================
 * cmol_free
 * ====================================================================== */

void cmol_free(cmol_model_t *m) {
    if (!m) return;
    pthread_mutex_destroy(&m->pool_lock);
    cmol_mmap_close(&m->mmap);
    free(m->arena_buf);
    /* m itself lives inside arena_buf, freed above */
}

/* =========================================================================
 * Session management
 * ====================================================================== */

cmol_session_t *cmol_session_acquire(cmol_model_t *m) {
    if (!m) return NULL;
    pthread_mutex_lock(&m->pool_lock);
    if (!m->pool_free) {
        pthread_mutex_unlock(&m->pool_lock);
        return NULL;
    }
    int slot = __builtin_ctz(m->pool_free);
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
    /* scratch intentionally not zeroed — overwritten on next forward pass */
}

/* =========================================================================
 * Tokenizer (public wrappers)
 * ====================================================================== */

int cmol_encode(cmol_model_t *m, const char *text,
                int32_t *out, int out_cap) {
    if (!m || !text || !out || out_cap <= 0) return CMOL_ERR_ARGS;
    return cmol_tokenizer_encode(&m->tokenizer, text, out, out_cap,
                                 m->tokenizer.add_bos);
}

const char *cmol_decode_token(cmol_model_t *m, int32_t token_id) {
    if (!m) return NULL;
    return cmol_tokenizer_decode_token(&m->tokenizer, token_id);
}

/* =========================================================================
 * ChatML prompt formatting helpers
 * ====================================================================== */

#define CMOL__DEFAULT_SYSTEM \
    "You are a helpful AI assistant named SmolLM, trained by Hugging Face"

int cmol_format_chatml(const char *system, const char *user,
                        char *buf, size_t buf_cap) {
    int n;
    if (!user) return (int)CMOL_ERR_ARGS;

    /* NULL  → SmolLM default system prompt
     * ""    → omit system turn
     * other → use verbatim                  */
    const char *sys = (system == NULL) ? CMOL__DEFAULT_SYSTEM : system;

    if (sys[0] != '\0') {
        n = snprintf(buf, buf_cap,
            "<|im_start|>system\n%s<|im_end|>\n"
            "<|im_start|>user\n%s<|im_end|>\n"
            "<|im_start|>assistant\n",
            sys, user);
    } else {
        n = snprintf(buf, buf_cap,
            "<|im_start|>user\n%s<|im_end|>\n"
            "<|im_start|>assistant\n",
            user);
    }

    if (n < 0) return (int)CMOL_ERR_ARGS;
    if (buf && buf_cap > 0 && (size_t)n >= buf_cap) return (int)CMOL_ERR_TRUNC;
    return n;
}

int cmol_format_chatml_turn(const char *user, char *buf, size_t buf_cap) {
    int n;
    if (!user) return (int)CMOL_ERR_ARGS;

    /* Close the previous (open) assistant turn, then open the new user turn. */
    n = snprintf(buf, buf_cap,
        "<|im_end|>\n"
        "<|im_start|>user\n%s<|im_end|>\n"
        "<|im_start|>assistant\n",
        user);

    if (n < 0) return (int)CMOL_ERR_ARGS;
    if (buf && buf_cap > 0 && (size_t)n >= buf_cap) return (int)CMOL_ERR_TRUNC;
    return n;
}

/* =========================================================================
 * cmol_generate
 *
 * Pipeline:
 *   1. Encode prompt → token IDs (written into session->token_buf).
 *   2. Prefill: call cmol_model_forward for each prompt token,
 *      building the KV cache.  Logits are discarded except after the
 *      final prompt token.
 *   3. Generation loop: sample → decode → callback → repeat.
 *      Stop on EOS, max_new_tokens, full context, or callback abort.
 * ====================================================================== */

cmol_err_t cmol_generate(cmol_session_t          *s,
                          const char              *prompt,
                          const cmol_gen_params_t *params,
                          cmol_token_cb_t          on_token,
                          void                    *userdata) {
    static const cmol_gen_params_t def_params = CMOL_DEFAULT_PARAMS;

    if (!s || !prompt) return CMOL_ERR_ARGS;
    if (!params) params = &def_params;

    cmol_model_t          *m  = s->model;
    const cmol_hparams_t  *hp = &m->hparams;

    /* Context capacity check */
    if (s->kvcache.n_tokens >= s->kvcache.max_tokens)
        return CMOL_ERR_CTX_FULL;

    /* ── 1. Encode prompt ────────────────────────────────────────────────── */
    int n_prompt = cmol_tokenizer_encode(
            &m->tokenizer, prompt,
            s->token_buf, s->token_buf_cap,
            m->tokenizer.add_bos);
    if (n_prompt < 0)  return (cmol_err_t)n_prompt;
    if (n_prompt == 0) return CMOL_OK;

    /* Clamp to available context */
    {
        int avail = s->kvcache.max_tokens - s->kvcache.n_tokens;
        if (n_prompt > avail) n_prompt = avail;
    }

    /* ── 2. Prefill: forward pass for each prompt token ─────────────────── */
    float  *logits = NULL;
    int     pos    = s->kvcache.n_tokens;
    int     i;

    for (i = 0; i < n_prompt; i++) {
        logits = cmol_model_forward(m, s, s->token_buf[i], pos);
        pos++;
    }
    s->kvcache.n_tokens = pos;

    if (!logits) return CMOL_ERR_INVALID; /* tensor missing */

    /* ── 3. Generation loop ──────────────────────────────────────────────── */
    int     max_new = params->max_new_tokens;
    if (max_new < 0) max_new = s->kvcache.max_tokens; /* generate until EOS */

    /* Seed RNG once for this call */
    uint64_t rng[4];
    cmol_rng_seed(rng, params->seed);

    /* Repetition-penalty ring buffer — seed with tail of prompt tokens */
    int32_t repeat_buf[CMOL_REPEAT_BUF];
    int     repeat_head = 0;   /* next write position (wraps)  */
    int     repeat_fill = 0;   /* entries currently valid       */
    {
        int last_n = params->repeat_last_n;
        if (last_n <= 0 || last_n > CMOL_REPEAT_BUF) last_n = CMOL_REPEAT_BUF;
        int seed_n = n_prompt < last_n ? n_prompt : last_n;
        for (i = 0; i < seed_n; i++)
            repeat_buf[i] = s->token_buf[n_prompt - seed_n + i];
        repeat_head = seed_n % CMOL_REPEAT_BUF;
        repeat_fill = seed_n;
    }

    int n_generated = 0;

    for (;;) {
        /* ── Repetition penalty (applied before temperature / softmax) ── */
        if (params->repeat_penalty > 1.0f && repeat_fill > 0)
            cmol_apply_repeat_penalty(logits, hp->vocab_size,
                                      repeat_buf, repeat_fill,
                                      params->repeat_penalty);

        /* ── Sample next token ─────────────────────────────────────────── */
        int32_t next_tok = cmol_sample(logits, hp->vocab_size, params, rng);

        /* ── Decode + fire callback ────────────────────────────────────── */
        int is_eos = (m->tokenizer.eos_id >= 0 &&
                      next_tok == m->tokenizer.eos_id);

        if (on_token) {
            const char *piece = cmol_tokenizer_decode_token(&m->tokenizer,
                                                             next_tok);
            size_t len = piece ? strlen(piece) : 0;
            int stop = on_token(piece ? piece : "", len, is_eos, userdata);
            if (stop) break; /* caller aborted */
        }

        if (is_eos) break;

        /* ── Push generated token into repeat window ──────────────────── */
        repeat_buf[repeat_head] = next_tok;
        repeat_head = (repeat_head + 1) % CMOL_REPEAT_BUF;
        if (repeat_fill < CMOL_REPEAT_BUF) repeat_fill++;

        n_generated++;
        if (max_new > 0 && n_generated >= max_new) break;

        /* ── Context full? ─────────────────────────────────────────────── */
        if (pos >= s->kvcache.max_tokens) break;

        /* ── Forward pass for the sampled token ───────────────────────── */
        logits = cmol_model_forward(m, s, next_tok, pos);
        pos++;
        s->kvcache.n_tokens = pos;

        if (!logits) return CMOL_ERR_INVALID;
    }

    return CMOL_OK;
}
