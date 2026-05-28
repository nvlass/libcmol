/*
 * cmol.h — libcmol public API
 *
 * Minimal embeddable inference library for SmolLM3 and smaller variants.
 *
 * Single-header release usage:
 *   Define CMOL_IMPLEMENTATION in exactly one translation unit before
 *   including this file.  In all other files, include without the define.
 *
 *     // myapp.c
 *     #define CMOL_IMPLEMENTATION
 *     #include "cmol.h"
 *
 * Multi-file (source tree) usage:
 *   Include cmol.h normally and link against libcmol.a or cmol.o.
 *
 * Minimal REPL example:
 *
 *   static int print_token(const char *p, size_t n, int eos, void *_) {
 *       if (!eos) fwrite(p, 1, n, stdout);
 *       return 0;
 *   }
 *   int main(void) {
 *       cmol_config_t    cfg = CMOL_DEFAULT_CONFIG;
 *       cmol_gen_params_t p  = CMOL_DEFAULT_PARAMS;
 *       cmol_model_t  *m = cmol_load("smollm3.gguf", &cfg, NULL);
 *       cmol_session_t *s = cmol_session_acquire(m);
 *       char line[512];
 *       while (fgets(line, sizeof line, stdin))
 *           cmol_generate(s, line, &p, print_token, NULL);
 *       cmol_session_release(s);
 *       cmol_free(m);
 *   }
 */

#ifndef CMOL_H
#define CMOL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Version
 * ====================================================================== */

#define CMOL_VERSION_MAJOR  0
#define CMOL_VERSION_MINOR  1
#define CMOL_VERSION_PATCH  0
#define CMOL_VERSION_STRING "0.1.0"

/* =========================================================================
 * Error codes
 *
 * All negative values are errors; CMOL_OK == 0.
 * Functions that return int use negative cmol_err_t values for errors and
 * non-negative values for counts (e.g. cmol_encode returns token count).
 * ====================================================================== */

typedef enum {
    CMOL_OK              =  0, /* success                                  */
    CMOL_ERR_OOM         = -1, /* arena or system out of memory            */
    CMOL_ERR_IO          = -2, /* file open / mmap failed                  */
    CMOL_ERR_INVALID     = -3, /* malformed or corrupt GGUF                */
    CMOL_ERR_UNSUPPORTED = -4, /* unsupported quant type or architecture   */
    CMOL_ERR_NO_SESSION  = -5, /* session pool exhausted — retry later     */
    CMOL_ERR_ARGS        = -6, /* NULL or out-of-range argument            */
    CMOL_ERR_CTX_FULL    = -7, /* KV cache full — call cmol_session_reset  */
    CMOL_ERR_TRUNC       = -8, /* output buffer too small (encode)         */
} cmol_err_t;

/* =========================================================================
 * Logging  (optional)
 *
 * Supply a log callback in cmol_config_t to receive diagnostic messages.
 * NULL = silent.
 * ====================================================================== */

#define CMOL_LOG_ERROR  0
#define CMOL_LOG_WARN   1
#define CMOL_LOG_INFO   2
#define CMOL_LOG_DEBUG  3

typedef void (*cmol_log_fn_t)(int level, const char *msg, void *userdata);

/* =========================================================================
 * Configuration  (passed to cmol_load)
 * ====================================================================== */

typedef struct {
    int           max_ctx;       /* max context window tokens (≤ 8192)    */
    int           max_sessions;  /* concurrent sessions to pre-allocate   */
    int           prefill_chunk; /* internal prefill batch size (tokens)  */
    cmol_log_fn_t log_fn;        /* optional log callback; NULL = silent  */
    void         *log_ud;        /* passed through to log_fn              */
} cmol_config_t;

/* Sensible defaults for interactive / REPL use */
#define CMOL_DEFAULT_CONFIG \
    { .max_ctx = 2048, .max_sessions = 4, .prefill_chunk = 512, \
      .log_fn = NULL, .log_ud = NULL }

/* =========================================================================
 * Generation parameters  (passed per-call to cmol_generate)
 * ====================================================================== */

typedef struct {
    float        temperature;    /* 0.0 = greedy                          */
    float        top_p;          /* nucleus cutoff; 1.0 = disabled        */
    int          top_k;          /* 0 = disabled                          */
    int          max_new_tokens; /* -1 = generate until EOS               */
    unsigned int seed;           /* 0 = non-deterministic                 */
} cmol_gen_params_t;

#define CMOL_DEFAULT_PARAMS \
    { .temperature = 0.8f, .top_p = 0.95f, .top_k = 40, \
      .max_new_tokens = 256, .seed = 0 }

/* =========================================================================
 * Opaque handle types
 * ====================================================================== */

typedef struct cmol_model   cmol_model_t;
typedef struct cmol_session cmol_session_t;

/* =========================================================================
 * Token callback
 *
 * Called once per generated token piece (a UTF-8 text fragment).
 *
 *   piece    — NOT null-terminated; valid for exactly `len` bytes.
 *   len      — byte length of this piece (may be 0 for control tokens).
 *   is_eos   — non-zero when this is the end-of-sequence token.
 *
 * Return 0 to continue generation.
 * Return non-zero to abort early (cmol_generate still returns CMOL_OK).
 * ====================================================================== */

typedef int (*cmol_token_cb_t)(const char *piece, size_t len,
                                int is_eos, void *userdata);

/* =========================================================================
 * Model lifecycle
 * ====================================================================== */

/*
 * cmol_load — load a GGUF model file and allocate all runtime memory.
 *
 * Performs a single malloc to create the arena, then never allocates again.
 * Returns NULL on failure; sets *err (if non-NULL) to the error code.
 *
 * Fails early with CMOL_ERR_OOM if the arena required for the given config
 * cannot be satisfied — use cmol_arena_estimate() to check first on
 * memory-constrained targets (e.g. Raspberry Pi Zero W).
 */
cmol_model_t *cmol_load(const char          *gguf_path,
                         const cmol_config_t *cfg,
                         cmol_err_t          *err);

/*
 * cmol_free — release all resources acquired by cmol_load.
 * Safe to call with NULL.
 */
void cmol_free(cmol_model_t *m);

/* =========================================================================
 * Session management
 *
 * One session = one conversation context with its own KV cache.
 * Sessions are pre-allocated from the arena at cmol_load() time.
 * Each concurrent inference thread needs its own session.
 * ====================================================================== */

/*
 * cmol_session_acquire — obtain a session from the pool.
 *
 * Thread-safe.  Returns NULL if all slots are busy (CMOL_ERR_NO_SESSION);
 * the caller is responsible for retrying.
 */
cmol_session_t *cmol_session_acquire(cmol_model_t *m);

/*
 * cmol_session_release — return a session to the pool.
 * The session must not be used after this call.
 */
void cmol_session_release(cmol_session_t *s);

/*
 * cmol_session_reset — clear the KV cache without releasing the slot.
 * Use this to start a fresh conversation without a release/acquire cycle.
 */
void cmol_session_reset(cmol_session_t *s);

/* =========================================================================
 * Inference
 * ====================================================================== */

/*
 * cmol_generate — run inference on `prompt`, streaming tokens via on_token.
 *
 *   s         — session (must be acquired and not in use on another thread)
 *   prompt    — null-terminated UTF-8 input text
 *   params    — generation parameters; NULL uses CMOL_DEFAULT_PARAMS
 *   on_token  — called for each generated piece; return non-zero to abort
 *   userdata  — passed through to on_token unchanged
 *
 * Returns CMOL_OK on success (including early abort via callback).
 * The prompt tokens are appended to the session's KV cache; call
 * cmol_session_reset() to start a new conversation.
 */
cmol_err_t cmol_generate(cmol_session_t          *s,
                          const char              *prompt,
                          const cmol_gen_params_t *params,
                          cmol_token_cb_t          on_token,
                          void                    *userdata);

/* =========================================================================
 * Tokenizer
 * ====================================================================== */

/*
 * cmol_encode — convert null-terminated UTF-8 text to token IDs.
 *
 * Writes up to `out_cap` token IDs into `out`.
 * Returns the number of tokens written (>= 0), or a negative cmol_err_t.
 * Returns CMOL_ERR_TRUNC if out_cap was too small (partial result written).
 */
int cmol_encode(cmol_model_t *m,
                const char   *text,
                int32_t      *out,
                int           out_cap);

/*
 * cmol_decode_token — return the UTF-8 string for a single token ID.
 *
 * The returned pointer is valid for the lifetime of the model; do not free.
 * Returns NULL for out-of-range IDs.
 */
const char *cmol_decode_token(cmol_model_t *m, int32_t token_id);

/* =========================================================================
 * Utilities
 * ====================================================================== */

/* Human-readable description of an error code. */
const char *cmol_strerror(cmol_err_t err);

/* Version string, e.g. "0.1.0". */
const char *cmol_version(void);

/*
 * cmol_arena_estimate — compute the arena bytes required to load a model
 * with the given config, without actually loading it.
 *
 * Reads only the GGUF header (fast).  Returns 0 on error.
 * Useful for pre-flight checks on memory-constrained targets.
 */
size_t cmol_arena_estimate(const char *gguf_path, const cmol_config_t *cfg);

/* =========================================================================
 * ChatML prompt formatting helpers  (SmolLM2 / SmolLM3 style)
 *
 * These are pure string utilities — no model handle required.
 * They are entirely optional; you can always hand-format prompts and pass
 * them directly to cmol_generate().
 *
 * Both functions behave like snprintf:
 *   - Write at most buf_cap bytes (including the NUL terminator).
 *   - Return the number of bytes that would have been written had buf_cap
 *     been unlimited (not counting the NUL).
 *   - Return CMOL_ERR_TRUNC (negative) when buf_cap is too small; the
 *     buffer is still NUL-terminated.
 *   - Pass buf=NULL / buf_cap=0 to probe the required size.
 * ====================================================================== */

/*
 * cmol_format_chatml — format the opening turn of a ChatML conversation.
 *
 *   system   — system message text.
 *              NULL  → SmolLM default ("You are a helpful AI assistant …")
 *              ""    → omit the system turn entirely
 *              other → use verbatim as the system message
 *   user     — user message text (required, must not be NULL)
 *
 * Output (with system):
 *   <|im_start|>system\n{system}<|im_end|>\n
 *   <|im_start|>user\n{user}<|im_end|>\n
 *   <|im_start|>assistant\n
 *
 * Output (system == ""):
 *   <|im_start|>user\n{user}<|im_end|>\n
 *   <|im_start|>assistant\n
 */
int cmol_format_chatml(const char *system, const char *user,
                        char *buf, size_t buf_cap);

/*
 * cmol_format_chatml_turn — format a subsequent user turn in an ongoing
 * session.
 *
 * Our generation loop stops before writing the EOS token (<|im_end|>) into
 * the KV cache, so each continuation must first close the previous assistant
 * turn and then open a new user turn.
 *
 * Output:
 *   <|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n
 */
int cmol_format_chatml_turn(const char *user, char *buf, size_t buf_cap);

/* =========================================================================
 * Single-header implementation
 * ====================================================================== */

#ifdef CMOL_IMPLEMENTATION
/* Injected by `make amalgamate` — do not edit here. */
#endif /* CMOL_IMPLEMENTATION */

#ifdef __cplusplus
}
#endif

#endif /* CMOL_H */
