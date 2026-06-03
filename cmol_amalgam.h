/*
 * cmol.h — libcmol single-header release
 *
 * AUTO-GENERATED — do not edit.
 * Regenerate with:  make amalgamate
 *
 * Usage:
 *   In exactly ONE translation unit:
 *       #define CMOL_IMPLEMENTATION
 *       #include "cmol.h"
 *   In all other files:
 *       #include "cmol.h"
 */

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
    float        repeat_penalty; /* > 1.0 penalises recently-seen tokens; */
                                 /* 1.0 = disabled                        */
    int          repeat_last_n;  /* window of recent tokens to penalise;  */
                                 /* clamped to CMOL_REPEAT_BUF (128)      */
} cmol_gen_params_t;

#define CMOL_DEFAULT_PARAMS \
    { .temperature = 0.8f, .top_p = 0.95f, .top_k = 40, \
      .max_new_tokens = 256, .seed = 0, \
      .repeat_penalty = 1.1f, .repeat_last_n = 64 }

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
 *              NULL  → omit the system turn entirely (same as "")
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



#ifdef __cplusplus
}
#endif

#ifdef CMOL_IMPLEMENTATION

/* ===== src/platform.h ===== */
/*
 * platform.h — OS and CPU abstraction layer for libcmol
 *
 * Provides:
 *   - Memory-mapped file I/O  (POSIX mmap / Windows stubs)
 *   - CPU feature detection   (x86 CPUID / ARM HWCAP / AArch64 mandatory)
 *
 * Internal header — not part of the public API.
 */

#include <stddef.h>

/* =========================================================================
 * Memory-mapped file
 * ====================================================================== */

typedef struct {
    void  *data;   /* start of mapped region (read-only)  */
    size_t size;   /* total size in bytes                  */
} cmol_mmap_t;

/*
 * cmol_mmap_open — open `path` and map it into read-only memory.
 * On success, populates *out and returns CMOL_OK.
 * The file descriptor is closed before returning (mapping keeps the data).
 */
cmol_err_t cmol_mmap_open(const char *path, cmol_mmap_t *out);

/*
 * cmol_mmap_close — unmap a previously opened region.
 * Safe to call with a zeroed or already-closed cmol_mmap_t.
 */
void cmol_mmap_close(cmol_mmap_t *m);

/* =========================================================================
 * CPU feature detection
 *
 * Results are used in cmol_kernels_select() (quant.c) to set the SIMD
 * function pointer table once at cmol_load() time.
 *
 * Targets:
 *   x86-64   — CPUID leaf 7 subleaf 0 (AVX2 / AVX-512F)
 *   AArch64  — ASIMD is mandatory; neon is always 1
 *   ARMv7    — Linux AT_HWCAP / HWCAP_NEON
 *   ARMv6    — no NEON (Raspberry Pi Zero W); all fields 0
 *   Other    — all fields 0; scalar fallback used
 * ====================================================================== */

typedef struct {
    int avx512f; /* x86: AVX-512 Foundation                */
    int avx2;    /* x86: Advanced Vector Extensions 2      */
    int neon;    /* ARM: NEON (ARMv7) / ASIMD (AArch64)    */
} cmol_cpu_t;

/*
 * cmol_detect_cpu — query hardware capabilities.
 * Safe to call multiple times; result is idempotent.
 */
cmol_cpu_t cmol_detect_cpu(void);

/* ===== src/cmol_internal.h ===== */
/*
 * cmol_internal.h — internal types for libcmol
 *
 * Full struct definitions for the opaque public handles (cmol_model_t,
 * cmol_session_t) plus all shared internal types.
 *
 * Internal header — not part of the public API.
 * Include AFTER cmol.h and platform.h.
 */

#include <stdint.h>
#include <stddef.h>
#include <pthread.h>


/* =========================================================================
 * GGUF quantisation type tags
 *
 * We only list the types we handle; the full GGUF spec has more.
 * ====================================================================== */

typedef enum {
    CMOL_DTYPE_F32  = 0,
    CMOL_DTYPE_F16  = 1,
    CMOL_DTYPE_Q4_0 = 2,
    CMOL_DTYPE_Q4_1 = 3,
    CMOL_DTYPE_Q5_0 = 6,  /* supported: 5-bit blocks of 32, 22 bytes    */
    CMOL_DTYPE_Q8_0 = 8,  /* supported: 8-bit blocks of 32              */
    CMOL_DTYPE_Q4_K = 12, /* supported: 4-bit blocks of 256 (Q4_K_M)   */
    CMOL_DTYPE_Q6_K = 14,
    CMOL_DTYPE_COUNT,
} cmol_dtype_t;

/* =========================================================================
 * Tensor descriptor
 *
 * Points directly into the mmap region — no copies of weight data.
 * ====================================================================== */

#define CMOL_MAX_DIMS        4
#define CMOL_MAX_TENSOR_NAME 96

typedef struct {
    char         name[CMOL_MAX_TENSOR_NAME];
    int          n_dims;
    int64_t      shape[CMOL_MAX_DIMS]; /* shape[0] is the innermost dim  */
    cmol_dtype_t dtype;
    uint64_t     file_offset;          /* byte offset from mmap base     */
    void        *data;                 /* = mmap.data + file_offset      */
    size_t       n_bytes;
} cmol_tensor_t;

/* =========================================================================
 * Model hyperparameters  (extracted from GGUF metadata)
 * ====================================================================== */

typedef struct {
    int     n_layers;
    int     n_heads;               /* query heads                         */
    int     n_kv_heads;            /* key/value heads (< n_heads for GQA) */
    int     d_model;               /* embedding dimension                 */
    int     d_head;                /* d_model / n_heads                   */
    int     d_ffn;                 /* feed-forward inner dimension        */
    int     vocab_size;
    int     model_max_ctx;         /* max context as reported in the GGUF */
    float   rope_freq_base;
    float   rms_norm_eps;
    int32_t bos_token_id;
    int32_t eos_token_id;
    int     tie_embeddings;        /* 1 if lm_head == token_embd          */

    /* SmolLM3 / NoPE hybrid: every no_rope_layer_interval-th layer
     * skips RoPE entirely.  0 means standard RoPE on all layers.        */
    int     no_rope_layer_interval;

    /* YARN context extension (inference-time RoPE scaling).
     * 0 = disabled (use native model_max_ctx).
     * Phase 5 will implement the actual YARN computation.               */
    int     yarn_factor_x100;      /* factor * 100, e.g. 200 for 2.0     */

    /* Architecture string from general.architecture, e.g. "llama".     */
    char    arch[32];
} cmol_hparams_t;

/* =========================================================================
 * BPE tokenizer  (loaded from GGUF metadata, stored in arena)
 * ====================================================================== */

/* Token type flags matching GGUF tokenizer.ggml.token_type */
typedef enum {
    CMOL_TOKEN_NORMAL  = 1,
    CMOL_TOKEN_UNKNOWN = 2,
    CMOL_TOKEN_CONTROL = 3,
    CMOL_TOKEN_BYTE    = 6,
} cmol_token_type_t;

/* Tokenizer pre-tokenization model (from tokenizer.ggml.model) */
#define CMOL_TOK_LLAMA  0   /* SentencePiece BPE ("llama") — default       */
#define CMOL_TOK_GPT2   1   /* Byte-level GPT-2 BPE ("gpt2", "qwen2", …)  */

typedef struct {
    /* ── raw data from GGUF parse (set by gguf.c) ──────────────────── */
    const char       **vocab;        /* [vocab_size] BPE strings in arena  */
    float             *scores;       /* [vocab_size] BPE merge scores      */
    uint8_t           *token_type;   /* [vocab_size] cmol_token_type_t     */
    int                vocab_size;
    int                tok_model;    /* CMOL_TOK_LLAMA or CMOL_TOK_GPT2   */

    /* BPE merge rules (index = priority; 0 = highest).
     * merge_left[i] + merge_right[i] → some result token.
     * Stored in arena; n_merges ≈ vocab_size - 256. */
    int32_t           *merge_left;
    int32_t           *merge_right;
    int                n_merges;

    int32_t bos_id;
    int32_t eos_id;
    int32_t unk_id;
    int     add_bos;   /* 1 = prepend BOS on encode (default), 0 = don't */

    /* ── built by cmol_tokenizer_build() (set by tokenizer.c) ──────── */
    const char       **decoded_vocab;   /* [vocab_size] ▁→space, <0xNN>→byte */
    int32_t           *vocab_sort_idx;  /* [vocab_size] sorted for str→ID lookup */
    int32_t           *merge_result;    /* [n_merges]   result token for merge i */
    int32_t           *msort_idx;       /* [n_merges]   sorted for pair→rank lookup */
} cmol_tokenizer_t;

/* =========================================================================
 * KV cache  (per session, carved from arena)
 *
 * Layout (both k and v are contiguous flat arrays):
 *   k[layer][pos][kv_head][dim]  →  index = ((layer * max_tokens + pos)
 *                                           * n_kv_heads + kv_head)
 *                                           * d_head
 * ====================================================================== */

typedef struct {
    float *k;           /* [n_layers * max_tokens * n_kv_heads * d_head]  */
    float *v;           /* same shape                                      */
    int    n_tokens;    /* current fill position (next write goes here)    */
    int    max_tokens;  /* = cfg.max_ctx                                   */
} cmol_kvcache_t;

/* =========================================================================
 * Arena allocator
 *
 * Single bump-pointer allocator.  The backing buffer is the one block
 * malloc'd in cmol_load(); everything else uses cmol_arena_alloc().
 * ====================================================================== */

typedef struct {
    uint8_t *base;
    size_t   size;
    size_t   used;
} cmol_arena_t;

/* =========================================================================
 * SIMD kernel dispatch table
 *
 * Set once at cmol_load() time by cmol_kernels_select() in quant.c.
 * Only the quantised matmul inner loop varies by SIMD level; all other
 * ops (RMSNorm, RoPE, SwiGLU, softmax) are cheap enough in scalar.
 *
 * matmul semantics:
 *   out[m][n] = A[m][k] · B[k][n]
 *   A is a quantised weight matrix (dtype); B is a float32 activation.
 *   Caller ensures n % 8 == 0 (SIMD alignment requirement).
 * ====================================================================== */

typedef void (*cmol_matmul_fn_t)(
        float        *out,     /* output [m * n], float32                */
        const void   *a,       /* weight [m * k], quantised              */
        const float  *b,       /* input  [k * n], float32                */
        int           m,       /* rows of A / output                     */
        int           k,       /* inner dimension                        */
        int           n,       /* columns of B / output                  */
        cmol_dtype_t  dtype);

typedef struct {
    cmol_matmul_fn_t matmul;
    const char      *name;    /* "avx512" | "avx2" | "neon" | "scalar"  */
} cmol_kernels_t;

/* =========================================================================
 * Full model handle  (opaque struct cmol_model in public API)
 * ====================================================================== */

/* Maximum concurrent sessions; pool bitmask fits in uint32_t. */
#define CMOL_MAX_SESSIONS 32

struct cmol_model {
    cmol_hparams_t    hparams;
    cmol_mmap_t       mmap;          /* memory-mapped GGUF file           */
    cmol_tensor_t    *tensors;       /* array[n_tensors] in arena         */
    int               n_tensors;
    cmol_tokenizer_t  tokenizer;
    cmol_kernels_t    kernels;       /* SIMD dispatch, set at load time   */
    cmol_config_t     cfg;           /* copy of the config passed in      */

    /* Arena — the single malloc backing everything below */
    uint8_t          *arena_buf;     /* raw buffer (the malloc'd block)   */
    cmol_arena_t      arena;

    /* Session pool */
    struct cmol_session *session_slots; /* array[cfg.max_sessions]        */
    pthread_mutex_t      pool_lock;
    uint32_t             pool_free;  /* bitmask: bit i set = slot i free  */
};

/* =========================================================================
 * Session handle  (opaque struct cmol_session in public API)
 * ====================================================================== */

struct cmol_session {
    cmol_model_t   *model;
    cmol_kvcache_t  kvcache;
    float          *scratch;       /* activation scratch for one chunk    */
    size_t          scratch_size;  /* bytes allocated                     */
    int32_t        *token_buf;     /* temp token array for cmol_generate  */
    int             token_buf_cap; /* capacity in tokens (= cfg.max_ctx)  */
    int             slot;          /* index in model->session_slots       */
};

/* =========================================================================
 * Internal logging
 *
 * Calls model->cfg.log_fn if set; no-op otherwise.
 * ====================================================================== */

void cmol_log(const cmol_model_t *m, int level, const char *fmt, ...);

#define CMOL_LOGE(m, ...) cmol_log((m), CMOL_LOG_ERROR, __VA_ARGS__)
#define CMOL_LOGW(m, ...) cmol_log((m), CMOL_LOG_WARN,  __VA_ARGS__)
#define CMOL_LOGI(m, ...) cmol_log((m), CMOL_LOG_INFO,  __VA_ARGS__)
#define CMOL_LOGD(m, ...) cmol_log((m), CMOL_LOG_DEBUG, __VA_ARGS__)

/* ===== src/arena.h ===== */
/*
 * arena.h — bump-pointer arena allocator
 * Internal header — not part of the public API.
 */

void cmol_arena_init(cmol_arena_t *a, void *buf, size_t size);

/*
 * Allocate `size` bytes aligned to `align` from the arena.
 * `align` must be a power of two.
 * Returns NULL if the arena is exhausted.
 */
void *cmol_arena_alloc(cmol_arena_t *a, size_t size, size_t align);

/* Helper: allocate an array of `count` elements of `elem_size` bytes. */
void *cmol_arena_alloc_n(cmol_arena_t *a, size_t count, size_t elem_size);

/* Reset the arena, making all memory available again (does not zero). */
void cmol_arena_reset(cmol_arena_t *a);

/* Return the number of bytes still available. */
size_t cmol_arena_remaining(const cmol_arena_t *a);

/* ===== src/gguf.h ===== */
/*
 * gguf.h — GGUF file parser
 * Implemented in Phase 1.
 * Internal header — not part of the public API.
 */

/*
 * cmol_gguf_parse — parse an mmap'd GGUF file.
 *
 * Populates hparams, tokenizer, and the tensor array.
 * All variable-length data (strings, vocab, merge rules) is written into
 * `arena`; the tensor data itself stays in the mmap region (no copies).
 *
 * Returns CMOL_OK on success.
 */
cmol_err_t cmol_gguf_parse(const cmol_mmap_t *mmap,
                             cmol_arena_t      *arena,
                             cmol_hparams_t    *hparams,
                             cmol_tensor_t    **tensors_out,
                             int               *n_tensors_out,
                             cmol_tokenizer_t  *tokenizer);

/*
 * cmol_gguf_find_tensor — look up a tensor by name.
 * Returns NULL if not found.
 */
cmol_tensor_t *cmol_gguf_find_tensor(cmol_tensor_t *tensors, int n,
                                      const char *name);

/*
 * cmol_gguf_peek — read only the GGUF header and architecture metadata
 * from `path`, without a full parse or mmap.
 *
 * Used by cmol_arena_estimate() to compute memory requirements before
 * committing to a full load.  Returns CMOL_OK on success.
 */
cmol_err_t cmol_gguf_peek(const char     *path,
                           cmol_hparams_t *hparams_out,
                           size_t         *n_tensors_out);

/* ===== src/tokenizer.h ===== */
/*
 * tokenizer.h — BPE tokenizer (SentencePiece variant)
 *
 * Internal header — not part of the public API.
 * Consumers: tokenizer.c (implementation), api.c, tests/test_tokenizer.c.
 *
 * Life-cycle:
 *   1. cmol_gguf_parse()        — fills raw fields (vocab, merge_left/right)
 *   2. cmol_tokenizer_build()   — builds runtime lookup tables in arena
 *   3. cmol_tokenizer_encode()  — BPE encode UTF-8 → token IDs
 *   4. cmol_tokenizer_decode_token() — token ID → UTF-8 piece
 */

/* cmol_arena_t is defined in cmol_internal.h */

/* =========================================================================
 * cmol_tokenizer_build
 *
 * Builds the four runtime lookup tables that encode/decode require:
 *
 *   decoded_vocab[vocab_size]   — decoded form of each token:
 *                                   ▁ prefix  → leading space
 *                                   <0xNN>    → single byte value
 *                                   other     → original vocab string
 *
 *   vocab_sort_idx[vocab_size]  — indices into vocab[], sorted by string;
 *                                 enables O(log V) string→token-ID lookup.
 *
 *   merge_result[n_merges]      — result token ID for merge rule i:
 *                                 vocab[merge_left[i]] + vocab[merge_right[i]]
 *                                 looked up via vocab_sort_idx.
 *
 *   msort_idx[n_merges]         — indices into merge_left/right[], sorted by
 *                                 pack64(left, right); enables O(log M)
 *                                 pair→rank (= merge priority) lookup.
 *
 * All allocations come from `arena`.  Must be called once before encode.
 * Returns CMOL_OK on success, CMOL_ERR_OOM / CMOL_ERR_INVALID on failure.
 * ====================================================================== */
cmol_err_t cmol_tokenizer_build(cmol_tokenizer_t *tok, cmol_arena_t *arena);

/* =========================================================================
 * cmol_tokenizer_encode
 *
 * BPE-encodes null-terminated UTF-8 `text` to token IDs.
 *
 * Pre-tokenization (CMOL_TOK_LLAMA / SentencePiece):
 *   - Inject one ▁ dummy prefix before the first character.
 *   - Replace each ' ' (space) with ▁.
 *   - Look up each UTF-8 character in the vocab; on miss emit byte-token
 *     fallback (<0xNN>) for each byte of the character.
 *
 * `add_bos`: prepend tok->bos_id as the first output token when non-zero.
 *
 * Returns the number of tokens written (≥ 0), or a negative cmol_err_t:
 *   CMOL_ERR_ARGS        — NULL pointer or out_cap ≤ 0
 *   CMOL_ERR_UNSUPPORTED — cmol_tokenizer_build() not yet called
 *   CMOL_ERR_TRUNC       — out_cap too small (partial result written)
 * ====================================================================== */
int cmol_tokenizer_encode(const cmol_tokenizer_t *tok,
                           const char             *text,
                           int32_t                *out,
                           int                     out_cap,
                           int                     add_bos);

/* =========================================================================
 * cmol_tokenizer_decode_token
 *
 * Returns the UTF-8 string for a single token ID.
 *
 * The returned pointer is into the arena (or into the original vocab string
 * literal for unchanged tokens) and is valid for the lifetime of the model.
 * Do not free.  Returns NULL for out-of-range IDs.
 *
 * If cmol_tokenizer_build() was not yet called, falls back to the raw
 * vocab string (▁ not converted, byte tokens not expanded).
 * ====================================================================== */
const char *cmol_tokenizer_decode_token(const cmol_tokenizer_t *tok,
                                         int32_t                 token_id);

/* ===== src/quant.h ===== */
/*
 * quant.h — quantisation kernels and SIMD dispatch
 * Implemented in Phase 4.
 * Internal header — not part of the public API.
 */

/*
 * cmol_kernels_select — detect CPU capabilities and return the fastest
 * available kernel set.  Called once in cmol_load().
 */
cmol_kernels_t cmol_kernels_select(void);

/*
 * cmol_dequant_row — dequantise `n` values from a quantised row into
 * float32.  `n` must be a multiple of the block size for `dtype`.
 *
 * Q8_0  block size: 32  values
 * Q4_K  block size: 256 values
 */
void cmol_dequant_row(const void *src, float *dst, int n, cmol_dtype_t dtype);

/* ---- Kernel implementations (one per SIMD level) ---------------------- */

void cmol_matmul_scalar(float *out, const void *a, const float *b,
                         int m, int k, int n, cmol_dtype_t dtype);

#if defined(__x86_64__) || defined(_M_X64)
void cmol_matmul_avx2  (float *out, const void *a, const float *b,
                         int m, int k, int n, cmol_dtype_t dtype);
void cmol_matmul_avx512(float *out, const void *a, const float *b,
                         int m, int k, int n, cmol_dtype_t dtype);
#endif

#if defined(__ARM_NEON) || defined(__aarch64__)
void cmol_matmul_neon  (float *out, const void *a, const float *b,
                         int m, int k, int n, cmol_dtype_t dtype);
#endif

/* ===== src/attn.h ===== */
/*
 * attn.h — grouped-query attention with KV cache
 * Internal header — not part of the public API.
 *
 * Callers (model.c) are responsible for allocating the scratch buffers
 * and arranging the scratch layout described in model.c.
 */

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

/* ===== src/model.h ===== */
/*
 * model.h — transformer forward pass
 * Implemented in Phase 5.
 * Internal header — not part of the public API.
 */

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

/* ===== src/sampler.h ===== */
/*
 * sampler.h — token sampling (greedy / temperature / top-k / top-p)
 * Implemented in Phase 6.
 * Internal header — not part of the public API.
 */

/*
 * cmol_sample — sample a token ID from a logit vector.
 *
 * `logits` is modified in-place (temperature scaling, softmax).
 * `rng_state` is updated in-place; seed it once with cmol_rng_seed().
 *
 * With temperature == 0.0 the result is always argmax (greedy).
 */
int32_t cmol_sample(float                   *logits,
                     int                      vocab_size,
                     const cmol_gen_params_t *params,
                     uint64_t                *rng_state);

/* ---- xoshiro256** RNG ------------------------------------------------- */

/*
 * cmol_rng_seed — initialise the 256-bit state from a 32-bit seed.
 * seed == 0 uses a time-based seed for non-deterministic output.
 */
void cmol_rng_seed(uint64_t state[4], unsigned int seed);

/* cmol_rng_next — return the next 64-bit value and advance the state. */
uint64_t cmol_rng_next(uint64_t state[4]);

/*
 * cmol_apply_repeat_penalty — discount logits of recently-seen tokens.
 *
 * Applies the standard llama.cpp repetition penalty formula in-place,
 * before temperature scaling and softmax:
 *   logit > 0  →  logit /= penalty
 *   logit ≤ 0  →  logit *= penalty
 *
 * `tokens`   — ring buffer of the last N generated/prompt tokens (IDs).
 *              Any entry with id < 0 or id >= vocab_size is skipped.
 * `n_tokens` — number of valid entries in `tokens` (≤ CMOL_REPEAT_BUF).
 * `penalty`  — multiplier; values ≤ 1.0f are a no-op.
 */
#define CMOL_REPEAT_BUF 128

void cmol_apply_repeat_penalty(float *logits, int vocab_size,
                                const int32_t *tokens, int n_tokens,
                                float penalty);

/* ===== src/platform.c ===== */
/*
 * platform.c — OS abstraction implementation
 * Included by src/cmol.c (unity build); do not compile standalone.
 */

/* =========================================================================
 * Memory-mapped file
 * ====================================================================== */

#ifdef _WIN32
/* ---- Windows (stub — implement when Windows target is available) ------- */
#include <windows.h>

cmol_err_t cmol_mmap_open(const char *path, cmol_mmap_t *out) {
    (void)path; (void)out;
    /* TODO: CreateFile → CreateFileMapping → MapViewOfFile */
    return CMOL_ERR_UNSUPPORTED;
}

void cmol_mmap_close(cmol_mmap_t *m) {
    if (!m || !m->data) return;
    /* TODO: UnmapViewOfFile(m->data); CloseHandle(...) */
    m->data = NULL;
    m->size = 0;
}

#else
/* ---- POSIX --------------------------------------------------------------- */

/* MADV_SEQUENTIAL is a Linux/BSD extension not visible under strict C99.
 * _DEFAULT_SOURCE (glibc >= 2.19) or _BSD_SOURCE re-exposes it without
 * pulling in the full GNU namespace.  Define before any system header. */
#ifndef _DEFAULT_SOURCE
#  define _DEFAULT_SOURCE
#endif
#ifndef _BSD_SOURCE          /* older glibc (< 2.19, e.g. Raspbian Wheezy) */
#  define _BSD_SOURCE
#endif

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

cmol_err_t cmol_mmap_open(const char *path, cmol_mmap_t *out) {
    if (!path || !out) return CMOL_ERR_ARGS;

    int fd = open(path, O_RDONLY);
    if (fd < 0) return CMOL_ERR_IO;

    struct stat st;
    if (fstat(fd, &st) < 0) { close(fd); return CMOL_ERR_IO; }

    void *data = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd); /* fd can be closed immediately; mapping keeps the data alive */
    if (data == MAP_FAILED) return CMOL_ERR_IO;

    /* Hint to the OS: we'll read sequentially during the parse pass.
     * Guarded: MADV_SEQUENTIAL may be absent on non-Linux POSIX targets
     * even with _DEFAULT_SOURCE (e.g. older musl, OpenBSD). */
#ifdef MADV_SEQUENTIAL
    madvise(data, (size_t)st.st_size, MADV_SEQUENTIAL);
#endif

    out->data = data;
    out->size = (size_t)st.st_size;
    return CMOL_OK;
}

void cmol_mmap_close(cmol_mmap_t *m) {
    if (!m || !m->data) return;
    munmap(m->data, m->size);
    m->data = NULL;
    m->size = 0;
}

#endif /* _WIN32 */

/* =========================================================================
 * CPU feature detection
 * ====================================================================== */

/* ---- x86 / x86-64 ------------------------------------------------------- */
#if defined(__x86_64__) || defined(_M_X64)

static void cmol__cpuid(unsigned int leaf, unsigned int subleaf,
                         unsigned int *eax, unsigned int *ebx,
                         unsigned int *ecx, unsigned int *edx) {
#if defined(_MSC_VER)
    int regs[4];
    __cpuidex(regs, (int)leaf, (int)subleaf);
    *eax = (unsigned int)regs[0]; *ebx = (unsigned int)regs[1];
    *ecx = (unsigned int)regs[2]; *edx = (unsigned int)regs[3];
#else
    __asm__ volatile(
        "cpuid"
        : "=a"(*eax), "=b"(*ebx), "=c"(*ecx), "=d"(*edx)
        : "a"(leaf), "c"(subleaf)
    );
#endif
}

cmol_cpu_t cmol_detect_cpu(void) {
    cmol_cpu_t cpu = {0};
    unsigned int eax, ebx, ecx, edx;

    /* Leaf 0: highest supported leaf */
    cmol__cpuid(0, 0, &eax, &ebx, &ecx, &edx);
    if (eax < 7) return cpu; /* no extended feature flags */

    /* Leaf 7, subleaf 0: extended features */
    cmol__cpuid(7, 0, &eax, &ebx, &ecx, &edx);
    cpu.avx2    = (ebx >> 5)  & 1; /* EBX bit  5: AVX2        */
    cpu.avx512f = (ebx >> 16) & 1; /* EBX bit 16: AVX-512F    */

    return cpu;
}

/* ---- AArch64 (Apple Silicon, RPi 3/4 64-bit) ----------------------------- */
#elif defined(__aarch64__)

cmol_cpu_t cmol_detect_cpu(void) {
    /* ASIMD (NEON equivalent) is architecturally mandatory on AArch64. */
    cmol_cpu_t cpu = {0};
    cpu.neon = 1;
    return cpu;
}

/* ---- ARMv7 (NEON optional — check at runtime via AT_HWCAP) --------------- */
#elif defined(__arm__) || defined(__ARM_ARCH)

#if defined(__linux__) && __has_include(<sys/auxv.h>)
#  include <sys/auxv.h>
#  ifndef HWCAP_NEON
#    define HWCAP_NEON (1 << 12)
#  endif
#endif

cmol_cpu_t cmol_detect_cpu(void) {
    cmol_cpu_t cpu = {0};
#if defined(__ARM_NEON)
    /* Compiled with -mfpu=neon: safe to assume NEON is present. */
    cpu.neon = 1;
#elif defined(__linux__) && __has_include(<sys/auxv.h>)
    unsigned long hwcap = getauxval(AT_HWCAP);
    cpu.neon = (hwcap & HWCAP_NEON) != 0;
#endif
    /* ARMv6 (RPi Zero W): __ARM_NEON not defined, getauxval returns 0
     * for HWCAP_NEON → cpu.neon stays 0 → scalar fallback selected. */
    return cpu;
}

/* ---- Everything else (scalar fallback) ----------------------------------- */
#else

cmol_cpu_t cmol_detect_cpu(void) {
    cmol_cpu_t cpu = {0};
    return cpu;
}

#endif

/* ===== src/arena.c ===== */
/*
 * arena.c — bump-pointer arena allocator implementation
 * Included by src/cmol.c (unity build); do not compile standalone.
 */

#include <string.h>

void cmol_arena_init(cmol_arena_t *a, void *buf, size_t size) {
    a->base = (uint8_t *)buf;
    a->size = size;
    a->used = 0;
}

void *cmol_arena_alloc(cmol_arena_t *a, size_t size, size_t align) {
    /* align must be a power of two */
    size_t start = (a->used + align - 1u) & ~(align - 1u);
    if (start + size > a->size) return NULL;
    void *ptr = a->base + start;
    a->used = start + size;
    memset(ptr, 0, size); /* zero on allocation — predictable behaviour */
    return ptr;
}

void *cmol_arena_alloc_n(cmol_arena_t *a, size_t count, size_t elem_size) {
    /* Check for overflow before multiplying */
    if (count && elem_size > (size_t)-1 / count) return NULL;
    return cmol_arena_alloc(a, count * elem_size, elem_size < 8 ? elem_size : 8);
}

void cmol_arena_reset(cmol_arena_t *a) {
    a->used = 0;
}

size_t cmol_arena_remaining(const cmol_arena_t *a) {
    return a->size - a->used;
}

/* ===== src/gguf.c ===== */
/*
 * gguf.c — GGUF v2/v3 file parser
 * Included by src/cmol.c (unity build); do not compile standalone.
 *
 * Spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
 *
 * Architecture-prefixed keys (e.g. "llama.context_length") are matched
 * dynamically using the value of "general.architecture", so the same
 * parser handles llama, smollm3, and any future architectures that
 * follow the same structural pattern.
 */


#include <string.h>
#include <stdio.h>
#include <stdlib.h>   /* qsort */

/* =========================================================================
 * Constants
 * ====================================================================== */

#define GGUF_MAGIC  0x46554747u  /* "GGUF" as little-endian uint32        */
#define GGUF_ALIGN  32u          /* default alignment of tensor data       */

/* GGUF metadata value types */
typedef enum {
    GV_UINT8   =  0,
    GV_INT8    =  1,
    GV_UINT16  =  2,
    GV_INT16   =  3,
    GV_UINT32  =  4,
    GV_INT32   =  5,
    GV_FLOAT32 =  6,
    GV_BOOL    =  7,
    GV_STRING  =  8,
    GV_ARRAY   =  9,
    GV_UINT64  = 10,
    GV_INT64   = 11,
    GV_FLOAT64 = 12,
} gv_type_t;

/* Byte widths for scalar GV types (0 = variable-length) */
static const uint8_t GV_WIDTH[] = {1,1,2,2,4,4,4,1,0,0,8,8,8};

/* =========================================================================
 * Metadata keys
 *
 * Architecture-specific keys use a dynamic prefix derived from
 * "general.architecture" (e.g. "llama", "smollm3").
 * Match them with kmatch(key, arch, SUFFIX_*).
 *
 * Non-prefixed keys (general.*, tokenizer.*) use strcmp directly.
 * ====================================================================== */

/* Suffixes for architecture-prefixed keys */
#define SUFFIX_CTX          "context_length"
#define SUFFIX_EMB          "embedding_length"
#define SUFFIX_LAYERS       "block_count"
#define SUFFIX_FFN          "feed_forward_length"
#define SUFFIX_HEADS        "attention.head_count"
#define SUFFIX_KV_HEADS     "attention.head_count_kv"
#define SUFFIX_ROPE_BASE    "rope.freq_base"
#define SUFFIX_RMS_EPS      "attention.layer_norm_rms_epsilon"
#define SUFFIX_VOCAB_SIZE   "vocab_size"
#define SUFFIX_NO_ROPE      "attention.no_rope_layer_interval"
#define SUFFIX_ROPE_SCALING "rope.scaling.type"   /* YARN etc. — Phase 5 */

/* Non-prefixed keys */
#define K_ARCH        "general.architecture"
#define K_TOK_TOKENS  "tokenizer.ggml.tokens"
#define K_TOK_SCORES  "tokenizer.ggml.scores"
#define K_TOK_TYPES   "tokenizer.ggml.token_type"
#define K_TOK_MERGES  "tokenizer.ggml.merges"
#define K_TOK_MODEL   "tokenizer.ggml.model"
#define K_TOK_BOS     "tokenizer.ggml.bos_token_id"
#define K_TOK_EOS     "tokenizer.ggml.eos_token_id"
#define K_TOK_UNK     "tokenizer.ggml.unknown_token_id"
#define K_TOK_ADD_BOS "tokenizer.ggml.add_bos_token"

/* =========================================================================
 * Dynamic key matching
 *
 * kmatch(key, arch, suffix) returns 1 iff key == arch + "." + suffix.
 * Example: kmatch("smollm3.block_count", "smollm3", "block_count") → 1
 *          kmatch("llama.block_count",   "smollm3", "block_count") → 0
 * No allocation — pure character-by-character comparison.
 * ====================================================================== */

static int kmatch(const char *key, const char *arch, const char *suffix) {
    /* match arch prefix */
    while (*arch && *key == *arch) { key++; arch++; }
    if (*arch || *key != '.') return 0;
    key++; /* skip '.' */
    /* match suffix */
    while (*suffix && *key == *suffix) { key++; suffix++; }
    return !*suffix && !*key;
}

/* =========================================================================
 * Cursor — sequential reader over the mmap'd region
 * ====================================================================== */

typedef struct {
    const uint8_t *base;
    size_t         size;
    size_t         pos;
    int            err;   /* sticky: once set, reads return 0 / NULL      */
} gcur_t;

static void gcur_init(gcur_t *c, const void *data, size_t size) {
    c->base = (const uint8_t *)data;
    c->size = size;
    c->pos  = 0;
    c->err  = 0;
}

static int gcur_check(gcur_t *c, size_t n) {
    if (!c->err && c->pos + n <= c->size) return 1;
    c->err = 1;
    return 0;
}

#define GCUR_READ_FN(T, suffix)                           \
static T gcur_##suffix(gcur_t *c) {                       \
    T v = (T)0;                                           \
    if (!gcur_check(c, sizeof(T))) return v;              \
    memcpy(&v, c->base + c->pos, sizeof(T));              \
    c->pos += sizeof(T);                                  \
    return v;                                             \
}

/* Instantiate only the readers used in gguf.c.
 * Phase 4 (quant.c) will add the remaining widths as needed. */
GCUR_READ_FN(uint8_t,  u8)
GCUR_READ_FN(uint32_t, u32)
GCUR_READ_FN(int32_t,  i32)
GCUR_READ_FN(uint64_t, u64)
GCUR_READ_FN(float,    f32)

/* Read a GGUF string into the arena as a null-terminated C string. */
static const char *gcur_str(gcur_t *c, cmol_arena_t *arena) {
    uint64_t len = gcur_u64(c);
    if (c->err || len > (uint64_t)128 * 1024 * 1024) { c->err = 1; return NULL; }
    if (!gcur_check(c, (size_t)len)) return NULL;
    char *buf = (char *)cmol_arena_alloc(arena, (size_t)len + 1, 1);
    if (!buf) { c->err = 1; return NULL; }
    if (len) memcpy(buf, c->base + c->pos, (size_t)len);
    buf[len] = '\0';
    c->pos += (size_t)len;
    return buf;
}

/* Discard a GGUF string without storing it. */
static void gcur_str_skip(gcur_t *c) {
    uint64_t len = gcur_u64(c);
    if (c->err) return;
    if (!gcur_check(c, (size_t)len)) return;
    c->pos += (size_t)len;
}

/* =========================================================================
 * Value skipping
 * ====================================================================== */

static void skip_scalar(gcur_t *c, gv_type_t t) {
    if (t == GV_STRING) { gcur_str_skip(c); return; }
    uint8_t w = (t < 13) ? GV_WIDTH[t] : 0;
    if (w && gcur_check(c, w)) c->pos += w;
    else if (!w) c->err = 1;
}

static void skip_value(gcur_t *c, gv_type_t t) {
    if (t != GV_ARRAY) { skip_scalar(c, t); return; }
    gv_type_t et = (gv_type_t)gcur_u32(c);
    uint64_t  n  = gcur_u64(c);
    if (c->err) return;
    for (uint64_t i = 0; i < n && !c->err; i++)
        skip_scalar(c, et);
}

/* =========================================================================
 * Tokenizer array readers
 * ====================================================================== */

static cmol_err_t read_tok_vocab(gcur_t *c, cmol_arena_t *a,
                                  cmol_tokenizer_t *tok, uint64_t count) {
    tok->vocab      = (const char **)cmol_arena_alloc_n(a, (size_t)count, sizeof(char *));
    tok->vocab_size = (int)count;
    if (!tok->vocab) return CMOL_ERR_OOM;
    for (uint64_t i = 0; i < count && !c->err; i++) {
        tok->vocab[i] = gcur_str(c, a);
        if (!tok->vocab[i]) return c->err ? CMOL_ERR_INVALID : CMOL_ERR_OOM;
    }
    return c->err ? CMOL_ERR_INVALID : CMOL_OK;
}

static cmol_err_t read_tok_scores(gcur_t *c, cmol_arena_t *a,
                                   cmol_tokenizer_t *tok, uint64_t count) {
    tok->scores = (float *)cmol_arena_alloc_n(a, (size_t)count, sizeof(float));
    if (!tok->scores) return CMOL_ERR_OOM;
    for (uint64_t i = 0; i < count && !c->err; i++)
        tok->scores[i] = gcur_f32(c);
    return c->err ? CMOL_ERR_INVALID : CMOL_OK;
}

static cmol_err_t read_tok_types(gcur_t *c, cmol_arena_t *a,
                                  cmol_tokenizer_t *tok, uint64_t count) {
    tok->token_type = (uint8_t *)cmol_arena_alloc_n(a, (size_t)count, sizeof(uint8_t));
    if (!tok->token_type) return CMOL_ERR_OOM;
    for (uint64_t i = 0; i < count && !c->err; i++)
        tok->token_type[i] = (uint8_t)gcur_i32(c);
    return c->err ? CMOL_ERR_INVALID : CMOL_OK;
}

/* =========================================================================
 * Merge string storage (raw strings, resolved after KV loop)
 * ====================================================================== */

typedef struct {
    const char **strings;
    int          count;
} gguf_merges_t;

static cmol_err_t read_tok_merges_raw(gcur_t *c, cmol_arena_t *a,
                                       gguf_merges_t *m, uint64_t count) {
    m->strings = (const char **)cmol_arena_alloc_n(a, (size_t)count, sizeof(char *));
    if (!m->strings) return CMOL_ERR_OOM;
    m->count = (int)count;
    for (uint64_t i = 0; i < count && !c->err; i++) {
        m->strings[i] = gcur_str(c, a);
        if (!m->strings[i]) return c->err ? CMOL_ERR_INVALID : CMOL_ERR_OOM;
    }
    return c->err ? CMOL_ERR_INVALID : CMOL_OK;
}

/* =========================================================================
 * Merge resolution
 * ====================================================================== */

static const char **g_rsort_vocab;

static int rsort_cmp(const void *a, const void *b) {
    int32_t ia = *(const int32_t *)a, ib = *(const int32_t *)b;
    return strcmp(g_rsort_vocab[ia], g_rsort_vocab[ib]);
}

static int32_t vocab_lookup(const int32_t *idx, int n, const char *piece) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        int c   = strcmp(g_rsort_vocab[idx[mid]], piece);
        if (c == 0) return idx[mid];
        if (c  < 0) lo = mid + 1;
        else        hi = mid - 1;
    }
    return -1;
}

static cmol_err_t resolve_merges(cmol_arena_t     *a,
                                  cmol_tokenizer_t *tok,
                                  const gguf_merges_t *raw) {
    if (!raw->count || !tok->vocab || !tok->vocab_size) return CMOL_OK;

    int32_t *idx = (int32_t *)cmol_arena_alloc_n(a, (size_t)tok->vocab_size, sizeof(int32_t));
    if (!idx) return CMOL_ERR_OOM;
    for (int i = 0; i < tok->vocab_size; i++) idx[i] = (int32_t)i;
    g_rsort_vocab = tok->vocab;
    qsort(idx, (size_t)tok->vocab_size, sizeof(int32_t), rsort_cmp);

    tok->merge_left  = (int32_t *)cmol_arena_alloc_n(a, (size_t)raw->count, sizeof(int32_t));
    tok->merge_right = (int32_t *)cmol_arena_alloc_n(a, (size_t)raw->count, sizeof(int32_t));
    tok->n_merges    = raw->count;
    if (!tok->merge_left || !tok->merge_right) return CMOL_ERR_OOM;

    char tmp[512];
    for (int i = 0; i < raw->count; i++) {
        const char *s   = raw->strings[i];
        const char *sep = strchr(s, ' ');
        if (!sep) { tok->merge_left[i] = tok->merge_right[i] = -1; continue; }
        size_t llen = (size_t)(sep - s);
        if (llen >= sizeof tmp) { tok->merge_left[i] = tok->merge_right[i] = -1; continue; }
        memcpy(tmp, s, llen);
        tmp[llen] = '\0';
        tok->merge_left[i]  = vocab_lookup(idx, tok->vocab_size, tmp);
        tok->merge_right[i] = vocab_lookup(idx, tok->vocab_size, sep + 1);
    }
    return CMOL_OK;
}

/* =========================================================================
 * KV pair dispatcher
 *
 * `arch` is the architecture prefix string (e.g. "llama", "smollm3").
 * It is updated in-place when "general.architecture" is encountered,
 * so callers must initialise it to "llama" as a safe default.
 * ====================================================================== */

static cmol_err_t read_kv(gcur_t *c, cmol_arena_t *a, char *arch,
                           cmol_hparams_t *hp, cmol_tokenizer_t *tok,
                           gguf_merges_t *merges) {
    const char *key = gcur_str(c, a);
    if (!key) return c->err ? CMOL_ERR_INVALID : CMOL_ERR_OOM;

    gv_type_t vtype = (gv_type_t)gcur_u32(c);
    if (c->err) return CMOL_ERR_INVALID;

    /* ---- Architecture detection (general.architecture) --------------- */
    if (!strcmp(key, K_ARCH) && vtype == GV_STRING) {
        const char *s = gcur_str(c, a);
        if (s) {
            strncpy(arch, s, 31);
            arch[31] = '\0';
            strncpy(hp->arch, s, sizeof(hp->arch) - 1);
            hp->arch[sizeof(hp->arch) - 1] = '\0';
        }
        return c->err ? CMOL_ERR_INVALID : CMOL_OK;
    }

    /* ---- Architecture-prefixed scalar uint32 ------------------------- */
    if (vtype == GV_UINT32) {
        uint32_t v = gcur_u32(c);
        if (c->err) return CMOL_ERR_INVALID;
        if      (kmatch(key, arch, SUFFIX_CTX))      hp->model_max_ctx         = (int)v;
        else if (kmatch(key, arch, SUFFIX_EMB))       hp->d_model               = (int)v;
        else if (kmatch(key, arch, SUFFIX_LAYERS))    hp->n_layers              = (int)v;
        else if (kmatch(key, arch, SUFFIX_FFN))       hp->d_ffn                 = (int)v;
        else if (kmatch(key, arch, SUFFIX_HEADS))     hp->n_heads               = (int)v;
        else if (kmatch(key, arch, SUFFIX_KV_HEADS))  hp->n_kv_heads            = (int)v;
        else if (kmatch(key, arch, SUFFIX_VOCAB_SIZE))hp->vocab_size            = (int)v;
        else if (kmatch(key, arch, SUFFIX_NO_ROPE))   hp->no_rope_layer_interval= (int)v;
        else if (!strcmp(key, K_TOK_BOS))             tok->bos_id               = (int32_t)v;
        else if (!strcmp(key, K_TOK_EOS))             tok->eos_id               = (int32_t)v;
        else if (!strcmp(key, K_TOK_UNK))             tok->unk_id               = (int32_t)v;
        return CMOL_OK;
    }

    /* ---- Scalar bool (GV_BOOL = 1 byte) ------------------------------- */
    if (vtype == GV_BOOL) {
        uint8_t v = gcur_u8(c);
        if (c->err) return CMOL_ERR_INVALID;
        if (!strcmp(key, K_TOK_ADD_BOS)) tok->add_bos = v ? 1 : 0;
        return CMOL_OK;
    }

    /* ---- Architecture-prefixed scalar float32 ------------------------ */
    if (vtype == GV_FLOAT32) {
        float v = gcur_f32(c);
        if (c->err) return CMOL_ERR_INVALID;
        if      (kmatch(key, arch, SUFFIX_ROPE_BASE)) hp->rope_freq_base = v;
        else if (kmatch(key, arch, SUFFIX_RMS_EPS))   hp->rms_norm_eps   = v;
        return CMOL_OK;
    }

    /* ---- Arrays ------------------------------------------------------ */
    if (vtype == GV_ARRAY) {
        gv_type_t et    = (gv_type_t)gcur_u32(c);
        uint64_t  count = gcur_u64(c);
        if (c->err) return CMOL_ERR_INVALID;

        if (!strcmp(key, K_TOK_TOKENS) && et == GV_STRING)
            return read_tok_vocab(c, a, tok, count);
        if (!strcmp(key, K_TOK_SCORES) && et == GV_FLOAT32)
            return read_tok_scores(c, a, tok, count);
        if (!strcmp(key, K_TOK_TYPES) && et == GV_INT32)
            return read_tok_types(c, a, tok, count);
        if (!strcmp(key, K_TOK_MERGES) && et == GV_STRING)
            return read_tok_merges_raw(c, a, merges, count);

        /* Skip unknown arrays */
        for (uint64_t i = 0; i < count && !c->err; i++)
            skip_scalar(c, et);
        return c->err ? CMOL_ERR_INVALID : CMOL_OK;
    }

    /* ---- Tokenizer model name (controls pre-tokenization style) ------- */
    if (vtype == GV_STRING && !strcmp(key, K_TOK_MODEL)) {
        const char *s = gcur_str(c, a);
        if (s) tok->tok_model = (strcmp(s, "llama") == 0) ? CMOL_TOK_LLAMA
                                                           : CMOL_TOK_GPT2;
        return c->err ? CMOL_ERR_INVALID : CMOL_OK;
    }

    /* ---- YARN / rope scaling (Phase 5) — log intent, skip for now --- */
    if (vtype == GV_STRING && kmatch(key, arch, SUFFIX_ROPE_SCALING)) {
        /* Read the type string so we can record it, but don't act on it yet */
        gcur_str_skip(c);
        hp->yarn_factor_x100 = -1;  /* -1 = "present but unimplemented"  */
        return c->err ? CMOL_ERR_INVALID : CMOL_OK;
    }

    /* ---- Everything else (skip) -------------------------------------- */
    skip_value(c, vtype);
    return c->err ? CMOL_ERR_INVALID : CMOL_OK;
}

/* =========================================================================
 * Tensor byte-size computation
 * ====================================================================== */

static size_t tensor_nbytes(cmol_dtype_t dtype, int64_t n_elems) {
    switch (dtype) {
        case CMOL_DTYPE_F32:  return (size_t)n_elems * 4;
        case CMOL_DTYPE_F16:  return (size_t)n_elems * 2;
        case CMOL_DTYPE_Q5_0: return (size_t)(n_elems / 32)  * 22;
        case CMOL_DTYPE_Q8_0: return (size_t)(n_elems / 32)  * 34;
        case CMOL_DTYPE_Q4_K: return (size_t)(n_elems / 256) * 144;
        case CMOL_DTYPE_Q6_K: return (size_t)(n_elems / 256) * 210;
        default:               return 0;
    }
}

/* =========================================================================
 * Tensor info reader
 * ====================================================================== */

static cmol_err_t read_tensor_info(gcur_t *c, cmol_arena_t *a, cmol_tensor_t *t) {
    const char *name = gcur_str(c, a);
    if (!name) return c->err ? CMOL_ERR_INVALID : CMOL_ERR_OOM;
    size_t nlen = strlen(name);
    if (nlen >= CMOL_MAX_TENSOR_NAME) nlen = CMOL_MAX_TENSOR_NAME - 1;
    memcpy(t->name, name, nlen);
    t->name[nlen] = '\0';

    uint32_t ndims = gcur_u32(c);
    if (c->err || ndims < 1 || ndims > CMOL_MAX_DIMS) return CMOL_ERR_INVALID;
    t->n_dims = (int)ndims;

    int64_t n_elems = 1;
    for (int i = 0; i < t->n_dims; i++) {
        t->shape[i] = (int64_t)gcur_u64(c);
        n_elems    *= t->shape[i];
    }
    for (int i = t->n_dims; i < CMOL_MAX_DIMS; i++) t->shape[i] = 1;

    t->dtype       = (cmol_dtype_t)gcur_u32(c);
    t->file_offset = gcur_u64(c);
    t->n_bytes     = tensor_nbytes(t->dtype, n_elems);
    t->data        = NULL;

    return c->err ? CMOL_ERR_INVALID : CMOL_OK;
}

/* =========================================================================
 * cmol_gguf_parse
 * ====================================================================== */

cmol_err_t cmol_gguf_parse(const cmol_mmap_t *mmap,
                             cmol_arena_t      *arena,
                             cmol_hparams_t    *hp,
                             cmol_tensor_t    **tensors_out,
                             int               *n_tensors_out,
                             cmol_tokenizer_t  *tok) {
    gcur_t c;
    gcur_init(&c, mmap->data, mmap->size);

    uint32_t magic   = gcur_u32(&c);
    uint32_t version = gcur_u32(&c);
    if (c.err || magic != GGUF_MAGIC)         return CMOL_ERR_INVALID;
    if (version < 2 || version > 3)           return CMOL_ERR_UNSUPPORTED;

    uint64_t n_tensors = gcur_u64(&c);
    uint64_t n_kv      = gcur_u64(&c);
    if (c.err) return CMOL_ERR_INVALID;
    if (n_tensors > 65536 || n_kv > 65536)    return CMOL_ERR_INVALID;

    memset(hp,  0, sizeof *hp);
    memset(tok, 0, sizeof *tok);
    hp->rope_freq_base = 10000.0f;
    hp->rms_norm_eps   = 1e-5f;
    tok->bos_id = tok->eos_id = tok->unk_id = -1;
    tok->add_bos = 1;  /* conservative default; overridden by tokenizer.ggml.add_bos_token */

    /* Architecture prefix — default "llama", overwritten by general.architecture */
    char arch[32] = "llama";
    strncpy(hp->arch, arch, sizeof(hp->arch) - 1);

    gguf_merges_t raw_merges = {NULL, 0};
    for (uint64_t i = 0; i < n_kv; i++) {
        cmol_err_t e = read_kv(&c, arena, arch, hp, tok, &raw_merges);
        if (e != CMOL_OK) return e;
    }
    if (c.err) return CMOL_ERR_INVALID;

    if (hp->n_kv_heads == 0)  hp->n_kv_heads = hp->n_heads;
    if (hp->n_heads > 0)      hp->d_head     = hp->d_model / hp->n_heads;
    if (hp->vocab_size == 0)  hp->vocab_size  = tok->vocab_size;
    if (tok->vocab_size == 0) tok->vocab_size  = hp->vocab_size;

    if (raw_merges.count > 0) {
        cmol_err_t e = resolve_merges(arena, tok, &raw_merges);
        if (e != CMOL_OK) return e;
    }

    cmol_tensor_t *tensors = (cmol_tensor_t *)cmol_arena_alloc_n(
            arena, (size_t)n_tensors, sizeof(cmol_tensor_t));
    if (!tensors) return CMOL_ERR_OOM;

    for (uint64_t i = 0; i < n_tensors; i++) {
        cmol_err_t e = read_tensor_info(&c, arena, &tensors[i]);
        if (e != CMOL_OK) return e;
    }
    if (c.err) return CMOL_ERR_INVALID;

    size_t data_off = (c.pos + GGUF_ALIGN - 1u) & ~(size_t)(GGUF_ALIGN - 1u);
    for (uint64_t i = 0; i < n_tensors; i++) {
        size_t off = data_off + (size_t)tensors[i].file_offset;
        if (off + tensors[i].n_bytes > mmap->size) return CMOL_ERR_INVALID;
        tensors[i].data = (uint8_t *)mmap->data + off;
    }

    hp->tie_embeddings = (cmol_gguf_find_tensor(tensors, (int)n_tensors,
                                                 "output.weight") == NULL);

    *tensors_out   = tensors;
    *n_tensors_out = (int)n_tensors;
    return CMOL_OK;
}

/* =========================================================================
 * cmol_gguf_find_tensor
 * ====================================================================== */

cmol_tensor_t *cmol_gguf_find_tensor(cmol_tensor_t *tensors, int n,
                                      const char *name) {
    for (int i = 0; i < n; i++)
        if (strcmp(tensors[i].name, name) == 0) return &tensors[i];
    return NULL;
}

/* =========================================================================
 * cmol_gguf_peek
 * ====================================================================== */

cmol_err_t cmol_gguf_peek(const char     *path,
                           cmol_hparams_t *hp_out,
                           size_t         *n_tensors_out) {
    if (!path || !hp_out || !n_tensors_out) return CMOL_ERR_ARGS;

    cmol_mmap_t mmap;
    cmol_err_t  err = cmol_mmap_open(path, &mmap);
    if (err != CMOL_OK) return err;

    gcur_t c;
    gcur_init(&c, mmap.data, mmap.size);

    err = CMOL_ERR_INVALID;

    uint32_t magic   = gcur_u32(&c);
    uint32_t version = gcur_u32(&c);
    if (c.err || magic != GGUF_MAGIC || version < 2 || version > 3) goto done;

    uint64_t n_tensors = gcur_u64(&c);
    uint64_t n_kv      = gcur_u64(&c);
    if (c.err) goto done;

    cmol_hparams_t hp;
    memset(&hp, 0, sizeof hp);
    hp.rope_freq_base = 10000.0f;
    hp.rms_norm_eps   = 1e-5f;

    /* Architecture prefix — updated when we see general.architecture */
    char arch[32] = "llama";
    strncpy(hp.arch, arch, sizeof(hp.arch) - 1);

    for (uint64_t i = 0; i < n_kv && !c.err; i++) {
        /* Read key manually (no arena needed for peek) */
        uint64_t klen = gcur_u64(&c);
        if (c.err) break;
        if (!gcur_check(&c, (size_t)klen)) break;

        char key[128] = {0};
        int  have_key = (klen < sizeof key);
        if (have_key) {
            memcpy(key, c.base + c.pos, (size_t)klen);
            key[klen] = '\0';
        }
        c.pos += (size_t)klen;

        gv_type_t vtype = (gv_type_t)gcur_u32(&c);
        if (c.err) break;

        /* Architecture detection — must precede arch-prefixed keys */
        if (have_key && !strcmp(key, K_ARCH) && vtype == GV_STRING) {
            uint64_t alen = gcur_u64(&c);
            if (!c.err && alen < sizeof arch && gcur_check(&c, (size_t)alen)) {
                memcpy(arch, c.base + c.pos, (size_t)alen);
                arch[alen] = '\0';
                strncpy(hp.arch, arch, sizeof(hp.arch) - 1);
                c.pos += (size_t)alen;
            } else { skip_value(&c, GV_STRING); }
            continue;
        }

        /* Extract scalar architecture values using dynamic prefix */
        if (have_key && vtype == GV_UINT32) {
            uint32_t v = gcur_u32(&c);
            if      (kmatch(key, arch, SUFFIX_CTX))       hp.model_max_ctx          = (int)v;
            else if (kmatch(key, arch, SUFFIX_EMB))        hp.d_model                = (int)v;
            else if (kmatch(key, arch, SUFFIX_LAYERS))     hp.n_layers               = (int)v;
            else if (kmatch(key, arch, SUFFIX_FFN))        hp.d_ffn                  = (int)v;
            else if (kmatch(key, arch, SUFFIX_HEADS))      hp.n_heads                = (int)v;
            else if (kmatch(key, arch, SUFFIX_KV_HEADS))   hp.n_kv_heads             = (int)v;
            else if (kmatch(key, arch, SUFFIX_VOCAB_SIZE)) hp.vocab_size             = (int)v;
            else if (kmatch(key, arch, SUFFIX_NO_ROPE))    hp.no_rope_layer_interval = (int)v;
        } else if (have_key && vtype == GV_FLOAT32) {
            float v = gcur_f32(&c);
            if      (kmatch(key, arch, SUFFIX_ROPE_BASE))  hp.rope_freq_base = v;
            else if (kmatch(key, arch, SUFFIX_RMS_EPS))    hp.rms_norm_eps   = v;
        } else {
            skip_value(&c, vtype);
        }
    }

    if (!c.err) {
        if (hp.n_kv_heads == 0) hp.n_kv_heads = hp.n_heads;
        if (hp.n_heads > 0)     hp.d_head     = hp.d_model / hp.n_heads;
        *hp_out        = hp;
        *n_tensors_out = (size_t)n_tensors;
        err = CMOL_OK;
    }

done:
    cmol_mmap_close(&mmap);
    return err;
}

/* ===== src/tokenizer.c ===== */
/*
 * tokenizer.c — BPE tokenizer (SentencePiece variant for SmolLM2/3)
 * Included by src/cmol.c (unity build); do not compile standalone.
 *
 * Supports CMOL_TOK_LLAMA (SentencePiece-style BPE), used by SmolLM2 and
 * SmolLM3.  CMOL_TOK_GPT2 (byte-level GPT-2 BPE) is stubbed for Phase 3
 * and will be fully implemented if needed for future model variants.
 *
 * Algorithm summary:
 *   1. Pre-tokenize: inject ▁ dummy prefix; replace ' ' with ▁; look up
 *      each UTF-8 character in the vocab (byte fallback on miss).
 *   2. BPE merge loop: O(w²) greedy scan; practical for w ≤ BPE_WORK_MAX.
 *   3. cmol_tokenizer_build() constructs three index arrays so that both
 *      string→ID and pair→rank lookups run in O(log n).
 */


#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

/* =========================================================================
 * qsort comparator state — valid only during cmol_tokenizer_build().
 *
 * Not thread-safe for concurrent cmol_load() calls.  In practice, model
 * loading is single-threaded; document this limitation.
 * ====================================================================== */

static const char   **g_bld_vocab;   /* vocab pointer for vocab sort      */
static const int32_t *g_bld_ml;      /* merge_left  for merge sort        */
static const int32_t *g_bld_mr;      /* merge_right for merge sort        */

/* Sort vocab indices by the string they point to. */
static int cmp_vocab_idx(const void *a, const void *b) {
    int32_t ia = *(const int32_t *)a;
    int32_t ib = *(const int32_t *)b;
    return strcmp(g_bld_vocab[ia], g_bld_vocab[ib]);
}

/* Sort merge indices by packed (left<<32 | right). */
static int cmp_merge_idx(const void *a, const void *b) {
    int32_t ia = *(const int32_t *)a;
    int32_t ib = *(const int32_t *)b;
    uint64_t ka = ((uint64_t)(uint32_t)g_bld_ml[ia] << 32) | (uint32_t)g_bld_mr[ia];
    uint64_t kb = ((uint64_t)(uint32_t)g_bld_ml[ib] << 32) | (uint32_t)g_bld_mr[ib];
    if (ka < kb) return -1;
    if (ka > kb) return  1;
    return 0;
}

/* =========================================================================
 * Binary search helpers
 * ====================================================================== */

/*
 * find_token_id — return the vocab index whose string equals `s`, or -1.
 * Requires tok->vocab_sort_idx to be populated by cmol_tokenizer_build().
 */
static int32_t find_token_id(const cmol_tokenizer_t *tok, const char *s) {
    if (!tok->vocab_sort_idx || tok->vocab_size == 0) return -1;
    int lo = 0, hi = tok->vocab_size - 1;
    while (lo <= hi) {
        int     mid = (lo + hi) >> 1;
        int32_t idx = tok->vocab_sort_idx[mid];
        int     cmp = strcmp(tok->vocab[idx], s);
        if (cmp == 0) return idx;
        if (cmp  < 0) lo = mid + 1;
        else          hi = mid - 1;
    }
    return -1;
}

/*
 * find_merge_rank — return the rank (original merge index = priority) of
 * the merge rule (L, R), or -1 if the pair has no merge rule.
 * Rank 0 = highest priority (applied first).
 */
static int32_t find_merge_rank(const cmol_tokenizer_t *tok,
                                int32_t L, int32_t R) {
    if (!tok->msort_idx || tok->n_merges == 0 || L < 0 || R < 0) return -1;
    uint64_t target = ((uint64_t)(uint32_t)L << 32) | (uint32_t)R;
    int lo = 0, hi = tok->n_merges - 1;
    while (lo <= hi) {
        int     mid = (lo + hi) >> 1;
        int32_t idx = tok->msort_idx[mid];
        uint64_t key = ((uint64_t)(uint32_t)tok->merge_left[idx]  << 32)
                     |              (uint32_t)tok->merge_right[idx];
        if (key == target) return idx;   /* idx IS the priority rank */
        if (key  < target) lo = mid + 1;
        else               hi = mid - 1;
    }
    return -1;
}

/* =========================================================================
 * UTF-8 helpers
 * ====================================================================== */

/* Returns the byte length of the UTF-8 sequence that starts with byte `c`. */
static int utf8_seqlen(unsigned char c) {
    if ((c & 0x80) == 0x00) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1; /* invalid / continuation byte — treat as 1 */
}

/* Parse one UTF-8 codepoint from *p, advancing *p past it. */
static unsigned int utf8_next_cp(const char **p) {
    const unsigned char *u = (const unsigned char *)*p;
    unsigned int cp; int seqlen, i;
    if (!u || !*u) return 0;
    if      (u[0] < 0x80u) { cp = u[0]; seqlen = 1; }
    else if (u[0] < 0xC0u) { (*p)++; return 0xFFFDu; } /* bare continuation */
    else if (u[0] < 0xE0u) { cp = u[0] & 0x1Fu; seqlen = 2; }
    else if (u[0] < 0xF0u) { cp = u[0] & 0x0Fu; seqlen = 3; }
    else                   { cp = u[0] & 0x07u; seqlen = 4; }
    for (i = 1; i < seqlen; i++) {
        if ((u[i] & 0xC0u) != 0x80u) break; /* truncated */
        cp = (cp << 6) | (u[i] & 0x3Fu);
    }
    *p += seqlen;
    return cp;
}

/* Encode Unicode codepoint cp as UTF-8 into buf[]; return byte length (1-4). */
static int cp_to_utf8(unsigned int cp, char buf[4]) {
    if (cp < 0x80u) {
        buf[0] = (char)cp; return 1;
    } else if (cp < 0x800u) {
        buf[0] = (char)(0xC0u | (cp >> 6));
        buf[1] = (char)(0x80u | (cp & 0x3Fu)); return 2;
    } else if (cp < 0x10000u) {
        buf[0] = (char)(0xE0u | (cp >> 12));
        buf[1] = (char)(0x80u | ((cp >> 6) & 0x3Fu));
        buf[2] = (char)(0x80u | (cp & 0x3Fu)); return 3;
    } else {
        buf[0] = (char)(0xF0u | (cp >> 18));
        buf[1] = (char)(0x80u | ((cp >> 12) & 0x3Fu));
        buf[2] = (char)(0x80u | ((cp >> 6)  & 0x3Fu));
        buf[3] = (char)(0x80u | (cp & 0x3Fu)); return 4;
    }
}

/* =========================================================================
 * GPT-2 byte-to-unicode mapping
 *
 * GPT-2 represents each byte 0x00-0xFF as a single Unicode codepoint so
 * that every byte sequence has a lossless text representation:
 *
 *   "Kept" bytes (map to themselves):
 *     0x21-0x7E  (printable ASCII: '!' to '~')
 *     0xA1-0xAC  (¡ to ¬)
 *     0xAE-0xFF  (® to ÿ)
 *
 *   "Extra" bytes (68 values, mapped to U+0100..U+0143 in byte-value order):
 *     0x00-0x20  → U+0100-U+0120  (U+0120 = Ġ = space 0x20)
 *     0x7F       → U+0121
 *     0x80-0xA0  → U+0122-U+0142
 *     0xAD       → U+0143
 * ====================================================================== */

/* Input byte → GPT-2 Unicode codepoint. */
static unsigned int gpt2_byte_to_cp(unsigned char b) {
    if (b <= 0x20u)              return 0x0100u + b;          /* 0x00-0x20 */
    if (b <= 0x7Eu)              return b;                    /* 0x21-0x7E */
    if (b == 0x7Fu)              return 0x0121u;
    if (b <= 0xA0u)              return 0x0122u + (b - 0x80u); /* 0x80-0xA0 */
    if (b == 0xADu)              return 0x0143u;
    return b;                                                  /* kept */
}

/* GPT-2 Unicode codepoint → original byte (inverse of gpt2_byte_to_cp). */
static unsigned char gpt2_cp_to_byte(unsigned int cp) {
    if (cp >= 0x0100u && cp <= 0x0120u) return (unsigned char)(cp - 0x0100u);
    if (cp == 0x0121u)                  return 0x7Fu;
    if (cp >= 0x0122u && cp <= 0x0142u) return (unsigned char)(0x80u + (cp - 0x0122u));
    if (cp == 0x0143u)                  return 0xADu;
    return (unsigned char)(cp & 0xFFu); /* kept bytes pass through */
}

/* =========================================================================
 * cmol_tokenizer_build
 * ====================================================================== */

cmol_err_t cmol_tokenizer_build(cmol_tokenizer_t *tok, cmol_arena_t *arena) {
    if (!tok || !arena)           return CMOL_ERR_ARGS;
    if (tok->vocab_size == 0)     return CMOL_ERR_INVALID;

    int V = tok->vocab_size;
    int M = tok->n_merges;

    /* ------------------------------------------------------------------
     * 1. vocab_sort_idx[V] — sorted by vocab string for string→ID lookup.
     * ------------------------------------------------------------------ */
    tok->vocab_sort_idx =
        (int32_t *)cmol_arena_alloc_n(arena, (size_t)V, sizeof(int32_t));
    if (!tok->vocab_sort_idx) return CMOL_ERR_OOM;

    for (int i = 0; i < V; i++) tok->vocab_sort_idx[i] = (int32_t)i;
    g_bld_vocab = tok->vocab;
    qsort(tok->vocab_sort_idx, (size_t)V, sizeof(int32_t), cmp_vocab_idx);

    /* ------------------------------------------------------------------
     * 2. msort_idx[M] — sorted by pack64(left, right) for pair→rank.
     * ------------------------------------------------------------------ */
    if (M > 0) {
        tok->msort_idx =
            (int32_t *)cmol_arena_alloc_n(arena, (size_t)M, sizeof(int32_t));
        if (!tok->msort_idx) return CMOL_ERR_OOM;

        for (int i = 0; i < M; i++) tok->msort_idx[i] = (int32_t)i;
        g_bld_ml = tok->merge_left;
        g_bld_mr = tok->merge_right;
        qsort(tok->msort_idx, (size_t)M, sizeof(int32_t), cmp_merge_idx);
    }

    /* ------------------------------------------------------------------
     * 3. merge_result[M] — result token ID for each merge rule.
     *    Concatenate vocab[left] + vocab[right], look up in sorted vocab.
     * ------------------------------------------------------------------ */
    if (M > 0) {
        tok->merge_result =
            (int32_t *)cmol_arena_alloc_n(arena, (size_t)M, sizeof(int32_t));
        if (!tok->merge_result) return CMOL_ERR_OOM;

        char concat[256];
        for (int i = 0; i < M; i++) {
            int32_t L = tok->merge_left[i];
            int32_t R = tok->merge_right[i];
            if (L < 0 || L >= V || R < 0 || R >= V) {
                tok->merge_result[i] = tok->unk_id;
                continue;
            }
            const char *ls = tok->vocab[L];
            const char *rs = tok->vocab[R];
            size_t ll = strlen(ls), rl = strlen(rs);
            if (ll + rl >= sizeof concat) {
                tok->merge_result[i] = tok->unk_id;
                continue;
            }
            memcpy(concat, ls, ll);
            memcpy(concat + ll, rs, rl);
            concat[ll + rl] = '\0';
            int32_t rid = find_token_id(tok, concat);
            tok->merge_result[i] = (rid >= 0) ? rid : tok->unk_id;
        }
    }

    /* ------------------------------------------------------------------
     * 4. decoded_vocab[V] — output form of each token:
     *      <0xNN> byte tokens    → single-byte string  "\xNN"
     *      ▁ (U+2581) prefix     → replace ▁ with ' '
     *      everything else       → point directly at vocab[i]
     * ------------------------------------------------------------------ */
    tok->decoded_vocab =
        (const char **)cmol_arena_alloc_n(arena, (size_t)V, sizeof(char *));
    if (!tok->decoded_vocab) return CMOL_ERR_OOM;

    for (int i = 0; i < V; i++) {
        const char *s = tok->vocab[i];

        /* ------------------------------------------------------------------
         * GPT-2 byte-level BPE: every vocab token is a sequence of GPT-2
         * Unicode codepoints; decode each codepoint back to the original byte.
         * Control tokens (special markers) are kept as-is.
         * ------------------------------------------------------------------ */
        if (tok->tok_model == CMOL_TOK_GPT2) {
            int is_ctrl = tok->token_type &&
                          tok->token_type[i] == CMOL_TOKEN_CONTROL;
            if (!is_ctrl) {
                /* Walk the UTF-8 string, decode each codepoint → byte. */
                char dbuf[256];
                int  dlen = 0;
                const char *p = s;
                while (*p && dlen < (int)sizeof(dbuf) - 1) {
                    unsigned int cp = utf8_next_cp(&p);
                    if (!cp) break;
                    dbuf[dlen++] = (char)gpt2_cp_to_byte(cp);
                }
                dbuf[dlen] = '\0';
                if (dlen > 0) {
                    char *ds = (char *)cmol_arena_alloc(arena,
                                                        (size_t)dlen + 1, 1);
                    if (!ds) return CMOL_ERR_OOM;
                    memcpy(ds, dbuf, (size_t)dlen + 1);
                    tok->decoded_vocab[i] = ds;
                    continue;
                }
            }
            /* Control token or empty decode → alias raw string. */
            tok->decoded_vocab[i] = s;
            continue;
        }

        /* ------------------------------------------------------------------
         * SentencePiece / Llama-style BPE
         * ------------------------------------------------------------------ */

        /* Byte token?  Prefer the token_type flag; fall back to heuristic. */
        int is_byte = tok->token_type &&
                      (tok->token_type[i] == CMOL_TOKEN_BYTE);
        if (!is_byte) {
            /* Heuristic: exactly "<0x??>" — 6 chars */
            int n = (int)strlen(s);
            if (n == 6 && s[0]=='<' && s[1]=='0' && s[2]=='x' && s[5]=='>')
                is_byte = 1;
        }
        if (is_byte) {
            unsigned bv = 0;
            if (sscanf(s, "<0x%02x>", &bv) == 1) {
                char *ds = (char *)cmol_arena_alloc(arena, 2, 1);
                if (!ds) return CMOL_ERR_OOM;
                ds[0] = (char)(unsigned char)bv;
                ds[1] = '\0';
                tok->decoded_vocab[i] = ds;
                continue;
            }
        }

        /* ▁ prefix (U+2581 = 0xE2 0x96 0x81)? */
        if ((unsigned char)s[0] == 0xE2 &&
            (unsigned char)s[1] == 0x96 &&
            (unsigned char)s[2] == 0x81) {
            size_t rest = strlen(s + 3);
            char *ds = (char *)cmol_arena_alloc(arena, rest + 2, 1);
            if (!ds) return CMOL_ERR_OOM;
            ds[0] = ' ';
            if (rest) memcpy(ds + 1, s + 3, rest);
            ds[rest + 1] = '\0';
            tok->decoded_vocab[i] = ds;
            continue;
        }

        /* No transformation needed — alias original string. */
        tok->decoded_vocab[i] = s;
    }

    return CMOL_OK;
}

/* =========================================================================
 * cmol_tokenizer_encode
 * ====================================================================== */

/*
 * Maximum number of initial tokens (before merges) that the working buffer
 * can hold.  One token per UTF-8 byte in the worst case (all byte fallback).
 * Covers prompts up to ~4 KB; O(w²) BPE merge is fast enough for w ≤ 4096.
 */
#define BPE_WORK_MAX 4096

/* UTF-8 encoding of ▁ (U+2581 LOWER ONE EIGHTH BLOCK — SP word boundary) */
static const char SPIECE_PREFIX[4] = "\xe2\x96\x81";  /* 3 bytes + NUL */

int cmol_tokenizer_encode(const cmol_tokenizer_t *tok,
                           const char             *text,
                           int32_t                *out,
                           int                     out_cap,
                           int                     add_bos) {
    if (!tok || !text || !out || out_cap <= 0) return CMOL_ERR_ARGS;
    if (!tok->vocab_sort_idx || !tok->decoded_vocab)
        return CMOL_ERR_UNSUPPORTED; /* build() not called */

    int n = 0;          /* tokens written to `out`          */
    int overflowed = 0; /* BPE work buffer overflowed       */

    /* Optional BOS special token */
    if (add_bos && tok->bos_id >= 0) {
        if (n >= out_cap) return CMOL_ERR_TRUNC;
        out[n++] = tok->bos_id;
    }

    /* ----------------------------------------------------------------
     * Pre-tokenisation — build the initial token sequence in `work[]`.
     *
     * For CMOL_TOK_LLAMA (SentencePiece):
     *   • Inject one ▁ dummy prefix before the first input character.
     *   • Replace each ASCII space with ▁.
     *   • Look up each resulting UTF-8 character in the vocab via
     *     binary search.  On miss, emit one <0xNN> byte token per byte.
     *
     * For CMOL_TOK_GPT2: stub — treat like LLAMA for now.
     * ---------------------------------------------------------------- */
    int32_t work[BPE_WORK_MAX];
    int     w = 0;

    /* Emit one character (given as pointer + byte length) into work[]. */
    /* On vocab miss: emit byte fallback tokens. */
#define EMIT_CHAR(ptr, len)  do {                                       \
    char _cb[8];                                                        \
    memcpy(_cb, (ptr), (size_t)(len));                                  \
    _cb[(len)] = '\0';                                                  \
    int32_t _id = find_token_id(tok, _cb);                              \
    if (_id >= 0) {                                                     \
        if (w < BPE_WORK_MAX) work[w++] = _id;                         \
        else overflowed = 1;                                            \
    } else {                                                            \
        int _b;                                                         \
        for (_b = 0; _b < (len) && !overflowed; _b++) {                \
            char _bt[8];                                                \
            snprintf(_bt, sizeof _bt, "<0x%02x>",                      \
                     (unsigned char)(ptr)[_b]);                         \
            int32_t _bid = find_token_id(tok, _bt);                    \
            if (_bid < 0) _bid = tok->unk_id;                          \
            if (_bid >= 0) {                                            \
                if (w < BPE_WORK_MAX) work[w++] = _bid;                \
                else overflowed = 1;                                    \
            }                                                           \
        }                                                               \
    }                                                                   \
} while (0)

    if (tok->tok_model == CMOL_TOK_GPT2) {
        /* ----------------------------------------------------------------
         * GPT-2 byte-level pre-tokenisation:
         * Iterate byte by byte, convert each to its GPT-2 Unicode char,
         * encode as UTF-8, and look up as a single-codepoint vocab token.
         *
         * Control tokens (e.g. <|im_start|>) are matched verbatim first
         * and emitted directly, bypassing the byte-level encoding.
         * ---------------------------------------------------------------- */
        const char *p = text;
        while (*p && !overflowed) {
            /* Greedy longest control-token match at current position. */
            if (tok->token_type) {
                int32_t best_id  = -1;
                size_t  best_len = 0;
                int     v;
                for (v = 0; v < tok->vocab_size; v++) {
                    if (tok->token_type[v] != CMOL_TOKEN_CONTROL) continue;
                    const char *ts = tok->vocab[v];
                    size_t tl = strlen(ts);
                    if (tl > best_len && strncmp(p, ts, tl) == 0) {
                        best_len = tl;
                        best_id  = v;
                    }
                }
                if (best_id >= 0) {
                    if (w < BPE_WORK_MAX) work[w++] = best_id;
                    else overflowed = 1;
                    p += best_len;
                    continue;
                }
            }
            /* Normal byte: map to GPT-2 Unicode codepoint and look up. */
            char vtok[5];
            int vlen = cp_to_utf8(gpt2_byte_to_cp((unsigned char)*p++), vtok);
            vtok[vlen] = '\0';
            int32_t id = find_token_id(tok, vtok);
            if (id < 0) id = tok->unk_id;
            if (id >= 0) {
                if (w < BPE_WORK_MAX) work[w++] = (int32_t)id;
                else overflowed = 1;
            }
        }
    } else {
        /* ----------------------------------------------------------------
         * SentencePiece / Llama-style pre-tokenisation:
         * Inject ▁ dummy prefix; replace spaces with ▁; look up UTF-8 chars.
         *
         * Special token handling: before each character, check if any
         * CMOL_TOKEN_CONTROL token matches verbatim at the current position.
         * If so, emit it directly (bypassing BPE) — this handles ChatML
         * delimiters like <|im_start|> and <|im_end|>.
         * ---------------------------------------------------------------- */
        const char *p = text;

        /* Whether we have injected the ▁ prefix yet (only on first real char) */
        int prefix_done = 0;

        while (*p && !overflowed) {
            /* Try control-token match (greedy longest) at current position. */
            if (tok->token_type) {
                int32_t best_id  = -1;
                size_t  best_len = 0;
                int     v;
                for (v = 0; v < tok->vocab_size; v++) {
                    if (tok->token_type[v] != CMOL_TOKEN_CONTROL) continue;
                    const char *ts = tok->vocab[v];
                    size_t tl = strlen(ts);
                    if (tl > best_len && strncmp(p, ts, tl) == 0) {
                        best_len = tl;
                        best_id  = v;
                    }
                }
                if (best_id >= 0) {
                    /* Control token matched — emit directly, reset ▁ state */
                    if (w < BPE_WORK_MAX) work[w++] = best_id;
                    else overflowed = 1;
                    p += best_len;
                    prefix_done = 0; /* next real text needs a fresh ▁ prefix */
                    continue;
                }
            }

            /* Normal character: inject ▁ prefix on first character of segment */
            if (!prefix_done && *p != '\0') {
                EMIT_CHAR(SPIECE_PREFIX, 3);
                prefix_done = 1;
            }

            if (*p == ' ') {
                /* Space → ▁ (already handles segment boundary) */
                EMIT_CHAR(SPIECE_PREFIX, 3);
                p++;
            } else {
                int len = utf8_seqlen((unsigned char)*p);
                EMIT_CHAR(p, len);
                p += len;
            }
        }
    }

#undef EMIT_CHAR

    /* ----------------------------------------------------------------
     * BPE merge loop
     *
     * Greedy O(w²) scan: find the adjacent pair with the lowest merge
     * rank (highest priority), apply it, repeat until no merges remain.
     * ---------------------------------------------------------------- */
    if (tok->n_merges > 0 && tok->merge_result && tok->msort_idx) {
        while (w > 1) {
            int     best_pos  = -1;
            int32_t best_rank = INT32_MAX;

            for (int i = 0; i < w - 1; i++) {
                int32_t rank = find_merge_rank(tok, work[i], work[i + 1]);
                if (rank >= 0 && rank < best_rank) {
                    best_rank = rank;
                    best_pos  = i;
                }
            }
            if (best_pos < 0) break; /* no mergeable pair remains */

            /* Replace pair with the merged token, shift array left. */
            work[best_pos] = tok->merge_result[best_rank];
            for (int i = best_pos + 1; i < w - 1; i++)
                work[i] = work[i + 1];
            w--;
        }
    }

    /* ----------------------------------------------------------------
     * Write merged tokens to `out`.
     * ---------------------------------------------------------------- */
    int trunc = overflowed;
    for (int i = 0; i < w; i++) {
        if (n >= out_cap) { trunc = 1; break; }
        out[n++] = work[i];
    }

    return trunc ? CMOL_ERR_TRUNC : n;
}

/* =========================================================================
 * cmol_tokenizer_decode_token
 * ====================================================================== */

const char *cmol_tokenizer_decode_token(const cmol_tokenizer_t *tok,
                                         int32_t                 token_id) {
    if (!tok || token_id < 0 || token_id >= tok->vocab_size) return NULL;
    /* Prefer decoded form (▁→space, byte tokens expanded). */
    if (tok->decoded_vocab) return tok->decoded_vocab[token_id];
    /* Fallback: raw vocab string (build() not yet called). */
    return tok->vocab ? tok->vocab[token_id] : NULL;
}

/* ===== src/quant.c ===== */
/*
 * quant.c — quantisation kernels and SIMD dispatch
 * Included by src/cmol.c (unity build); do not compile standalone.
 *
 * Supported quantised formats:
 *   Q8_0 : 32 values / block,  34 bytes/block  (float16 scale + 32 × int8)
 *   Q4_K : 256 values / block, 144 bytes/block (Q4_K_M super-block)
 *   Q6_K : 256 values / block, 210 bytes/block (Q6_K super-block)
 *   F16  : passthrough, no block structure
 *   F32  : passthrough, no block structure
 *
 * SIMD dispatch:
 *   cmol_kernels_select() detects CPU features at run time (via platform.c)
 *   and returns a kernel table pointing to the fastest available matmul.
 *   Each SIMD kernel is annotated with __attribute__((target(...))) so it
 *   can be compiled in the same translation unit as the scalar fallback even
 *   when the global -march flag does not enable the target ISA.
 */

#include <string.h>  /* memcpy */
#include <stdint.h>

/* SIMD headers — guarded so the unity build compiles everywhere */
#if defined(__x86_64__) || defined(_M_X64)
#  include <immintrin.h>   /* AVX2, AVX-512 */
#endif

#if defined(__ARM_NEON) || defined(__aarch64__)
#  include <arm_neon.h>
#endif

/* =========================================================================
 * Float16 → Float32 (scalar, portable)
 * ====================================================================== */

static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1FU;
    uint32_t mant = h & 0x3FFU;
    uint32_t f32;

    if (exp == 0u) {
        if (mant == 0u) {
            f32 = sign;                        /* ±0 */
        } else {
            /* Subnormal: normalise into the f32 range */
            exp = 1u;
            while (!(mant & 0x400U)) { mant <<= 1; exp--; }
            mant &= 0x3FFU;
            f32 = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
        }
    } else if (exp == 31u) {
        f32 = sign | (0xFFu << 23) | (mant << 13); /* Inf / NaN */
    } else {
        f32 = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
    }

    float r;
    memcpy(&r, &f32, sizeof r);
    return r;
}

/* =========================================================================
 * Quantised block layouts
 *
 * __attribute__((packed)) ensures no implicit padding between fields.
 * The compile-time assertions below verify the expected sizes.
 * ====================================================================== */

#if defined(__GNUC__) || defined(__clang__)
#  define CMOL_PACKED __attribute__((packed))
#else
#  define CMOL_PACKED  /* MSVC: use #pragma pack(push,1) around structs */
#endif

/* Q5_0 — 22 bytes */
typedef struct CMOL_PACKED {
    uint16_t d;       /* float16 scale                  */
    uint8_t  qh[4];   /* 5th bit of each of 32 values   */
    int8_t   qs[16];  /* lower 4 bits, 2 values/byte    */
} q5_0_block_t;

/* Q8_0 — 34 bytes */
typedef struct CMOL_PACKED {
    uint16_t d;       /* float16 scale                  */
    int8_t   qs[32];  /* 32 × int8 quantised values     */
} q8_0_block_t;

/* Q4_K — 144 bytes (super-block of 8 × 32-value sub-blocks) */
typedef struct CMOL_PACKED {
    uint16_t d;          /* float16 super-block scale for scales  */
    uint16_t dmin;       /* float16 super-block scale for mins    */
    uint8_t  scales[12]; /* 8×6-bit scale values + 8×6-bit mins  */
    uint8_t  qs[128];    /* 256 × 4-bit quantised values          */
} q4_k_block_t;

/* Q6_K — 210 bytes (super-block of 16 × 16-value sub-blocks) */
typedef struct CMOL_PACKED {
    uint8_t  ql[128];    /* lower 4 bits of each of 256 values   */
    uint8_t  qh[64];     /* upper 2 bits of each of 256 values   */
    int8_t   scales[16]; /* per-16-value int8 scales              */
    uint16_t d;          /* float16 super-block scale             */
} q6_k_block_t;

/* Compile-time size checks */
typedef char chk_q5_0[(sizeof(q5_0_block_t)  == 22)  ? 1 : -1];
typedef char chk_q8_0[(sizeof(q8_0_block_t)  == 34)  ? 1 : -1];
typedef char chk_q4_k[(sizeof(q4_k_block_t)  == 144) ? 1 : -1];
typedef char chk_q6_k[(sizeof(q6_k_block_t)  == 210) ? 1 : -1];

/* =========================================================================
 * Q4_K scale/min extraction
 *
 * The 12 bytes of `scales` pack 8 × 6-bit scale values and 8 × 6-bit min
 * values using the following layout (j = sub-block index 0..7):
 *
 *   j = 0..3:  sc = scales[j]   & 0x3F
 *              mn = scales[j+4] & 0x3F
 *   j = 4..7:  sc = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4)
 *              mn = (scales[j+4] >> 4)   | ((scales[j-0] >> 6) << 4)
 * ====================================================================== */

static inline void q4k_get_scale_min(int j, const uint8_t *s,
                                      uint8_t *sc, uint8_t *mn) {
    if (j < 4) {
        *sc = s[j]   & 0x3F;
        *mn = s[j+4] & 0x3F;
    } else {
        *sc = (s[j+4] & 0x0F) | ((s[j-4] >> 6) << 4);
        *mn = (s[j+4] >> 4)   | ((s[j-0] >> 6) << 4);
    }
}

/* =========================================================================
 * Per-block dequantisation helpers
 *
 * Each writes exactly block_size floats to dst (32 for Q8_0, 256 for Q4/Q6).
 * These are small enough that the compiler inlines / auto-vectorises them
 * in the SIMD kernel functions.
 * ====================================================================== */

/*
 * Q5_0 block layout:
 *   qs[j]  holds the lower nibble  of value j       (j = 0..15, first  half)
 *              and the upper nibble of value j + 16  (j = 0..15, second half).
 *   qh bits 0-15  hold the 5th bit of values  0-15 (bit j → value j).
 *   qh bits 16-31 hold the 5th bit of values 16-31 (bit j+16 → value j+16).
 * Combined value [0..31] → subtract 16 → signed [-16..15].
 */
static void dequant_block_q5_0(const q5_0_block_t *b, float *dst) {
    float    d = f16_to_f32(b->d);
    uint32_t qh;
    int      j;
    memcpy(&qh, b->qh, 4);
    for (j = 0; j < 16; j++) {
        uint8_t xh0 = (uint8_t)(((qh >>  j)      & 1u) << 4);
        uint8_t xh1 = (uint8_t)(((qh >> (j + 16)) & 1u) << 4);
        uint8_t q   = (uint8_t)b->qs[j]; /* cast to unsigned before shifting */
        dst[j]      = d * (float)((int)((q & 0x0Fu) | xh0) - 16);
        dst[j + 16] = d * (float)((int)((q >>  4  ) | xh1) - 16);
    }
}

static void dequant_block_q8_0(const q8_0_block_t *b, float *dst) {
    float d = f16_to_f32(b->d);
    int i;
    for (i = 0; i < 32; i++) dst[i] = d * (float)b->qs[i];
}

static void dequant_block_q4_k(const q4_k_block_t *b, float *dst) {
    float d    = f16_to_f32(b->d);
    float dmin = f16_to_f32(b->dmin);
    int sub, k;

    for (sub = 0; sub < 8; sub++) {
        uint8_t sc, mn;
        q4k_get_scale_min(sub, b->scales, &sc, &mn);
        float db = d    * (float)sc;
        float mb = dmin * (float)mn;

        /* Pairs of sub-blocks share the same 32-byte qs chunk:
         *   sub 0,1 → qs[  0.. 31]   sub 2,3 → qs[ 32.. 63]
         *   sub 4,5 → qs[ 64.. 95]   sub 6,7 → qs[ 96..127]
         * Even sub → lower nibble (shift 0); odd sub → upper nibble (shift 4).
         * Matches llama.cpp dequantize_row_q4_K (64-value inner loop). */
        const uint8_t *q = b->qs + (sub / 2) * 32;
        int shift = (sub & 1) ? 4 : 0;
        for (k = 0; k < 32; k++) {
            dst[sub * 32 + k] = db * (float)((q[k] >> shift) & 0xFu) - mb;
        }
    }
}

static void dequant_block_q6_k(const q6_k_block_t *b, float *dst) {
    float d = f16_to_f32(b->d);
    int i;

    /*
     * Q6_K output ordering — derived from llama.cpp dequantize_row_q6_K.
     * The 256 outputs are arranged in 8 groups of 32 (g8 = i/32, j32 = i%32):
     *
     * ql (128 bytes):
     *   g8  0,1 → bytes   0-31,  32-63  lower nibble (& 0xF)
     *   g8  2,3 → bytes   0-31,  32-63  upper nibble (>> 4)
     *   g8  4,5 → bytes  64-95,  96-127 lower nibble
     *   g8  6,7 → bytes  64-95,  96-127 upper nibble
     *   Compact: ql_idx = (g8<4 ? 0 : 64) + (g8&1)*32 + j32
     *            ql_shift = (g8&2) ? 4 : 0
     *
     * qh (64 bytes): 2-bit groups, two bit-pairs per byte
     *   g8  0-3 → qh[ 0..31], bit-pair (g8%4)*2
     *   g8  4-7 → qh[32..63], bit-pair (g8%4)*2
     *   Compact: qh_idx = (g8<4 ? 0 : 32) + j32
     *            qh_shift = (g8 & 3) * 2
     *
     * scales: 16 int8 scale values, one per 16 outputs → scales[i/16]
     */
    for (i = 0; i < 256; i++) {
        int     g8     = i / 32;
        int     j32    = i % 32;
        int     ql_idx = (g8 < 4 ? 0 : 64) + (g8 & 1) * 32 + j32;
        int     ql_sh  = (g8 & 2) ? 4 : 0;
        uint8_t lo     = (b->ql[ql_idx] >> ql_sh) & 0xFu;
        int     qh_idx = (g8 < 4 ? 0 : 32) + j32;
        int     qh_sh  = (g8 & 3) * 2;
        uint8_t hi     = (b->qh[qh_idx] >> qh_sh) & 0x3u;
        int     q      = (int)(lo | ((unsigned)hi << 4)) - 32;
        dst[i] = d * (float)b->scales[i / 16] * (float)q;
    }
}

/* =========================================================================
 * cmol_dequant_row
 *
 * Public entry point: dequantise `n` values from a packed quantised row.
 * `n` must be a multiple of the format's block size.
 * ====================================================================== */

void cmol_dequant_row(const void *src, float *dst, int n, cmol_dtype_t dtype) {
    if (!src || !dst || n <= 0) return;

    switch (dtype) {

    case CMOL_DTYPE_F32: {
        memcpy(dst, src, (size_t)n * sizeof(float));
        break;
    }

    case CMOL_DTYPE_F16: {
        const uint16_t *s = (const uint16_t *)src;
        int i;
        for (i = 0; i < n; i++) dst[i] = f16_to_f32(s[i]);
        break;
    }

    case CMOL_DTYPE_Q5_0: {
        const q5_0_block_t *b = (const q5_0_block_t *)src;
        int bi, nb = n / 32;
        for (bi = 0; bi < nb; bi++) dequant_block_q5_0(&b[bi], dst + bi * 32);
        break;
    }

    case CMOL_DTYPE_Q8_0: {
        const q8_0_block_t *b = (const q8_0_block_t *)src;
        int bi, nb = n / 32;
        for (bi = 0; bi < nb; bi++) dequant_block_q8_0(&b[bi], dst + bi * 32);
        break;
    }

    case CMOL_DTYPE_Q4_K: {
        const q4_k_block_t *b = (const q4_k_block_t *)src;
        int bi, nb = n / 256;
        for (bi = 0; bi < nb; bi++) dequant_block_q4_k(&b[bi], dst + bi * 256);
        break;
    }

    case CMOL_DTYPE_Q6_K: {
        const q6_k_block_t *b = (const q6_k_block_t *)src;
        int bi, nb = n / 256;
        for (bi = 0; bi < nb; bi++) dequant_block_q6_k(&b[bi], dst + bi * 256);
        break;
    }

    default:
        break;
    }
}

/* =========================================================================
 * Block-level dispatch helpers (used by all matmul kernels)
 *
 * Returns the block size in values and populates `bsz` with the block
 * size in bytes.  Returns 0 for unknown types.
 * ====================================================================== */

static inline int block_params(cmol_dtype_t dtype, size_t *bsz) {
    switch (dtype) {
    case CMOL_DTYPE_F32:  *bsz = sizeof(float);          return 1;
    case CMOL_DTYPE_F16:  *bsz = sizeof(uint16_t);       return 1;
    case CMOL_DTYPE_Q5_0: *bsz = sizeof(q5_0_block_t);   return 32;
    case CMOL_DTYPE_Q8_0: *bsz = sizeof(q8_0_block_t);   return 32;
    case CMOL_DTYPE_Q4_K: *bsz = sizeof(q4_k_block_t);   return 256;
    case CMOL_DTYPE_Q6_K: *bsz = sizeof(q6_k_block_t);   return 256;
    default:              *bsz = 0;                       return 0;
    }
}

/* Dequantise one block at address `blk` of `dtype` into `tmp`.
 * Assumes `blk` is correctly aligned (it comes from the GGUF mmap). */
static inline void dequant_block(const void *blk, float *tmp, cmol_dtype_t dtype) {
    switch (dtype) {
    case CMOL_DTYPE_Q5_0: dequant_block_q5_0((const q5_0_block_t *)blk, tmp); break;
    case CMOL_DTYPE_Q8_0: dequant_block_q8_0((const q8_0_block_t *)blk, tmp); break;
    case CMOL_DTYPE_Q4_K: dequant_block_q4_k((const q4_k_block_t *)blk, tmp); break;
    case CMOL_DTYPE_Q6_K: dequant_block_q6_k((const q6_k_block_t *)blk, tmp); break;
    default: break;
    }
}

/* =========================================================================
 * Scalar matmul
 *
 *   out[m × n] = A[m × k]  ·  B[k × n]
 *
 * A is a quantised weight matrix (row-major, packed blocks).
 * B is a float32 activation matrix (row-major: B[ki][ni] = B[ki*n + ni]).
 *
 * Processed one block of A at a time — no k-sized scratch buffer on the
 * heap; the local `tmp[256]` covers the largest block size.
 * ====================================================================== */

void cmol_matmul_scalar(float *out, const void *a, const float *b,
                         int m, int k, int n, cmol_dtype_t dtype) {
    size_t bsz;
    int    bs = block_params(dtype, &bsz);
    if (!bs) return;

    const uint8_t *abytes   = (const uint8_t *)a;
    size_t         row_bytes = (size_t)(k / bs) * bsz;
    float          tmp[256]; /* max block size */
    int mi, ni, bi;

    for (mi = 0; mi < m; mi++) {
        const uint8_t *row_a = abytes + (size_t)mi * row_bytes;

        for (ni = 0; ni < n; ni++) {
            float acc = 0.0f;

            if (dtype == CMOL_DTYPE_F32) {
                const float *af = (const float *)row_a;
                int ki;
                for (ki = 0; ki < k; ki++) acc += af[ki] * b[ki * n + ni];

            } else if (dtype == CMOL_DTYPE_F16) {
                const uint16_t *ah = (const uint16_t *)row_a;
                int ki;
                for (ki = 0; ki < k; ki++)
                    acc += f16_to_f32(ah[ki]) * b[ki * n + ni];

            } else {
                int nb = k / bs;
                for (bi = 0; bi < nb; bi++) {
                    int   base = bi * bs;
                    int   j;
                    dequant_block(row_a + (size_t)bi * bsz, tmp, dtype);
                    for (j = 0; j < bs; j++)
                        acc += tmp[j] * b[(base + j) * n + ni];
                }
            }

            out[mi * n + ni] = acc;
        }
    }
}

/* =========================================================================
 * AVX2 matmul  (x86-64, compiled with target("avx2,fma"))
 *
 * Dequantises one block at a time into a 256-element float buffer, then
 * accumulates the dot product using 256-bit FMA instructions (8 f32/cycle).
 *
 * For n == 1 (the common autoregressive inference case) B is loaded with
 * a contiguous 256-bit load; for n > 1 we scalar-gather into a small
 * local array to keep the inner loop clean.
 * ====================================================================== */

#if defined(__x86_64__) || defined(_M_X64)

/* Horizontal sum of a __m256 */
static __attribute__((target("avx,avx2"))) inline float hsum_m256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

__attribute__((target("avx2,fma")))
void cmol_matmul_avx2(float *out, const void *a, const float *b,
                       int m, int k, int n, cmol_dtype_t dtype) {
    size_t bsz;
    int    bs = block_params(dtype, &bsz);
    if (!bs) return;

    const uint8_t *abytes   = (const uint8_t *)a;
    size_t         row_bytes = (size_t)(k / bs) * bsz;
    float          tmp[256];
    int mi, ni, bi;

    for (mi = 0; mi < m; mi++) {
        const uint8_t *row_a = abytes + (size_t)mi * row_bytes;

        for (ni = 0; ni < n; ni++) {
            __m256 acc = _mm256_setzero_ps();

            if (dtype == CMOL_DTYPE_F32) {
                /* F32: pure AVX2 dot product, 8 floats/iter */
                const float *af = (const float *)row_a;
                int ki;
                for (ki = 0; ki + 7 < k; ki += 8) {
                    __m256 av = _mm256_loadu_ps(af + ki);
                    __m256 bv = (n == 1)
                        ? _mm256_loadu_ps(b + ki)
                        : _mm256_set_ps(b[(ki+7)*n+ni], b[(ki+6)*n+ni],
                                        b[(ki+5)*n+ni], b[(ki+4)*n+ni],
                                        b[(ki+3)*n+ni], b[(ki+2)*n+ni],
                                        b[(ki+1)*n+ni], b[(ki+0)*n+ni]);
                    acc = _mm256_fmadd_ps(av, bv, acc);
                }
                float tail = hsum_m256(acc);
                /* scalar tail for k not a multiple of 8 */
                for (; ki < k; ki++) tail += af[ki] * b[ki * n + ni];
                out[mi * n + ni] = tail;
                continue; /* next ni */

            } else if (dtype == CMOL_DTYPE_F16) {
                /* F16: scalar path — rarely a bottleneck */
                const uint16_t *ah = (const uint16_t *)row_a;
                float s = 0.0f;
                int ki;
                for (ki = 0; ki < k; ki++)
                    s += f16_to_f32(ah[ki]) * b[ki * n + ni];
                out[mi * n + ni] = s;
                continue;
            }

            /* Quantised path: dequant one block, AVX2 dot product */
            int nb = k / bs;
            for (bi = 0; bi < nb; bi++) {
                int base = bi * bs;
                int j;
                dequant_block(row_a + (size_t)bi * bsz, tmp, dtype);

                for (j = 0; j + 7 < bs; j += 8) {
                    __m256 av = _mm256_loadu_ps(tmp + j);
                    __m256 bv;
                    if (n == 1) {
                        bv = _mm256_loadu_ps(b + base + j);
                    } else {
                        bv = _mm256_set_ps(
                            b[(base+j+7)*n+ni], b[(base+j+6)*n+ni],
                            b[(base+j+5)*n+ni], b[(base+j+4)*n+ni],
                            b[(base+j+3)*n+ni], b[(base+j+2)*n+ni],
                            b[(base+j+1)*n+ni], b[(base+j+0)*n+ni]);
                    }
                    acc = _mm256_fmadd_ps(av, bv, acc);
                }
                /* tail for bs not a multiple of 8 (Q8_0: bs=32, always ×8) */
                for (; j < bs; j++)
                    acc = _mm256_add_ps(acc,
                          _mm256_set1_ps(tmp[j] * b[(base+j)*n+ni]));
            }

            out[mi * n + ni] = hsum_m256(acc);
        }
    }
}

/* =========================================================================
 * AVX-512 matmul  (x86-64, target("avx512f"))
 *
 * Same structure as AVX2 but uses 512-bit ZMM registers (16 f32/cycle).
 * ====================================================================== */

__attribute__((target("avx512f")))
void cmol_matmul_avx512(float *out, const void *a, const float *b,
                          int m, int k, int n, cmol_dtype_t dtype) {
    size_t bsz;
    int    bs = block_params(dtype, &bsz);
    if (!bs) return;

    const uint8_t *abytes   = (const uint8_t *)a;
    size_t         row_bytes = (size_t)(k / bs) * bsz;
    float          tmp[256];
    int mi, ni, bi;

    for (mi = 0; mi < m; mi++) {
        const uint8_t *row_a = abytes + (size_t)mi * row_bytes;

        for (ni = 0; ni < n; ni++) {
            __m512 acc = _mm512_setzero_ps();

            if (dtype == CMOL_DTYPE_F32) {
                const float *af = (const float *)row_a;
                int ki;
                for (ki = 0; ki + 15 < k; ki += 16) {
                    __m512 av = _mm512_loadu_ps(af + ki);
                    __m512 bv;
                    if (n == 1) {
                        bv = _mm512_loadu_ps(b + ki);
                    } else {
                        float vals[16];
                        int jj;
                        for (jj = 0; jj < 16; jj++) vals[jj] = b[(ki+jj)*n+ni];
                        bv = _mm512_loadu_ps(vals);
                    }
                    acc = _mm512_fmadd_ps(av, bv, acc);
                }
                float tail = _mm512_reduce_add_ps(acc);
                for (; ki < k; ki++) tail += af[ki] * b[ki * n + ni];
                out[mi * n + ni] = tail;
                continue;

            } else if (dtype == CMOL_DTYPE_F16) {
                const uint16_t *ah = (const uint16_t *)row_a;
                float s = 0.0f;
                int ki;
                for (ki = 0; ki < k; ki++)
                    s += f16_to_f32(ah[ki]) * b[ki * n + ni];
                out[mi * n + ni] = s;
                continue;
            }

            int nb = k / bs;
            for (bi = 0; bi < nb; bi++) {
                int base = bi * bs;
                int j;
                dequant_block(row_a + (size_t)bi * bsz, tmp, dtype);

                for (j = 0; j + 15 < bs; j += 16) {
                    __m512 av = _mm512_loadu_ps(tmp + j);
                    __m512 bv;
                    if (n == 1) {
                        bv = _mm512_loadu_ps(b + base + j);
                    } else {
                        float vals[16];
                        int jj;
                        for (jj = 0; jj < 16; jj++)
                            vals[jj] = b[(base+j+jj)*n+ni];
                        bv = _mm512_loadu_ps(vals);
                    }
                    acc = _mm512_fmadd_ps(av, bv, acc);
                }
                /* tail (Q8_0 bs=32: 32/16=2 iterations, no tail) */
                for (; j < bs; j++)
                    acc = _mm512_add_ps(acc,
                          _mm512_set1_ps(tmp[j] * b[(base+j)*n+ni]));
            }

            out[mi * n + ni] = _mm512_reduce_add_ps(acc);
        }
    }
}

#endif /* x86_64 */

/* =========================================================================
 * NEON matmul  (AArch64 always; ARMv7 when compiled with -mfpu=neon)
 *
 * 4 × float32 lanes.  Same block-at-a-time structure as AVX2.
 * ====================================================================== */

#if defined(__ARM_NEON) || defined(__aarch64__)

void cmol_matmul_neon(float *out, const void *a, const float *b,
                       int m, int k, int n, cmol_dtype_t dtype) {
    size_t bsz;
    int    bs = block_params(dtype, &bsz);
    if (!bs) return;

    const uint8_t *abytes   = (const uint8_t *)a;
    size_t         row_bytes = (size_t)(k / bs) * bsz;
    float          tmp[256];
    int mi, ni, bi;

    for (mi = 0; mi < m; mi++) {
        const uint8_t *row_a = abytes + (size_t)mi * row_bytes;

        for (ni = 0; ni < n; ni++) {
            float32x4_t acc = vdupq_n_f32(0.0f);

            if (dtype == CMOL_DTYPE_F32) {
                const float *af = (const float *)row_a;
                int ki;
                for (ki = 0; ki + 3 < k; ki += 4) {
                    float32x4_t av = vld1q_f32(af + ki);
                    float32x4_t bv;
                    if (n == 1) {
                        bv = vld1q_f32(b + ki);
                    } else {
                        float vals[4] = {
                            b[(ki+0)*n+ni], b[(ki+1)*n+ni],
                            b[(ki+2)*n+ni], b[(ki+3)*n+ni]
                        };
                        bv = vld1q_f32(vals);
                    }
                    acc = vmlaq_f32(acc, av, bv);
                }
                /* horizontal sum */
                float32x2_t lo = vget_low_f32(acc);
                float32x2_t hi = vget_high_f32(acc);
                float32x2_t s  = vadd_f32(lo, hi);
                float tail = vget_lane_f32(vpadd_f32(s, s), 0);
                for (; ki < k; ki++) tail += af[ki] * b[ki * n + ni];
                out[mi * n + ni] = tail;
                continue;

            } else if (dtype == CMOL_DTYPE_F16) {
                const uint16_t *ah = (const uint16_t *)row_a;
                float s = 0.0f;
                int ki;
                for (ki = 0; ki < k; ki++)
                    s += f16_to_f32(ah[ki]) * b[ki * n + ni];
                out[mi * n + ni] = s;
                continue;
            }

            int nb = k / bs;
            for (bi = 0; bi < nb; bi++) {
                int base = bi * bs;
                int j;
                dequant_block(row_a + (size_t)bi * bsz, tmp, dtype);

                for (j = 0; j + 3 < bs; j += 4) {
                    float32x4_t av = vld1q_f32(tmp + j);
                    float32x4_t bv;
                    if (n == 1) {
                        bv = vld1q_f32(b + base + j);
                    } else {
                        float vals[4] = {
                            b[(base+j+0)*n+ni], b[(base+j+1)*n+ni],
                            b[(base+j+2)*n+ni], b[(base+j+3)*n+ni]
                        };
                        bv = vld1q_f32(vals);
                    }
                    acc = vmlaq_f32(acc, av, bv);
                }
                /* tail (Q8_0 bs=32, Q4_K/Q6_K bs=256: all multiples of 4) */
                for (; j < bs; j++)
                    acc = vaddq_f32(acc,
                          vdupq_n_f32(tmp[j] * b[(base+j)*n+ni]));
            }

            {
                float32x2_t lo = vget_low_f32(acc);
                float32x2_t hi = vget_high_f32(acc);
                float32x2_t s  = vadd_f32(lo, hi);
                out[mi * n + ni] = vget_lane_f32(vpadd_f32(s, s), 0);
            }
        }
    }
}

#endif /* ARM_NEON */

/* =========================================================================
 * cmol_kernels_select
 *
 * Detects the CPU at run time and returns the fastest available kernel.
 * Called once from cmol_load().
 * ====================================================================== */

cmol_kernels_t cmol_kernels_select(void) {
    cmol_kernels_t kn;
    cmol_cpu_t     cpu = cmol_detect_cpu();

#if defined(__x86_64__) || defined(_M_X64)
    if (cpu.avx512f) {
        kn.matmul = cmol_matmul_avx512;
        kn.name   = "avx512";
        return kn;
    }
    if (cpu.avx2) {
        kn.matmul = cmol_matmul_avx2;
        kn.name   = "avx2";
        return kn;
    }
#endif

#if defined(__ARM_NEON) || defined(__aarch64__)
    if (cpu.neon) {
        kn.matmul = cmol_matmul_neon;
        kn.name   = "neon";
        return kn;
    }
#endif

    (void)cpu;
    kn.matmul = cmol_matmul_scalar;
    kn.name   = "scalar";
    return kn;
}

/* ===== src/model.c ===== */
/*
 * model.c — transformer forward pass
 * Included by src/cmol.c (unity build); do not compile standalone.
 *
 * Implements:
 *   cmol_rms_norm    — in-place RMSNorm
 *   cmol_softmax     — numerically stable softmax
 *   cmol_swiglu      — SiLU(gate) * up  (SwiGLU activation)
 *   cmol_rope_apply  — rotary position encoding (LLaMA NORM style: adjacent pairs)
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
 * cmol_rope_apply — LLaMA NORM-style RoPE (GGML_ROPE_TYPE_NORM)
 *
 * Rotates adjacent dimension pairs within each head:
 *   For each head h and pair index i in [0, head_dim/2):
 *     θ_i = pos / freq_base^(2i / head_dim)
 *     q[h][2i]   ← q[h][2i]   * cos(θ) − q[h][2i+1] * sin(θ)
 *     q[h][2i+1] ← q[h][2i]   * sin(θ) + q[h][2i+1] * cos(θ)
 *   (same for k, up to n_kv_heads)
 *
 * Note: LLaMA uses NORM style (adjacent pairs), not NEOX style (split halves).
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
            float q0 = qh[2 * i], q1 = qh[2 * i + 1];
            qh[2 * i]     = q0 * cs - q1 * sn;
            qh[2 * i + 1] = q0 * sn + q1 * cs;
        }
    }

    for (h = 0; h < n_kv_heads; h++) {
        float *kh = k + h * head_dim;
        for (i = 0; i < half; i++) {
            float theta = (float)pos /
                          powf(freq_base, (float)(2 * i) / (float)head_dim);
            float cs = cosf(theta), sn = sinf(theta);
            float k0 = kh[2 * i], k1 = kh[2 * i + 1];
            kh[2 * i]     = k0 * cs - k1 * sn;
            kh[2 * i + 1] = k0 * sn + k1 * cs;
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

/* ===== src/attn.c ===== */
/*
 * attn.c — grouped-query attention with KV cache
 * Included by src/cmol.c (unity build); do not compile standalone.
 *
 * Depends on helpers defined in model.c (included before this file in
 * the unity build):
 *   cmol__find_blk, cmol__row_bytes
 *   cmol_rms_norm, cmol_rope_apply, cmol_softmax
 */


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

/* ===== src/sampler.c ===== */
/*
 * sampler.c — token sampling (greedy / temperature / top-k / top-p)
 * Included by src/cmol.c (unity build); do not compile standalone.
 *
 * Pipeline for cmol_sample():
 *   temperature == 0  →  greedy argmax, done
 *   scale logits by 1/temperature
 *   softmax → probabilities
 *   top-k   →  keep up to k tokens via min-heap (O(V log k))
 *   sort k tokens descending
 *   top-p   →  find smallest nucleus with cumulative prob ≥ top_p
 *   sample  →  CDF walk over the nucleus
 *
 * When both top-k and top-p are disabled the full vocabulary is sampled
 * via a direct CDF walk over all V probabilities.
 *
 * RNG:   xoshiro256** — 64-bit, period 2^256 - 1, excellent quality.
 * Seed:  splitmix64 expansion of a 32-bit seed; seed=0 uses time(NULL).
 */

#include <math.h>     /* expf               */
#include <string.h>   /* memcpy             */
#include <stdlib.h>   /* qsort              */
#include <time.h>     /* time()             */
#include <stdint.h>   /* uint64_t, uintptr_t */

/* =========================================================================
 * xoshiro256** — https://prng.di.unimi.it/xoshiro256starstar.c
 * ====================================================================== */

static inline uint64_t rotl64(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

uint64_t cmol_rng_next(uint64_t s[4]) {
    uint64_t result = rotl64(s[1] * 5u, 7) * 9u;
    uint64_t t      = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3]  = rotl64(s[3], 45);

    return result;
}

/* splitmix64 — one-way expansion of a 64-bit value */
static uint64_t splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

void cmol_rng_seed(uint64_t state[4], unsigned int seed) {
    uint64_t x;
    if (seed == 0) {
        /* Non-deterministic: XOR time, a counter, and the state pointer. */
        static uint64_t g_counter = 0;
        x  = (uint64_t)time(NULL);
        x ^= ++g_counter * 6364136223846793005ULL;
        x ^= (uint64_t)(uintptr_t)state;
        if (!x) x = 0xDEADBEEFCAFEBABEULL;
    } else {
        x = (uint64_t)seed;
    }
    state[0] = splitmix64(&x);
    state[1] = splitmix64(&x);
    state[2] = splitmix64(&x);
    state[3] = splitmix64(&x);
}

/* Random float in [0, 1) using the IEEE 754 mantissa trick. */
static float rng_f32(uint64_t state[4]) {
    uint64_t r    = cmol_rng_next(state);
    uint32_t bits = 0x3F800000u | (uint32_t)(r >> 41); /* exp=127, 23 random mantissa bits */
    float    f;
    memcpy(&f, &bits, sizeof f);
    return f - 1.0f;
}

/* =========================================================================
 * Internal softmax (static — avoids cross-module dependency on model.c)
 * ====================================================================== */

static void smpl_softmax(float *x, int n) {
    float mx = x[0], sm = 0.0f;
    int i;
    for (i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    for (i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sm += x[i]; }
    sm = 1.0f / sm;
    for (i = 0; i < n; i++) x[i] *= sm;
}

/* =========================================================================
 * (probability, token_id) pair used by the top-k heap
 * ====================================================================== */

typedef struct { float val; int32_t idx; } token_prob_t;

/* ---- Min-heap helpers (root = smallest val) --------------------------- */

static void heap_sift_down(token_prob_t *h, int i, int n) {
    for (;;) {
        int s = i, l = 2*i+1, r = 2*i+2;
        if (l < n && h[l].val < h[s].val) s = l;
        if (r < n && h[r].val < h[s].val) s = r;
        if (s == i) break;
        token_prob_t tmp = h[i]; h[i] = h[s]; h[s] = tmp;
        i = s;
    }
}

/* Push a new element onto the heap (heap must have space). */
static void heap_push(token_prob_t *h, int *n, token_prob_t v) {
    int i = (*n)++;
    h[i] = v;
    while (i > 0) {
        int p = (i - 1) / 2;
        if (h[p].val <= h[i].val) break;
        token_prob_t tmp = h[i]; h[i] = h[p]; h[p] = tmp;
        i = p;
    }
}

/* Replace the root (current minimum) and restore heap. */
static void heap_replace_root(token_prob_t *h, int n, token_prob_t v) {
    h[0] = v;
    heap_sift_down(h, 0, n);
}

/* qsort comparator: descending probability */
static int cmp_prob_desc(const void *a, const void *b) {
    float va = ((const token_prob_t *)a)->val;
    float vb = ((const token_prob_t *)b)->val;
    return (va < vb) ? 1 : (va > vb) ? -1 : 0;
}

/* =========================================================================
 * Top-k selection via a min-heap of size k
 *
 * Scans probs[0..V-1] and places the k largest into out[0..k-1]
 * (in min-heap order, not sorted).  Returns actual count placed.
 * ====================================================================== */

#define CMOL_TOPK_BUF 512   /* maximum supported top-k value */

static int topk_select(const float *probs, int V, int k,
                        token_prob_t *out) {
    int n = 0, i;
    if (k > CMOL_TOPK_BUF) k = CMOL_TOPK_BUF;

    for (i = 0; i < V; i++) {
        if (n < k) {
            heap_push(out, &n, (token_prob_t){probs[i], i});
        } else if (probs[i] > out[0].val) {
            heap_replace_root(out, k, (token_prob_t){probs[i], i});
        }
    }
    return n;
}

/* =========================================================================
 * cmol_apply_repeat_penalty
 * ====================================================================== */

void cmol_apply_repeat_penalty(float *logits, int vocab_size,
                                const int32_t *tokens, int n_tokens,
                                float penalty) {
    int i;
    if (penalty <= 1.0f || !logits || !tokens || n_tokens <= 0) return;
    for (i = 0; i < n_tokens; i++) {
        int32_t id = tokens[i];
        if (id < 0 || id >= vocab_size) continue;
        if (logits[id] > 0.0f)
            logits[id] /= penalty;
        else
            logits[id] *= penalty;
    }
}

/* =========================================================================
 * cmol_sample
 * ====================================================================== */

int32_t cmol_sample(float                   *logits,
                     int                      vocab_size,
                     const cmol_gen_params_t *params,
                     uint64_t                *rng_state) {

    int i;
    if (!logits || vocab_size <= 0) return 0;

    /* ── Greedy (temperature == 0 or no params) ──────────────────────── */
    if (!params || params->temperature == 0.0f) {
        int32_t best = 0;
        for (i = 1; i < vocab_size; i++)
            if (logits[i] > logits[best]) best = i;
        return best;
    }

    /* ── Temperature scaling ──────────────────────────────────────────── */
    {
        float inv_t = 1.0f / params->temperature;
        for (i = 0; i < vocab_size; i++) logits[i] *= inv_t;
    }

    /* ── Softmax → probabilities (in-place) ────────────────────────────── */
    smpl_softmax(logits, vocab_size);

    /* ── Decide whether to use the heap path ──────────────────────────── */
    int use_topk = (params->top_k > 0 && params->top_k < vocab_size);
    int use_topp = (params->top_p > 0.0f && params->top_p < 1.0f);

    if (!use_topk && !use_topp) {
        /* ── Plain categorical sampling: CDF walk over all probs ─────── */
        if (!rng_state) {
            /* No RNG provided — return the argmax of probabilities */
            int32_t best = 0;
            for (i = 1; i < vocab_size; i++)
                if (logits[i] > logits[best]) best = i;
            return best;
        }
        float r = rng_f32(rng_state), acc = 0.0f;
        for (i = 0; i < vocab_size - 1; i++) {
            acc += logits[i];
            if (r < acc) return (int32_t)i;
        }
        return (int32_t)(vocab_size - 1);
    }

    /* ── Top-k selection ─────────────────────────────────────────────── */
    token_prob_t buf[CMOL_TOPK_BUF];
    int k = use_topk ? params->top_k : vocab_size; /* cap applied inside */
    int n = topk_select(logits, vocab_size, k, buf);

    /* Sort descending (required for nucleus scan) */
    qsort(buf, (size_t)n, sizeof(token_prob_t), cmp_prob_desc);

    /* ── Top-p nucleus ───────────────────────────────────────────────── */
    int nucleus = n;
    if (use_topp) {
        float cum = 0.0f;
        for (i = 0; i < n; i++) {
            cum += buf[i].val;
            if (cum >= params->top_p) { nucleus = i + 1; break; }
        }
        /* nucleus stays n if we never exceeded top_p (sum of top-k < top_p) */
    }

    /* ── Re-normalise and sample ─────────────────────────────────────── */
    float sum = 0.0f;
    for (i = 0; i < nucleus; i++) sum += buf[i].val;

    /* Guard: no RNG or degenerate distribution → deterministic top-1 */
    if (!rng_state || sum <= 0.0f) return buf[0].idx;

    float r = rng_f32(rng_state) * sum, acc = 0.0f;
    for (i = 0; i < nucleus; i++) {
        acc += buf[i].val;
        if (r < acc) return buf[i].idx;
    }
    return buf[nucleus - 1].idx;   /* rounding/fp safety */
}

/* ===== src/api.c ===== */
/*
 * api.c — public API implementation
 * Included by src/cmol.c (unity build); do not compile standalone.
 */








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

int cmol_format_chatml(const char *system, const char *user,
                        char *buf, size_t buf_cap) {
    int n;
    if (!user) return (int)CMOL_ERR_ARGS;

    /* NULL  → omit system turn (same as "")
     * ""    → omit system turn
     * other → use verbatim                  */
    const char *sys = (system != NULL) ? system : "";

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

#endif /* CMOL_IMPLEMENTATION */

#endif /* CMOL_H */
