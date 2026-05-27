/*
 * cmol_internal.h — internal types for libcmol
 *
 * Full struct definitions for the opaque public handles (cmol_model_t,
 * cmol_session_t) plus all shared internal types.
 *
 * Internal header — not part of the public API.
 * Include AFTER cmol.h and platform.h.
 */

#ifndef CMOL_INTERNAL_H
#define CMOL_INTERNAL_H

#include <stdint.h>
#include <stddef.h>
#include <pthread.h>

#include "../include/cmol.h"
#include "platform.h"

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
    int     n_heads;        /* query heads                                */
    int     n_kv_heads;     /* key/value heads (< n_heads when GQA)       */
    int     d_model;        /* embedding dimension                        */
    int     d_head;         /* d_model / n_heads                          */
    int     d_ffn;          /* feed-forward inner dimension               */
    int     vocab_size;
    int     model_max_ctx;  /* max context as reported in the GGUF        */
    float   rope_freq_base;
    float   rms_norm_eps;
    int32_t bos_token_id;
    int32_t eos_token_id;
    int     tie_embeddings; /* 1 if lm_head shares weights with token_embd */
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

typedef struct {
    const char       **vocab;       /* [vocab_size] strings in arena      */
    float             *scores;      /* [vocab_size] BPE merge scores      */
    uint8_t           *token_type;  /* [vocab_size] cmol_token_type_t     */
    int                vocab_size;

    /* BPE merge rules, sorted by priority (score descending).
     * merge_left[i] + merge_right[i] -> merged token.
     * Stored in arena; n_merges == vocab_size - 256 (roughly). */
    int32_t           *merge_left;
    int32_t           *merge_right;
    int                n_merges;

    int32_t bos_id;
    int32_t eos_id;
    int32_t unk_id;
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

#endif /* CMOL_INTERNAL_H */
