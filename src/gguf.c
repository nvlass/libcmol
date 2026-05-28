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

#include "gguf.h"
#include "arena.h"
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
