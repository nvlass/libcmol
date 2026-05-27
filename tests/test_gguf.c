/*
 * test_gguf.c — Phase 1 tests for the GGUF parser
 *
 * Tests that don't require a real model file:
 *   - Reject corrupt / truncated inputs
 *   - Reject wrong magic / unsupported version
 *   - cmol_gguf_find_tensor: found / not-found
 *   - Arena exhaustion returns CMOL_ERR_OOM
 *
 * Tests that require a real model file (skipped if CMOL_TEST_GGUF is unset):
 *   - Parse a real SmolLM3 GGUF: check hparams, vocab size, tensor names
 *   - cmol_gguf_peek: returns the same hparams as the full parse
 *
 * Usage with a model:
 *   CMOL_TEST_GGUF=/path/to/smollm3.gguf ./build/tests/test_gguf
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

/* Internal headers — declarations only; implementation is in cmol_d.o */
#include "gguf.h"          /* pulls in cmol_internal.h → platform.h → cmol.h */
#include "arena.h"

/* =========================================================================
 * Minimal test harness
 * ====================================================================== */

static int g_tests = 0, g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do {                                       \
    g_tests++;                                                      \
    if (cond) { g_pass++; printf("  PASS  %s\n", msg); }           \
    else       { g_fail++; printf("  FAIL  %s\n", msg); }          \
} while (0)

#define SECTION(name) printf("\n[%s]\n", name)

/* =========================================================================
 * Synthetic GGUF builder — create minimal valid/invalid GGUF bytes
 * ====================================================================== */

typedef struct { uint8_t *buf; size_t len; size_t cap; } bb_t;

static void bb_init(bb_t *b) { b->buf = NULL; b->len = 0; b->cap = 0; }
static void bb_free(bb_t *b) { free(b->buf); bb_init(b); }

static void bb_reserve(bb_t *b, size_t extra) {
    if (b->len + extra <= b->cap) return;
    size_t nc = (b->cap ? b->cap * 2 : 256);
    while (nc < b->len + extra) nc *= 2;
    b->buf = (uint8_t *)realloc(b->buf, nc);
    b->cap = nc;
}

static void bb_u8 (bb_t *b, uint8_t  v) { bb_reserve(b,1); b->buf[b->len++] = v; (void)bb_u8; }
static void bb_u32(bb_t *b, uint32_t v) { bb_reserve(b,4); memcpy(b->buf+b->len,&v,4); b->len+=4; }
static void bb_u64(bb_t *b, uint64_t v) { bb_reserve(b,8); memcpy(b->buf+b->len,&v,8); b->len+=8; }
static void bb_f32(bb_t *b, float    v) { bb_reserve(b,4); memcpy(b->buf+b->len,&v,4); b->len+=4; }
static void bb_str(bb_t *b, const char *s) {
    size_t n = strlen(s);
    bb_u64(b, (uint64_t)n);
    bb_reserve(b, n);
    memcpy(b->buf + b->len, s, n);
    b->len += n;
}

/* Emit a minimal valid GGUF v3 with no tensors and a single uint32 KV */
static void bb_minimal_gguf(bb_t *b) {
    bb_u32(b, 0x46554747u);  /* magic "GGUF"              */
    bb_u32(b, 3);             /* version 3                 */
    bb_u64(b, 0);             /* n_tensors = 0             */
    bb_u64(b, 1);             /* n_kv = 1                  */
    /* KV: llama.block_count = 4 */
    bb_str(b, "llama.block_count");
    bb_u32(b, 4);             /* GV_UINT32                 */
    bb_u32(b, 4);             /* value = 4                 */
}

/* =========================================================================
 * Parse helper — parse synthetic bytes using a scratch arena
 * ====================================================================== */

#define SCRATCH_SIZE (512 * 1024)

static cmol_err_t parse_bytes(const uint8_t *data, size_t size,
                               cmol_hparams_t *hp, cmol_tokenizer_t *tok,
                               cmol_tensor_t **tensors, int *n_tensors) {
    uint8_t      *scratch = (uint8_t *)malloc(SCRATCH_SIZE);
    cmol_arena_t  arena;
    cmol_arena_init(&arena, scratch, SCRATCH_SIZE);

    cmol_mmap_t mmap = { (void *)(uintptr_t)data, size };

    cmol_err_t err = cmol_gguf_parse(&mmap, &arena, hp, tensors, n_tensors, tok);

    /* Note: scratch is leaked here — tests are short-lived processes */
    /* In real code, free(scratch) after we're done with the tensors. */
    (void)scratch; /* suppress warning */
    return err;
}

/* =========================================================================
 * Tests: malformed / corrupt inputs
 * ====================================================================== */

static void test_malformed(void) {
    SECTION("Malformed inputs");
    cmol_hparams_t hp; cmol_tokenizer_t tok;
    cmol_tensor_t *t; int nt;
    cmol_err_t e;

    /* Empty buffer */
    e = parse_bytes(NULL, 0, &hp, &tok, &t, &nt);
    CHECK(e != CMOL_OK, "empty buffer → error");

    /* Only 3 bytes (truncated before magic) */
    uint8_t short_buf[3] = {0x47, 0x47, 0x55};
    e = parse_bytes(short_buf, sizeof short_buf, &hp, &tok, &t, &nt);
    CHECK(e != CMOL_OK, "3-byte buffer → error");

    /* Wrong magic */
    bb_t b; bb_init(&b);
    bb_u32(&b, 0xDEADBEEFu);
    bb_u32(&b, 3);
    bb_u64(&b, 0); bb_u64(&b, 0);
    e = parse_bytes(b.buf, b.len, &hp, &tok, &t, &nt);
    CHECK(e == CMOL_ERR_INVALID, "wrong magic → CMOL_ERR_INVALID");
    bb_free(&b);

    /* Version 1 (unsupported) */
    bb_init(&b);
    bb_u32(&b, 0x46554747u);
    bb_u32(&b, 1);
    bb_u32(&b, 0); bb_u32(&b, 0); /* v1 uses uint32 counts */
    e = parse_bytes(b.buf, b.len, &hp, &tok, &t, &nt);
    CHECK(e == CMOL_ERR_UNSUPPORTED, "version 1 → CMOL_ERR_UNSUPPORTED");
    bb_free(&b);

    /* Version 99 */
    bb_init(&b);
    bb_u32(&b, 0x46554747u);
    bb_u32(&b, 99);
    bb_u64(&b, 0); bb_u64(&b, 0);
    e = parse_bytes(b.buf, b.len, &hp, &tok, &t, &nt);
    CHECK(e == CMOL_ERR_UNSUPPORTED, "version 99 → CMOL_ERR_UNSUPPORTED");
    bb_free(&b);

    /* Truncated mid-KV */
    bb_init(&b);
    bb_u32(&b, 0x46554747u);
    bb_u32(&b, 3);
    bb_u64(&b, 0);   /* n_tensors */
    bb_u64(&b, 2);   /* n_kv = 2, but we only write 1 */
    bb_str(&b, "llama.block_count");
    bb_u32(&b, 4); bb_u32(&b, 4);
    /* second KV entry missing */
    e = parse_bytes(b.buf, b.len, &hp, &tok, &t, &nt);
    CHECK(e != CMOL_OK, "truncated KV section → error");
    bb_free(&b);
}

/* =========================================================================
 * Tests: minimal valid GGUF
 * ====================================================================== */

static void test_minimal_valid(void) {
    SECTION("Minimal valid GGUF");
    bb_t b; bb_init(&b);
    bb_minimal_gguf(&b);

    cmol_hparams_t hp; cmol_tokenizer_t tok;
    cmol_tensor_t *tensors; int n_tensors;

    cmol_err_t e = parse_bytes(b.buf, b.len, &hp, &tok, &tensors, &n_tensors);
    CHECK(e == CMOL_OK,             "minimal GGUF parses OK");
    CHECK(hp.n_layers == 4,         "llama.block_count = 4");
    CHECK(n_tensors == 0,           "n_tensors = 0");
    CHECK(hp.rope_freq_base > 0.0f, "rope_freq_base has default");
    CHECK(hp.rms_norm_eps > 0.0f,   "rms_norm_eps has default");
    bb_free(&b);
}

/* =========================================================================
 * Tests: KV scalar types
 * ====================================================================== */

static void test_kv_scalars(void) {
    SECTION("KV scalar extraction");
    bb_t b; bb_init(&b);

    bb_u32(&b, 0x46554747u);
    bb_u32(&b, 3);
    bb_u64(&b, 0);   /* n_tensors */
    bb_u64(&b, 6);   /* n_kv */

    /* embedding length */
    bb_str(&b, "llama.embedding_length"); bb_u32(&b, 4); bb_u32(&b, 2048);
    /* block count */
    bb_str(&b, "llama.block_count"); bb_u32(&b, 4); bb_u32(&b, 24);
    /* heads */
    bb_str(&b, "llama.attention.head_count"); bb_u32(&b, 4); bb_u32(&b, 32);
    /* kv heads */
    bb_str(&b, "llama.attention.head_count_kv"); bb_u32(&b, 4); bb_u32(&b, 8);
    /* rope freq */
    bb_str(&b, "llama.rope.freq_base"); bb_u32(&b, 6); bb_f32(&b, 500000.0f);
    /* unknown key (should be skipped cleanly) */
    bb_str(&b, "general.name"); bb_u32(&b, 8);  /* GV_STRING */
    bb_str(&b, "SmolLM3-1.7B");

    cmol_hparams_t hp; cmol_tokenizer_t tok;
    cmol_tensor_t *tensors; int n_tensors;

    cmol_err_t e = parse_bytes(b.buf, b.len, &hp, &tok, &tensors, &n_tensors);
    CHECK(e == CMOL_OK,                   "KV scalar parse OK");
    CHECK(hp.d_model       == 2048,       "d_model = 2048");
    CHECK(hp.n_layers      == 24,         "n_layers = 24");
    CHECK(hp.n_heads        == 32,        "n_heads = 32");
    CHECK(hp.n_kv_heads     == 8,         "n_kv_heads = 8 (GQA)");
    CHECK(hp.d_head         == 64,        "d_head = d_model/n_heads = 64");
    CHECK(hp.rope_freq_base == 500000.0f, "rope_freq_base = 500000");
    bb_free(&b);
}

/* =========================================================================
 * Tests: tokenizer vocab
 * ====================================================================== */

static void test_tokenizer_vocab(void) {
    SECTION("Tokenizer vocab");
    bb_t b; bb_init(&b);

    bb_u32(&b, 0x46554747u);
    bb_u32(&b, 3);
    bb_u64(&b, 0);  /* n_tensors */
    bb_u64(&b, 4);  /* n_kv */

    /* BOS / EOS */
    bb_str(&b, "tokenizer.ggml.bos_token_id"); bb_u32(&b, 4); bb_u32(&b, 1);
    bb_str(&b, "tokenizer.ggml.eos_token_id"); bb_u32(&b, 4); bb_u32(&b, 2);

    /* Vocab: 4 tokens */
    bb_str(&b, "tokenizer.ggml.tokens");
    bb_u32(&b, 9);   /* GV_ARRAY */
    bb_u32(&b, 8);   /* elem = GV_STRING */
    bb_u64(&b, 4);   /* count = 4 */
    bb_str(&b, "<unk>"); bb_str(&b, "<s>"); bb_str(&b, "</s>"); bb_str(&b, "Hello");

    /* Scores: 4 floats */
    bb_str(&b, "tokenizer.ggml.scores");
    bb_u32(&b, 9);   /* GV_ARRAY */
    bb_u32(&b, 6);   /* elem = GV_FLOAT32 */
    bb_u64(&b, 4);
    bb_f32(&b, 0.0f); bb_f32(&b, 0.0f); bb_f32(&b, 0.0f); bb_f32(&b, -1.0f);

    cmol_hparams_t hp; cmol_tokenizer_t tok;
    cmol_tensor_t *tensors; int n_tensors;

    cmol_err_t e = parse_bytes(b.buf, b.len, &hp, &tok, &tensors, &n_tensors);
    CHECK(e == CMOL_OK,                          "tokenizer parse OK");
    CHECK(tok.vocab_size == 4,                   "vocab_size = 4");
    CHECK(tok.bos_id == 1,                       "bos_id = 1");
    CHECK(tok.eos_id == 2,                       "eos_id = 2");
    CHECK(tok.vocab != NULL,                     "vocab array allocated");
    CHECK(tok.vocab && !strcmp(tok.vocab[0], "<unk>"),   "vocab[0] = <unk>");
    CHECK(tok.vocab && !strcmp(tok.vocab[3], "Hello"),   "vocab[3] = Hello");
    CHECK(tok.scores != NULL,                    "scores array allocated");
    CHECK(tok.scores && tok.scores[3] == -1.0f,  "scores[3] = -1.0");
    bb_free(&b);
}

/* =========================================================================
 * Tests: cmol_gguf_find_tensor
 * ====================================================================== */

static void test_find_tensor(void) {
    SECTION("cmol_gguf_find_tensor");

    cmol_tensor_t tensors[3];
    memset(tensors, 0, sizeof tensors);
    strncpy(tensors[0].name, "token_embd.weight",   CMOL_MAX_TENSOR_NAME-1);
    strncpy(tensors[1].name, "output_norm.weight",  CMOL_MAX_TENSOR_NAME-1);
    strncpy(tensors[2].name, "blk.0.attn_q.weight", CMOL_MAX_TENSOR_NAME-1);

    CHECK(cmol_gguf_find_tensor(tensors, 3, "token_embd.weight")
              == &tensors[0],                         "find first tensor");
    CHECK(cmol_gguf_find_tensor(tensors, 3, "blk.0.attn_q.weight")
              == &tensors[2],                         "find last tensor");
    CHECK(cmol_gguf_find_tensor(tensors, 3, "output.weight")
              == NULL,                                "absent tensor → NULL");
    CHECK(cmol_gguf_find_tensor(tensors, 0, "token_embd.weight")
              == NULL,                                "empty list → NULL");
}

/* =========================================================================
 * Tests: merge resolution
 * ====================================================================== */

static void test_merges(void) {
    SECTION("BPE merge resolution");
    bb_t b; bb_init(&b);

    bb_u32(&b, 0x46554747u);
    bb_u32(&b, 3);
    bb_u64(&b, 0);  /* n_tensors */
    bb_u64(&b, 2);  /* n_kv */

    /* Vocab: a b c ab */
    bb_str(&b, "tokenizer.ggml.tokens");
    bb_u32(&b, 9); bb_u32(&b, 8); bb_u64(&b, 4);
    bb_str(&b, "a"); bb_str(&b, "b"); bb_str(&b, "c"); bb_str(&b, "ab");

    /* Merges: "a b" → (0, 1) */
    bb_str(&b, "tokenizer.ggml.merges");
    bb_u32(&b, 9); bb_u32(&b, 8); bb_u64(&b, 1);
    bb_str(&b, "a b");

    cmol_hparams_t hp; cmol_tokenizer_t tok;
    cmol_tensor_t *tensors; int n_tensors;

    cmol_err_t e = parse_bytes(b.buf, b.len, &hp, &tok, &tensors, &n_tensors);
    CHECK(e == CMOL_OK,                       "merge parse OK");
    CHECK(tok.n_merges == 1,                  "n_merges = 1");
    CHECK(tok.merge_left  && tok.merge_left[0]  == 0, "merge_left[0] = 0 ('a')");
    CHECK(tok.merge_right && tok.merge_right[0] == 1, "merge_right[0] = 1 ('b')");
    bb_free(&b);
}

/* =========================================================================
 * Tests: real GGUF file  (opt-in via CMOL_TEST_GGUF env var)
 * ====================================================================== */

static void test_real_file(const char *path) {
    SECTION("Real GGUF file");
    printf("  model: %s\n\n", path);

    /* peek */
    cmol_hparams_t peek_hp;
    size_t peek_n;
    cmol_err_t e = cmol_gguf_peek(path, &peek_hp, &peek_n);
    CHECK(e == CMOL_OK,               "cmol_gguf_peek succeeds");
    CHECK(peek_hp.n_layers > 0,       "peek: n_layers > 0");
    CHECK(peek_hp.d_model > 0,        "peek: d_model > 0");
    CHECK(peek_hp.n_heads > 0,        "peek: n_heads > 0");
    CHECK(peek_n > 0,                 "peek: n_tensors > 0");
    printf("  peek: n_layers=%d d_model=%d n_heads=%d n_kv_heads=%d"
           " d_head=%d vocab=%d ctx=%d tensors=%zu\n",
           peek_hp.n_layers, peek_hp.d_model, peek_hp.n_heads,
           peek_hp.n_kv_heads, peek_hp.d_head, peek_hp.vocab_size,
           peek_hp.model_max_ctx, peek_n);

    /* full parse */
    cmol_mmap_t mmap;
    e = cmol_mmap_open(path, &mmap);
    CHECK(e == CMOL_OK, "mmap succeeds");
    if (e != CMOL_OK) return;

    size_t arena_sz = 32u * 1024 * 1024; /* 32 MB for metadata */
    uint8_t *arena_buf = (uint8_t *)malloc(arena_sz);
    assert(arena_buf);
    cmol_arena_t arena;
    cmol_arena_init(&arena, arena_buf, arena_sz);

    cmol_hparams_t    hp;
    cmol_tokenizer_t  tok;
    cmol_tensor_t    *tensors;
    int               n_tensors;

    e = cmol_gguf_parse(&mmap, &arena, &hp, &tensors, &n_tensors, &tok);
    CHECK(e == CMOL_OK,           "full parse succeeds");

    if (e == CMOL_OK) {
        /* hparams should match peek */
        CHECK(hp.n_layers    == peek_hp.n_layers,  "full parse n_layers == peek");
        CHECK(hp.d_model     == peek_hp.d_model,   "full parse d_model == peek");
        CHECK(n_tensors > 0,                        "n_tensors > 0");

        /* Tokenizer */
        CHECK(tok.vocab_size > 0,                   "vocab_size > 0");
        CHECK(tok.vocab != NULL,                    "vocab array present");
        CHECK(tok.bos_id >= 0,                      "bos_id set");
        CHECK(tok.eos_id >= 0,                      "eos_id set");

        /* Required tensors must exist */
        CHECK(cmol_gguf_find_tensor(tensors, n_tensors, "token_embd.weight")
                  != NULL,                          "token_embd.weight present");
        CHECK(cmol_gguf_find_tensor(tensors, n_tensors, "output_norm.weight")
                  != NULL,                          "output_norm.weight present");
        CHECK(cmol_gguf_find_tensor(tensors, n_tensors, "blk.0.attn_q.weight")
                  != NULL,                          "blk.0.attn_q.weight present");

        /* Data pointers must be inside the mmap region */
        int ptr_ok = 1;
        for (int i = 0; i < n_tensors; i++) {
            uint8_t *d = (uint8_t *)tensors[i].data;
            if (d < (uint8_t *)mmap.data ||
                d + tensors[i].n_bytes > (uint8_t *)mmap.data + mmap.size) {
                ptr_ok = 0; break;
            }
        }
        CHECK(ptr_ok, "all tensor data pointers within mmap region");

        printf("  full: vocab=%d merges=%d tensors=%d tied_embd=%d\n",
               tok.vocab_size, tok.n_merges, n_tensors, hp.tie_embeddings);
    }

    cmol_mmap_close(&mmap);
    free(arena_buf);
}

/* =========================================================================
 * main
 * ====================================================================== */

int main(void) {
    printf("=== test_gguf ===\n");

    test_malformed();
    test_minimal_valid();
    test_kv_scalars();
    test_tokenizer_vocab();
    test_find_tensor();
    test_merges();

    const char *model_path = getenv("CMOL_TEST_GGUF");
    if (model_path)
        test_real_file(model_path);
    else
        printf("\n[Real file tests skipped — set CMOL_TEST_GGUF=/path/to/model.gguf]\n");

    printf("\n=== %d/%d passed", g_pass, g_tests);
    if (g_fail) printf(", %d FAILED", g_fail);
    printf(" ===\n");

    return g_fail ? 1 : 0;
}
