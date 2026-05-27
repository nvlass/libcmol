/*
 * test_tokenizer.c — Phase 3 tests for the BPE tokenizer
 *
 * Tests use a self-contained 21-token vocabulary with 10 merge rules,
 * exercising every code path without requiring a real GGUF model file.
 *
 * Vocabulary (indices):
 *   0  <unk>          3  ▁         6  l       9   r       12  ▁He
 *   1  <s>  (BOS)     4  H         7  o       10  d       13  ▁Hel
 *   2  </s> (EOS)     5  e         8  w       11  ▁H      14  ▁Hell
 *                                                          15  ▁Hello
 *                                             16  ▁w      19  ▁worl
 *                                             17  ▁wo     20  ▁world
 *                                             18  ▁wor
 *
 * Merge rules (priority = index, 0 = highest):
 *   0: (▁,H)→▁H      1: (▁H,e)→▁He     2: (▁He,l)→▁Hel
 *   3: (▁Hel,l)→▁Hell 4: (▁Hell,o)→▁Hello
 *   5: (▁,w)→▁w       6: (▁w,o)→▁wo     7: (▁wo,r)→▁wor
 *   8: (▁wor,l)→▁worl 9: (▁worl,d)→▁world
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

#include "tokenizer.h"  /* pulls in cmol_internal.h → cmol.h */
#include "arena.h"

/* =========================================================================
 * Minimal test harness  (same pattern as test_gguf.c)
 * ====================================================================== */

static int g_tests = 0, g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do {                                           \
    g_tests++;                                                          \
    if (cond) { g_pass++; printf("  PASS  %s\n", msg); }              \
    else       { g_fail++; printf("  FAIL  %s\n", msg); }             \
} while (0)

#define SECTION(name) printf("\n[%s]\n", name)

/* =========================================================================
 * Test vocabulary (file-scope so decoded pointers remain valid)
 * ====================================================================== */

/* ▁ (U+2581) in UTF-8: 0xE2 0x96 0x81 */
#define SP "\xe2\x96\x81"

static const char *g_vocab[] = {
    /* 0 */  "<unk>",
    /* 1 */  "<s>",
    /* 2 */  "</s>",
    /* 3 */  SP,
    /* 4 */  "H",
    /* 5 */  "e",
    /* 6 */  "l",
    /* 7 */  "o",
    /* 8 */  "w",
    /* 9 */  "r",
    /* 10 */ "d",
    /* 11 */ SP "H",
    /* 12 */ SP "He",
    /* 13 */ SP "Hel",
    /* 14 */ SP "Hell",
    /* 15 */ SP "Hello",
    /* 16 */ SP "w",
    /* 17 */ SP "wo",
    /* 18 */ SP "wor",
    /* 19 */ SP "worl",
    /* 20 */ SP "world",
};
#define VOCAB_SIZE 21

/* merge_left[i] + merge_right[i] → result token */
static int32_t g_ml[] = { 3, 11, 12, 13, 14,  3, 16, 17, 18, 19 };
static int32_t g_mr[] = { 4,  5,  6,  6,  7,  8,  7,  9,  6, 10 };
#define N_MERGES 10

/* =========================================================================
 * Setup helpers
 * ====================================================================== */

#define ARENA_SZ (256 * 1024)  /* 256 KB — plenty for these tests */

/* Populate a tokenizer with the test vocabulary.
 * The arena is used by cmol_tokenizer_build(); pass a fresh arena each time
 * if you want independent test isolation. */
static void setup_tok(cmol_tokenizer_t *tok) {
    memset(tok, 0, sizeof *tok);
    tok->vocab       = g_vocab;
    tok->vocab_size  = VOCAB_SIZE;
    tok->merge_left  = g_ml;
    tok->merge_right = g_mr;
    tok->n_merges    = N_MERGES;
    tok->bos_id      = 1;
    tok->eos_id      = 2;
    tok->unk_id      = 0;
    tok->tok_model   = CMOL_TOK_LLAMA;
    /* scores / token_type left NULL — optional fields */
}

/* =========================================================================
 * Tests: build phase
 * ====================================================================== */

static void test_build(void) {
    SECTION("cmol_tokenizer_build");

    uint8_t *abuf = (uint8_t *)malloc(ARENA_SZ);
    cmol_arena_t arena;
    cmol_arena_init(&arena, abuf, ARENA_SZ);

    cmol_tokenizer_t tok;
    setup_tok(&tok);

    cmol_err_t e = cmol_tokenizer_build(&tok, &arena);
    CHECK(e == CMOL_OK,                   "build succeeds");
    CHECK(tok.vocab_sort_idx  != NULL,    "vocab_sort_idx allocated");
    CHECK(tok.msort_idx       != NULL,    "msort_idx allocated");
    CHECK(tok.merge_result    != NULL,    "merge_result allocated");
    CHECK(tok.decoded_vocab   != NULL,    "decoded_vocab allocated");

    /* Spot-check merge_result: merge 4 = (▁Hell,o) → ▁Hello = 15 */
    CHECK(tok.merge_result[4] == 15,      "merge_result[4] = 15 (▁Hello)");
    /* merge 9 = (▁worl,d) → ▁world = 20 */
    CHECK(tok.merge_result[9] == 20,      "merge_result[9] = 20 (▁world)");

    /* Spot-check decoded_vocab: ▁Hello (15) → " Hello" */
    const char *dv15 = tok.decoded_vocab[15];
    CHECK(dv15 && strcmp(dv15, " Hello") == 0, "decoded_vocab[15] = ' Hello'");
    /* Plain token "H" (4) → unchanged */
    CHECK(tok.decoded_vocab[4] == g_vocab[4], "decoded_vocab[4] points to raw 'H'");

    free(abuf);
}

/* =========================================================================
 * Tests: encode
 * ====================================================================== */

/* Shared built tokenizer for encode/decode tests */
static cmol_tokenizer_t g_tok;
static uint8_t          g_arena_buf[ARENA_SZ];
static cmol_arena_t     g_arena;
static int              g_tok_ready = 0;

static void ensure_tok(void) {
    if (g_tok_ready) return;
    setup_tok(&g_tok);
    cmol_arena_init(&g_arena, g_arena_buf, ARENA_SZ);
    cmol_err_t e = cmol_tokenizer_build(&g_tok, &g_arena);
    if (e != CMOL_OK) {
        fprintf(stderr, "FATAL: cmol_tokenizer_build failed: %d\n", e);
        exit(1);
    }
    g_tok_ready = 1;
}

static void test_encode_hello(void) {
    SECTION("Encode \"Hello\"");
    ensure_tok();

    int32_t out[16];
    int n = cmol_tokenizer_encode(&g_tok, "Hello", out, 16, 0);

    CHECK(n == 1,                   "1 token produced");
    CHECK(n == 1 && out[0] == 15,   "token = 15 (▁Hello)");
}

static void test_encode_bos(void) {
    SECTION("Encode with BOS prepend");
    ensure_tok();

    int32_t out[16];
    int n = cmol_tokenizer_encode(&g_tok, "Hello", out, 16, 1);

    CHECK(n == 2,                   "BOS + ▁Hello = 2 tokens");
    CHECK(n >= 2 && out[0] == 1,    "out[0] = BOS (1)");
    CHECK(n >= 2 && out[1] == 15,   "out[1] = 15 (▁Hello)");
}

static void test_encode_two_words(void) {
    SECTION("Encode \"Hello world\"");
    ensure_tok();

    int32_t out[16];
    int n = cmol_tokenizer_encode(&g_tok, "Hello world", out, 16, 0);

    CHECK(n == 2,                   "2 tokens produced");
    CHECK(n >= 2 && out[0] == 15,   "out[0] = 15 (▁Hello)");
    CHECK(n >= 2 && out[1] == 20,   "out[1] = 20 (▁world)");
}

static void test_encode_empty(void) {
    SECTION("Encode empty string");
    ensure_tok();

    int32_t out[4];
    int n;

    n = cmol_tokenizer_encode(&g_tok, "", out, 4, 0);
    CHECK(n == 0,                   "\"\" without BOS → 0 tokens");

    n = cmol_tokenizer_encode(&g_tok, "", out, 4, 1);
    CHECK(n == 1 && out[0] == 1,    "\"\" with BOS → [BOS]");
}

static void test_encode_trunc(void) {
    SECTION("Output buffer truncation");
    ensure_tok();

    int32_t out[1];
    /* BOS + ▁Hello needs 2 slots; only 1 available → TRUNC */
    int n = cmol_tokenizer_encode(&g_tok, "Hello", out, 1, 1);
    CHECK(n == CMOL_ERR_TRUNC,      "BOS+word with 1-slot cap → CMOL_ERR_TRUNC");
    CHECK(out[0] == 1,              "BOS written before truncation");
}

static void test_encode_bad_args(void) {
    SECTION("Encode bad arguments");
    ensure_tok();

    int32_t out[4];
    CHECK(cmol_tokenizer_encode(NULL,    "x",   out, 4, 0) == CMOL_ERR_ARGS,
          "NULL tok → CMOL_ERR_ARGS");
    CHECK(cmol_tokenizer_encode(&g_tok,  NULL,  out, 4, 0) == CMOL_ERR_ARGS,
          "NULL text → CMOL_ERR_ARGS");
    CHECK(cmol_tokenizer_encode(&g_tok,  "x",  NULL, 4, 0) == CMOL_ERR_ARGS,
          "NULL out → CMOL_ERR_ARGS");
    CHECK(cmol_tokenizer_encode(&g_tok,  "x",   out, 0, 0) == CMOL_ERR_ARGS,
          "out_cap=0 → CMOL_ERR_ARGS");
}

static void test_encode_before_build(void) {
    SECTION("Encode before build (UNSUPPORTED)");

    cmol_tokenizer_t raw;
    setup_tok(&raw);
    /* Do NOT call cmol_tokenizer_build() */

    int32_t out[4];
    CHECK(cmol_tokenizer_encode(&raw, "Hello", out, 4, 0) == CMOL_ERR_UNSUPPORTED,
          "encode without build → CMOL_ERR_UNSUPPORTED");
}

/* =========================================================================
 * Tests: decode
 * ====================================================================== */

static void test_decode(void) {
    SECTION("Decode tokens");
    ensure_tok();

    const char *d;

    /* ▁Hello (15) → " Hello" */
    d = cmol_tokenizer_decode_token(&g_tok, 15);
    CHECK(d && strcmp(d, " Hello") == 0,  "decode 15 → \" Hello\"");

    /* ▁world (20) → " world" */
    d = cmol_tokenizer_decode_token(&g_tok, 20);
    CHECK(d && strcmp(d, " world") == 0,  "decode 20 → \" world\"");

    /* ▁ alone (3) → " " (single space) */
    d = cmol_tokenizer_decode_token(&g_tok, 3);
    CHECK(d && strcmp(d, " ") == 0,       "decode 3 (▁) → \" \"");

    /* Plain 'H' (4) → "H" */
    d = cmol_tokenizer_decode_token(&g_tok, 4);
    CHECK(d && strcmp(d, "H") == 0,       "decode 4 (H) → \"H\"");

    /* BOS (1) → "<s>" — control token, unchanged */
    d = cmol_tokenizer_decode_token(&g_tok, 1);
    CHECK(d && strcmp(d, "<s>") == 0,     "decode 1 (<s>) → \"<s>\"");

    /* Out-of-range → NULL */
    d = cmol_tokenizer_decode_token(&g_tok, VOCAB_SIZE);
    CHECK(d == NULL,                      "OOB token_id → NULL");
    d = cmol_tokenizer_decode_token(&g_tok, -1);
    CHECK(d == NULL,                      "negative token_id → NULL");
    d = cmol_tokenizer_decode_token(NULL, 0);
    CHECK(d == NULL,                      "NULL tok decode → NULL");
}

static void test_decode_before_build(void) {
    SECTION("Decode before build (raw vocab fallback)");

    cmol_tokenizer_t raw;
    setup_tok(&raw);
    /* No build — should fall back to raw vocab string */

    const char *d = cmol_tokenizer_decode_token(&raw, 15);
    /* Should return raw vocab[15] = SP "Hello" = "▁Hello" (not decoded) */
    CHECK(d != NULL,                      "pre-build decode non-NULL");
    CHECK(d && strcmp(d, SP "Hello") == 0,"pre-build decode = raw \"▁Hello\"");
}

/* =========================================================================
 * Tests: byte-token decoding (synthetic <0xNN> token)
 * ====================================================================== */

static void test_byte_token_decode(void) {
    SECTION("Byte token decode (<0x48> → 'H')");

    uint8_t *abuf = (uint8_t *)malloc(ARENA_SZ);
    cmol_arena_t arena;
    cmol_arena_init(&arena, abuf, ARENA_SZ);

    /* Build a minimal tokenizer with one byte token */
    static const char *bvocab[] = { "<unk>", "<0x48>" }; /* 0x48 = 'H' */
    static uint8_t     btypes[] = { CMOL_TOKEN_NORMAL, CMOL_TOKEN_BYTE };

    cmol_tokenizer_t tok;
    memset(&tok, 0, sizeof tok);
    tok.vocab       = bvocab;
    tok.vocab_size  = 2;
    tok.token_type  = btypes;
    tok.bos_id = tok.eos_id = tok.unk_id = 0;

    cmol_err_t e = cmol_tokenizer_build(&tok, &arena);
    CHECK(e == CMOL_OK,                  "byte tok build OK");

    const char *d = cmol_tokenizer_decode_token(&tok, 1);
    CHECK(d != NULL,                     "byte token decoded non-NULL");
    /* decoded_vocab[1] should be "\x48" i.e. "H" */
    CHECK(d && d[0] == 'H' && d[1] == '\0', "byte token 0x48 → 'H'");

    free(abuf);
}

/* =========================================================================
 * main
 * ====================================================================== */

int main(void) {
    printf("=== test_tokenizer ===\n");

    test_build();
    test_encode_hello();
    test_encode_bos();
    test_encode_two_words();
    test_encode_empty();
    test_encode_trunc();
    test_encode_bad_args();
    test_encode_before_build();
    test_decode();
    test_decode_before_build();
    test_byte_token_decode();

    printf("\n=== %d/%d passed", g_pass, g_tests);
    if (g_fail) printf(", %d FAILED", g_fail);
    printf(" ===\n");

    return g_fail ? 1 : 0;
}
