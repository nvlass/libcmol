/*
 * test_generate.c — Phase 8 integration tests for cmol_load + cmol_generate
 *
 * Tests are split into two groups:
 *
 *   Unit tests (always run, no model required)
 *     — parameter validation, error path coverage, stub behaviour
 *
 *   Live tests (skip unless CMOL_TEST_GGUF is set)
 *     — real GGUF load, full generate pipeline, greedy determinism
 *
 * Usage:
 *   make test                                          # unit tests only
 *   CMOL_TEST_GGUF=models/SmolLM2-135M-Instruct-Q4_K_M.gguf make test
 */

#include "../include/cmol.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* =========================================================================
 * Minimal test harness
 * ====================================================================== */

static int g_pass = 0;
static int g_fail = 0;

#define PASS(msg) do { printf("  PASS  %s\n", (msg)); g_pass++; } while (0)
#define FAIL(msg) do { printf("  FAIL  %s\n", (msg)); g_fail++; } while (0)
#define CHECK(cond, msg) do { if (cond) PASS(msg); else FAIL(msg); } while (0)

/* =========================================================================
 * Section 1: API-level unit tests (no model file required)
 * ====================================================================== */

static void test_strerror(void) {
    printf("\n[cmol_strerror]\n");
    CHECK(strcmp(cmol_strerror(CMOL_OK),            "success") == 0,
          "CMOL_OK");
    CHECK(strcmp(cmol_strerror(CMOL_ERR_OOM),       "out of memory") == 0,
          "CMOL_ERR_OOM");
    CHECK(strcmp(cmol_strerror(CMOL_ERR_CTX_FULL),
                 "KV cache full — call cmol_session_reset()") == 0,
          "CMOL_ERR_CTX_FULL");
    CHECK(cmol_strerror((cmol_err_t)-999) != NULL,
          "unknown code → non-NULL");
}

static void test_version(void) {
    printf("\n[cmol_version]\n");
    const char *v = cmol_version();
    CHECK(v != NULL,     "version string non-NULL");
    CHECK(strlen(v) > 0, "version string non-empty");
    /* Must be in major.minor.patch form */
    int major = 0, minor = 0, patch = 0;
    int n = sscanf(v, "%d.%d.%d", &major, &minor, &patch);
    CHECK(n == 3, "version has 3 numeric parts");
}

static void test_load_null_args(void) {
    printf("\n[cmol_load null/bad args]\n");
    cmol_err_t err;

    cmol_model_t *m = cmol_load(NULL, NULL, &err);
    CHECK(m == NULL,            "NULL path → NULL");
    CHECK(err == CMOL_ERR_ARGS, "NULL path → CMOL_ERR_ARGS");

    m = cmol_load("/nonexistent/path.gguf", NULL, &err);
    CHECK(m == NULL,           "missing file → NULL");
    CHECK(err != CMOL_OK,      "missing file → error code");

    /* cmol_free(NULL) must not crash */
    cmol_free(NULL);
    PASS("cmol_free(NULL) is safe");
}

static void test_session_null_guards(void) {
    printf("\n[session null guards]\n");
    cmol_session_t *s = cmol_session_acquire(NULL);
    CHECK(s == NULL, "acquire(NULL) → NULL");

    /* These must not crash */
    cmol_session_release(NULL);
    PASS("release(NULL) is safe");
    cmol_session_reset(NULL);
    PASS("reset(NULL) is safe");
}

static void test_generate_null_args(void) {
    printf("\n[cmol_generate null args]\n");
    cmol_err_t rc;

    rc = cmol_generate(NULL, "hi", NULL, NULL, NULL);
    CHECK(rc == CMOL_ERR_ARGS, "NULL session → CMOL_ERR_ARGS");

    rc = cmol_generate((cmol_session_t *)(void *)1, NULL, NULL, NULL, NULL);
    CHECK(rc == CMOL_ERR_ARGS, "NULL prompt → CMOL_ERR_ARGS");
}

static void test_arena_estimate_null(void) {
    printf("\n[cmol_arena_estimate guards]\n");
    size_t n = cmol_arena_estimate(NULL, NULL);
    CHECK(n == 0, "NULL path → 0");

    n = cmol_arena_estimate("/nonexistent.gguf", NULL);
    CHECK(n == 0, "missing file → 0");
}

static void test_encode_null_guards(void) {
    printf("\n[cmol_encode / cmol_decode_token null guards]\n");
    int32_t buf[4];

    int rc = cmol_encode(NULL, "hi", buf, 4);
    CHECK(rc == CMOL_ERR_ARGS, "encode NULL model → CMOL_ERR_ARGS");

    rc = cmol_encode((cmol_model_t *)(void *)1, NULL, buf, 4);
    CHECK(rc == CMOL_ERR_ARGS, "encode NULL text → CMOL_ERR_ARGS");

    const char *p = cmol_decode_token(NULL, 0);
    CHECK(p == NULL, "decode NULL model → NULL");
}

static void test_chatml_format(void) {
    char buf[512];
    int  n;

    printf("\n[cmol_format_chatml — default system message]\n");
    n = cmol_format_chatml(NULL, "Hi", buf, sizeof buf);
    CHECK(n > 0, "returns positive length");
    CHECK(strstr(buf, "<|im_start|>system\n") != NULL,
          "contains system block");
    CHECK(strstr(buf, "SmolLM") != NULL, "contains 'SmolLM' default");
    CHECK(strstr(buf, "<|im_start|>user\nHi<|im_end|>") != NULL,
          "contains user message");
    CHECK(strstr(buf, "<|im_start|>assistant\n") != NULL,
          "ends with assistant header");

    printf("\n[cmol_format_chatml — custom system message]\n");
    n = cmol_format_chatml("Be brief.", "Hi", buf, sizeof buf);
    CHECK(n > 0, "returns positive length");
    CHECK(strstr(buf, "<|im_start|>system\nBe brief.<|im_end|>") != NULL,
          "custom system text present");

    printf("\n[cmol_format_chatml — empty system (omit block)]\n");
    n = cmol_format_chatml("", "Hi", buf, sizeof buf);
    CHECK(n > 0, "returns positive length");
    CHECK(strstr(buf, "system") == NULL, "no system block when sys=''");
    CHECK(strstr(buf, "<|im_start|>user\nHi<|im_end|>") != NULL,
          "user message still present");

    printf("\n[cmol_format_chatml_turn]\n");
    n = cmol_format_chatml_turn("Next question", buf, sizeof buf);
    CHECK(n > 0, "turn: returns positive length");
    CHECK(strstr(buf, "<|im_end|>\n<|im_start|>user\n") != NULL,
          "turn: closes previous + opens user");
    CHECK(strstr(buf, "<|im_start|>assistant\n") != NULL,
          "turn: ends with assistant header");

    printf("\n[cmol_format_chatml — size probe]\n");
    n = cmol_format_chatml(NULL, "Hi", NULL, 0);
    CHECK(n > 0, "probe: returns positive size (not error)");
    /* Exact re-check: the real write should give same n */
    int n2 = cmol_format_chatml(NULL, "Hi", buf, sizeof buf);
    CHECK(n == n2, "probe matches real write length");

    printf("\n[cmol_format_chatml — truncation]\n");
    char small[4];
    n = cmol_format_chatml(NULL, "Hi", small, sizeof small);
    CHECK(n == (int)CMOL_ERR_TRUNC, "small buffer → CMOL_ERR_TRUNC");
    CHECK(small[sizeof small - 1] == '\0', "buffer NUL-terminated on truncation");

    printf("\n[cmol_format_chatml — NULL user guard]\n");
    n = cmol_format_chatml(NULL, NULL, buf, sizeof buf);
    CHECK(n == (int)CMOL_ERR_ARGS, "NULL user → CMOL_ERR_ARGS");
    n = cmol_format_chatml_turn(NULL, buf, sizeof buf);
    CHECK(n == (int)CMOL_ERR_ARGS, "turn NULL user → CMOL_ERR_ARGS");
}

/* =========================================================================
 * Section 2: live tests — require CMOL_TEST_GGUF
 * ====================================================================== */

#ifdef CMOL_TEST_GGUF

/* ── Callback helpers (live tests only) ──────────────────────────────────── */

typedef struct {
    char   buf[4096];
    size_t len;
    int    eos_seen;
    int    n_tokens;
} collect_ctx_t;

static int collect_cb(const char *piece, size_t len, int is_eos, void *ud) {
    collect_ctx_t *c = (collect_ctx_t *)ud;
    if (!is_eos && len > 0) {
        size_t copy = len;
        if (c->len + copy + 1 > sizeof c->buf)
            copy = sizeof(c->buf) - c->len - 1;
        memcpy(c->buf + c->len, piece, copy);
        c->len += copy;
        c->buf[c->len] = '\0';
    }
    if (is_eos) c->eos_seen = 1;
    c->n_tokens++;
    return 0; /* continue */
}

/* Callback that aborts after the first token */
static int abort_after_one_cb(const char *piece, size_t len,
                               int is_eos, void *ud) {
    int *count = (int *)ud;
    (void)piece; (void)len; (void)is_eos;
    (*count)++;
    return 1; /* abort */
}

static void live_test_load(const char *path, cmol_model_t **m_out) {
    printf("\n[Live: cmol_load]\n");

    cmol_config_t cfg = CMOL_DEFAULT_CONFIG;
    cfg.max_ctx      = 512;  /* keep it small for test speed    */
    cfg.max_sessions = 2;

    cmol_err_t err = CMOL_OK;
    cmol_model_t *m = cmol_load(path, &cfg, &err);

    CHECK(m   != NULL,    "cmol_load returns non-NULL");
    CHECK(err == CMOL_OK, "cmol_load err == CMOL_OK");

    if (!m) { *m_out = NULL; return; }

    /* Sanity-check hparams via tokenizer (indirectly: encode should work) */
    int32_t toks[8];
    int n = cmol_encode(m, "Hello", toks, 8);
    CHECK(n > 0, "encode 'Hello' → at least 1 token");
    CHECK(n < 8, "encode 'Hello' → fewer than 8 tokens");

    const char *piece = cmol_decode_token(m, toks[0]);
    CHECK(piece != NULL, "decode first token → non-NULL");

    *m_out = m;
}

static void live_test_arena_estimate(const char *path) {
    printf("\n[Live: cmol_arena_estimate]\n");
    cmol_config_t cfg = CMOL_DEFAULT_CONFIG;
    cfg.max_ctx      = 512;
    cfg.max_sessions = 2;
    size_t est = cmol_arena_estimate(path, &cfg);
    CHECK(est > 0, "estimate > 0 for real model");
    printf("       estimate = %zu MB\n", est / (1024 * 1024));
}

static void live_test_session_pool(cmol_model_t *m) {
    printf("\n[Live: session pool]\n");
    if (!m) { printf("  SKIP  (no model)\n"); return; }

    cmol_session_t *s1 = cmol_session_acquire(m);
    CHECK(s1 != NULL, "first acquire succeeds");

    cmol_session_t *s2 = cmol_session_acquire(m);
    CHECK(s2 != NULL, "second acquire succeeds (max_sessions=2)");

    /* Pool is now full (max_sessions=2) */
    cmol_session_t *s3 = cmol_session_acquire(m);
    CHECK(s3 == NULL, "third acquire fails (pool full)");

    /* Release one and re-acquire */
    cmol_session_release(s2);
    s2 = cmol_session_acquire(m);
    CHECK(s2 != NULL, "re-acquire after release succeeds");

    cmol_session_release(s1);
    cmol_session_release(s2);
    PASS("all sessions released");
}

static void live_test_generate_basic(cmol_model_t *m) {
    printf("\n[Live: basic generation]\n");
    if (!m) { printf("  SKIP  (no model)\n"); return; }

    cmol_session_t *s = cmol_session_acquire(m);
    if (!s) { printf("  SKIP  (no session available)\n"); return; }

    cmol_gen_params_t p = CMOL_DEFAULT_PARAMS;
    p.max_new_tokens  = 20;
    p.seed            = 42;

    collect_ctx_t ctx;
    memset(&ctx, 0, sizeof ctx);

    cmol_err_t rc = cmol_generate(s, "Hello", &p, collect_cb, &ctx);

    CHECK(rc == CMOL_OK,      "generate returns CMOL_OK");
    CHECK(ctx.n_tokens > 0,   "callback fired at least once");
    CHECK(ctx.len > 0,        "output text non-empty");
    printf("       output: \"%.*s\"\n", (int)ctx.len, ctx.buf);

    cmol_session_release(s);
}

static void live_test_generate_greedy_deterministic(cmol_model_t *m) {
    printf("\n[Live: greedy determinism]\n");
    if (!m) { printf("  SKIP  (no model)\n"); return; }

    cmol_gen_params_t p = CMOL_DEFAULT_PARAMS;
    p.temperature    = 0.0f;   /* greedy */
    p.max_new_tokens = 10;

    char out1[1024] = {0};
    char out2[1024] = {0};

    {
        cmol_session_t *s = cmol_session_acquire(m);
        if (!s) { printf("  SKIP  (no session)\n"); return; }
        collect_ctx_t ctx; memset(&ctx, 0, sizeof ctx);
        cmol_generate(s, "Once upon", &p, collect_cb, &ctx);
        memcpy(out1, ctx.buf, ctx.len);
        cmol_session_release(s);
    }
    {
        cmol_session_t *s = cmol_session_acquire(m);
        if (!s) { printf("  SKIP  (no session)\n"); return; }
        collect_ctx_t ctx; memset(&ctx, 0, sizeof ctx);
        cmol_generate(s, "Once upon", &p, collect_cb, &ctx);
        memcpy(out2, ctx.buf, ctx.len);
        cmol_session_release(s);
    }

    CHECK(strlen(out1) > 0,         "greedy run 1 non-empty");
    CHECK(strcmp(out1, out2) == 0,  "greedy identical on two fresh sessions");
    printf("       output: \"%s\"\n", out1);
}

static void live_test_generate_eos_terminates(cmol_model_t *m) {
    printf("\n[Live: EOS terminates generation]\n");
    if (!m) { printf("  SKIP  (no model)\n"); return; }

    cmol_session_t *s = cmol_session_acquire(m);
    if (!s) { printf("  SKIP  (no session)\n"); return; }

    cmol_gen_params_t p = CMOL_DEFAULT_PARAMS;
    p.temperature    = 0.0f;
    p.max_new_tokens = 512;  /* big limit — should stop on EOS */

    collect_ctx_t ctx; memset(&ctx, 0, sizeof ctx);
    cmol_err_t rc = cmol_generate(s, "<|im_end|>", &p, collect_cb, &ctx);

    CHECK(rc == CMOL_OK,       "generate with EOS-inducing prompt returns OK");
    /* The model may immediately produce EOS; we just want no crash */
    PASS("EOS prompt did not crash");

    cmol_session_release(s);
}

static void live_test_callback_abort(cmol_model_t *m) {
    printf("\n[Live: callback abort]\n");
    if (!m) { printf("  SKIP  (no model)\n"); return; }

    cmol_session_t *s = cmol_session_acquire(m);
    if (!s) { printf("  SKIP  (no session)\n"); return; }

    cmol_gen_params_t p = CMOL_DEFAULT_PARAMS;
    p.max_new_tokens = 100;
    p.seed           = 1;

    int count = 0;
    cmol_err_t rc = cmol_generate(s, "Hello", &p, abort_after_one_cb, &count);

    CHECK(rc == CMOL_OK, "generate returns CMOL_OK after callback abort");
    CHECK(count == 1,    "exactly one callback call before abort");

    cmol_session_release(s);
}

static void live_test_session_reset(cmol_model_t *m) {
    printf("\n[Live: session reset]\n");
    if (!m) { printf("  SKIP  (no model)\n"); return; }

    cmol_gen_params_t p = CMOL_DEFAULT_PARAMS;
    p.temperature    = 0.0f;
    p.max_new_tokens = 5;

    char out1[256] = {0};
    char out2[256] = {0};

    cmol_session_t *s = cmol_session_acquire(m);
    if (!s) { printf("  SKIP  (no session)\n"); return; }

    { collect_ctx_t ctx; memset(&ctx, 0, sizeof ctx);
      cmol_generate(s, "Hello", &p, collect_cb, &ctx);
      memcpy(out1, ctx.buf, ctx.len); }

    cmol_session_reset(s);

    { collect_ctx_t ctx; memset(&ctx, 0, sizeof ctx);
      cmol_generate(s, "Hello", &p, collect_cb, &ctx);
      memcpy(out2, ctx.buf, ctx.len); }

    CHECK(strlen(out1) > 0,        "first run produced output");
    CHECK(strcmp(out1, out2) == 0, "same output after session_reset (greedy)");

    cmol_session_release(s);
}

static void live_test_multi_turn(cmol_model_t *m) {
    printf("\n[Live: multi-turn (KV cache accumulation)]\n");
    if (!m) { printf("  SKIP  (no model)\n"); return; }

    cmol_session_t *s = cmol_session_acquire(m);
    if (!s) { printf("  SKIP  (no session)\n"); return; }

    cmol_gen_params_t p = CMOL_DEFAULT_PARAMS;
    p.max_new_tokens = 5;
    p.seed           = 7;

    collect_ctx_t ctx1; memset(&ctx1, 0, sizeof ctx1);
    cmol_err_t rc1 = cmol_generate(s, "Hello", &p, collect_cb, &ctx1);

    collect_ctx_t ctx2; memset(&ctx2, 0, sizeof ctx2);
    cmol_err_t rc2 = cmol_generate(s, " world", &p, collect_cb, &ctx2);

    CHECK(rc1 == CMOL_OK,    "first turn OK");
    CHECK(rc2 == CMOL_OK,    "second turn OK");
    CHECK(ctx1.n_tokens > 0, "first turn produced tokens");
    CHECK(ctx2.n_tokens > 0, "second turn produced tokens");

    cmol_session_release(s);
}

#endif /* CMOL_TEST_GGUF */

/* =========================================================================
 * main
 * ====================================================================== */

int main(void) {
    /* ── Unit tests (always run) ───────────────────────────────────────── */
    printf("=== test_generate ===\n");
    printf("\n[Unit tests — no model required]\n");

    test_strerror();
    test_version();
    test_load_null_args();
    test_session_null_guards();
    test_generate_null_args();
    test_arena_estimate_null();
    test_encode_null_guards();
    test_chatml_format();

#ifdef CMOL_TEST_GGUF
    /* ── Live tests ─────────────────────────────────────────────────────── */
    const char *gguf_path = CMOL_TEST_GGUF;
    printf("\n[Live tests — model: %s]\n", gguf_path);

    cmol_model_t *m = NULL;
    live_test_load(gguf_path, &m);
    live_test_arena_estimate(gguf_path);
    live_test_session_pool(m);
    live_test_generate_basic(m);
    live_test_generate_greedy_deterministic(m);
    live_test_generate_eos_terminates(m);
    live_test_callback_abort(m);
    live_test_session_reset(m);
    live_test_multi_turn(m);

    cmol_free(m);
    PASS("cmol_free succeeds");
#else
    printf("\n[Live tests skipped — set CMOL_TEST_GGUF=/path/to/model.gguf]\n");
#endif

    printf("\n=== %d/%d passed ===\n", g_pass, g_pass + g_fail);
    return g_fail ? 1 : 0;
}
