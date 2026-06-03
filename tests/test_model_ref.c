/*
 * test_model_ref.c — compare C forward-pass logits against Python reference.
 *
 * For each position recorded by tools/gen_ref.py, runs our C
 * cmol_model_forward() with the identical token sequence and checks
 * that the resulting logits agree within a tight tolerance.
 *
 * Environment variables (both required; test is skipped if absent):
 *   CMOL_TEST_GGUF  — path to the GGUF model file
 *   CMOL_REF_BIN    — path to the binary reference file from gen_ref.py
 *
 * Binary format (little-endian, must match gen_ref.py):
 *   uint32  magic    = 0x4D4F4C52
 *   uint32  version  = 1
 *   int32   n_prompt
 *   int32   vocab
 *   int32   n_gen
 *   int32   tokens[n_prompt]
 *   # n_gen+1 records (last prefill + n_gen generation positions):
 *   for each record:
 *       int32    best_tok
 *       float32  logits[vocab]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "model.h"   /* pulls in cmol_internal.h, cmol.h */

#define REF_MAGIC   0x4D4F4C52u
#define REF_VERSION 1u

/* Max logit diff tolerated: Python uses float64 internally, we use float32,
 * so small rounding differences are expected.  1e-3 is generous.          */
#define PASS_THRESH 1e-3f

/* =========================================================================
 * Helpers
 * ====================================================================== */

static float max_abs_diff(const float *a, const float *b, int n) {
    float d = 0.0f;
    int i;
    for (i = 0; i < n; i++) {
        float x = a[i] - b[i];
        if (x < 0.0f) x = -x;
        if (x > d) d = x;
    }
    return d;
}

static int argmax_f(const float *x, int n) {
    int best = 0, i;
    for (i = 1; i < n; i++) if (x[i] > x[best]) best = i;
    return best;
}

/* Print top-N token IDs and their logits from a logit array. */
static void print_top(const float *logits, int vocab, int topn,
                      cmol_model_t *m, const char *label) {
    int   idx[8];
    float val[8];
    int   i, j;
    if (topn > 8) topn = 8;
    for (i = 0; i < topn; i++) { idx[i] = -1; val[i] = -1e38f; }
    for (j = 0; j < vocab; j++) {
        if (logits[j] > val[topn-1]) {
            int p = topn - 1;
            while (p > 0 && logits[j] > val[p-1]) p--;
            for (i = topn-1; i > p; i--) { idx[i]=idx[i-1]; val[i]=val[i-1]; }
            idx[p] = j; val[p] = logits[j];
        }
    }
    printf("    %s top-%d:", label, topn);
    for (i = 0; i < topn; i++) {
        const char *dec = m ? cmol_decode_token(m, idx[i]) : NULL;
        printf("  [%d]=%.3f(%s)", idx[i], val[i], dec ? dec : "?");
    }
    printf("\n");
}

/* =========================================================================
 * main
 * ====================================================================== */

int main(void) {
    const char *gguf_path = getenv("CMOL_TEST_GGUF");
    const char *ref_path  = getenv("CMOL_REF_BIN");

    if (!gguf_path || !ref_path) {
        printf("test_model_ref: CMOL_TEST_GGUF and CMOL_REF_BIN required — skipping\n");
        return 0;
    }

    /* ── Read reference binary ─────────────────────────────────────────── */
    FILE *fp = fopen(ref_path, "rb");
    if (!fp) { perror(ref_path); return 1; }

    uint32_t magic, version;
    int32_t  n_prompt, vocab, n_gen;
    if (fread(&magic,   4, 1, fp) != 1 ||
        fread(&version, 4, 1, fp) != 1) { fprintf(stderr, "short read\n"); return 1; }
    if (magic != REF_MAGIC || version != REF_VERSION) {
        fprintf(stderr, "bad magic/version (got 0x%08X v%u)\n", magic, version);
        fclose(fp); return 1;
    }
    fread(&n_prompt, 4, 1, fp);
    fread(&vocab,    4, 1, fp);
    fread(&n_gen,    4, 1, fp);

    int32_t *tokens   = (int32_t *)malloc((size_t)n_prompt * 4);
    int      n_rec    = n_gen + 1;
    int32_t *ref_best = (int32_t *)malloc((size_t)n_rec * 4);
    float   *ref_lgs  = (float *)malloc((size_t)n_rec * (size_t)vocab * 4);

    if (!tokens || !ref_best || !ref_lgs) {
        fprintf(stderr, "OOM allocating reference buffers\n"); return 1;
    }

    fread(tokens, 4, (size_t)n_prompt, fp);
    {
        int r;
        for (r = 0; r < n_rec; r++) {
            fread(&ref_best[r], 4, 1, fp);
            fread(ref_lgs + (size_t)r * (size_t)vocab, 4, (size_t)vocab, fp);
        }
    }
    fclose(fp);

    printf("=== test_model_ref ===\n");
    printf("ref: n_prompt=%d  vocab=%d  n_gen=%d\n", n_prompt, vocab, n_gen);
    printf("prompt tokens:");
    { int i; for (i = 0; i < n_prompt; i++) printf(" %d", tokens[i]); }
    printf("\n\n");

    /* ── Load model ────────────────────────────────────────────────────── */
    cmol_err_t    err;
    cmol_config_t cfg = CMOL_DEFAULT_CONFIG;
    cmol_model_t *m   = cmol_load(gguf_path, &cfg, &err);
    if (!m) {
        fprintf(stderr, "cmol_load: %s\n", cmol_strerror(err));
        free(tokens); free(ref_best); free(ref_lgs); return 1;
    }
    printf("model: kernel=%s  vocab=%d  d=%d  layers=%d\n",
           m->kernels.name, m->hparams.vocab_size,
           m->hparams.d_model, m->hparams.n_layers);

    if (m->hparams.vocab_size != vocab) {
        fprintf(stderr, "FAIL vocab mismatch: model=%d ref=%d\n",
                m->hparams.vocab_size, vocab);
        cmol_free(m); free(tokens); free(ref_best); free(ref_lgs); return 1;
    }

    cmol_session_t *s = cmol_session_acquire(m);
    if (!s) { fprintf(stderr, "no session slot\n"); cmol_free(m); return 1; }

    /* ── Prefill ───────────────────────────────────────────────────────── */
    float *logits = NULL;
    int    i;
    printf("Prefilling %d tokens ...\n", n_prompt);
    for (i = 0; i < n_prompt; i++) {
        logits = cmol_model_forward(m, s, tokens[i], i);
        if (!logits) {
            fprintf(stderr, "cmol_model_forward failed at pos %d\n", i);
            cmol_free(m); return 1;
        }
    }
    printf("\n");

    /* ── Compare at each recorded position ────────────────────────────── */
    int  n_pass = 0, n_fail = 0;
    int  tok    = (int)ref_best[0];   /* use reference greedy tokens to stay in sync */

    for (i = 0; i < n_rec; i++) {
        int   pos      = (n_prompt - 1) + i;   /* positions: n_prompt-1, n_prompt, ... */
        float *ref     = ref_lgs + (size_t)i * (size_t)vocab;
        float  md      = max_abs_diff(logits, ref, vocab);
        int    our_top = argmax_f(logits, vocab);
        int    ref_top = (int)ref_best[i];
        int    pass    = (md < PASS_THRESH) && (our_top == ref_top);

        if (i == 0)
            printf("--- pos %d (last prefill) ---\n", pos);
        else
            printf("--- pos %d (gen step %d, input tok=%d) ---\n", pos, i - 1, tok);

        printf("  max_diff=%.4e  our_best=%d  ref_best=%d  %s\n",
               (double)md, our_top, ref_top, pass ? "PASS" : "FAIL");

        if (!pass) {
            /* Print diagnostics to help locate the bug */
            print_top(logits, vocab, 5, m, "our");
            print_top(ref,    vocab, 5, m, "ref");
            n_fail++;
        } else {
            n_pass++;
        }

        /* Feed the reference's greedy token for the next generation step */
        tok    = ref_top;

        if (i < n_rec - 1) {
            /* Forward pass for the next position using the reference token */
            int next_pos = pos + 1;
            logits = cmol_model_forward(m, s, (int32_t)tok, next_pos);
            if (!logits) {
                fprintf(stderr, "forward failed at pos %d\n", next_pos);
                n_fail++;
                break;
            }
        }
    }

    printf("\n=== %d/%d passed", n_pass, n_rec);
    if (n_fail) printf(", %d FAILED", n_fail);
    printf(" ===\n");

    cmol_session_release(s);
    cmol_free(m);
    free(tokens); free(ref_best); free(ref_lgs);
    return (n_fail == 0) ? 0 : 1;
}
