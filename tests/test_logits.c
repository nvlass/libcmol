/*
 * test_logits.c — diagnostic: dump top-15 logits after full prefill
 *
 * Requires CMOL_TEST_GGUF to be set (skipped silently otherwise).
 * Run: CMOL_TEST_GGUF=models/SmolLM2-360M-Instruct-Q4_K_M.gguf make test
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"          /* cmol_model_forward; also pulls in cmol_internal.h */

/* Find top-n logit indices using a simple insertion-based approach. */
static void top_n(const float *logits, int vocab, int n,
                  int *idx_out, float *val_out) {
    int i, j;
    /* Initialize with worst possible values */
    for (i = 0; i < n; i++) { idx_out[i] = -1; val_out[i] = -1e38f; }
    for (j = 0; j < vocab; j++) {
        float v = logits[j];
        /* Insert if greater than current minimum in our top-n */
        if (v > val_out[n - 1]) {
            /* Find insertion point (keep sorted descending) */
            int pos = n - 1;
            while (pos > 0 && v > val_out[pos - 1]) pos--;
            /* Shift down */
            for (i = n - 1; i > pos; i--) {
                idx_out[i] = idx_out[i - 1];
                val_out[i] = val_out[i - 1];
            }
            idx_out[pos] = j;
            val_out[pos] = v;
        }
    }
}

int main(void) {
    const char *gguf = getenv("CMOL_TEST_GGUF");
    if (!gguf) {
        printf("test_logits: no CMOL_TEST_GGUF set — skipping\n");
        return 0;
    }

    cmol_config_t cfg = CMOL_DEFAULT_CONFIG;
    cmol_err_t    err;
    cmol_model_t *m = cmol_load(gguf, &cfg, &err);
    if (!m) { fprintf(stderr, "cmol_load: %s\n", cmol_strerror(err)); return 1; }

    char prompt[4096];
    cmol_format_chatml(NULL, "Mr. Anderson.", prompt, sizeof prompt);
    printf("Prompt: [%s]\n\n", prompt);

    /* Tokenize */
    int32_t toks[512];
    int     ntok = cmol_encode(m, prompt, toks, 512);
    if (ntok < 0) {
        fprintf(stderr, "encode failed: %d\n", ntok);
        cmol_free(m);
        return 1;
    }
    printf("Tokens (%d):", ntok);
    { int i; for (i = 0; i < ntok; i++) printf(" %d", toks[i]); }
    printf("\n\n");

    /* Run prefill via cmol_model_forward on each prompt token */
    cmol_session_t *s = cmol_session_acquire(m);
    if (!s) { fprintf(stderr, "no session\n"); cmol_free(m); return 1; }

    float *logits = NULL;
    {
        int i;
        for (i = 0; i < ntok; i++) {
            logits = cmol_model_forward(m, s, toks[i], i);
            if (!logits) {
                fprintf(stderr, "forward failed at token %d\n", i);
                cmol_free(m);
                return 1;
            }
        }
    }

    int vocab = m->hparams.vocab_size;

    printf("=== Model metadata ===\n");
    printf("  add_bos=%d bos_id=%d eos_id=%d tok_model=%d\n",
           m->tokenizer.add_bos, m->tokenizer.bos_id, m->tokenizer.eos_id,
           (int)m->tokenizer.tok_model);
    printf("  kernel: %s\n", m->kernels.name);
    printf("=== First 8 tensor dtypes ===\n");
    {
        int ti;
        for (ti = 0; ti < m->n_tensors && ti < 8; ti++)
            printf("  %-40s dtype=%d\n", m->tensors[ti].name, (int)m->tensors[ti].dtype);
    }

    printf("=== Top-15 logits after last prompt token ===\n");
    {
        int   idx[15]; float val[15]; int i;
        top_n(logits, vocab, 15, idx, val);
        for (i = 0; i < 15; i++) {
            const char *dec = cmol_decode_token(m, idx[i]);
            printf("  token %5d  logit %8.3f  decoded: %s\n",
                   idx[i], val[i], dec ? dec : "(null)");
        }
    }

    /* Probe specific tokens of interest */
    printf("\n--- Specific token probes ---\n");
    {
        const char *probes[] = {
            "I", " I", "I'm", " I'm", "I'm", " I'm",
            "Hello", " Hello", "Hi", " Hi",
            " ready", " here", NULL
        };
        int i;
        for (i = 0; probes[i]; i++) {
            int32_t t[4];
            int tn = cmol_encode(m, probes[i], t, 4);
            if (tn == 1 && t[0] >= 0 && t[0] < vocab)
                printf("  %-12s  token %5d  logit %8.3f\n",
                       probes[i], t[0], logits[t[0]]);
        }
    }

    cmol_session_release(s);
    cmol_free(m);
    printf("\ntest_logits: done\n");
    return 0;
}
