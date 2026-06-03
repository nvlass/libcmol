/*
 * test_kernel.c — compare NEON matmul output against scalar for layer 0 Q projection
 *
 * Uses token_embd.weight (Q8_0) for the embedding of token 1, runs it through
 * blk.0.attn_norm.weight and blk.0.attn_q.weight using BOTH the selected kernel
 * and the scalar fallback, then diffs the results.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "model.h"
#include "quant.h"

int main(void) {
    const char *gguf = getenv("CMOL_TEST_GGUF");
    if (!gguf) { printf("no CMOL_TEST_GGUF — skip\n"); return 0; }

    cmol_config_t cfg = CMOL_DEFAULT_CONFIG;
    cmol_err_t err;
    cmol_model_t *m = cmol_load(gguf, &cfg, &err);
    if (!m) { fprintf(stderr, "load: %s\n", cmol_strerror(err)); return 1; }

    int d = m->hparams.d_model;           /* 960 */
    float eps = m->hparams.rms_norm_eps;

    /* --- Find needed tensors ------------------------------------------ */
    cmol_tensor_t *embd = NULL, *attn_norm = NULL, *attn_q = NULL;
    int i;
    for (i = 0; i < m->n_tensors; i++) {
        if (!strcmp(m->tensors[i].name, "token_embd.weight"))       embd      = &m->tensors[i];
        if (!strcmp(m->tensors[i].name, "blk.0.attn_norm.weight"))  attn_norm = &m->tensors[i];
        if (!strcmp(m->tensors[i].name, "blk.0.attn_q.weight"))     attn_q    = &m->tensors[i];
    }
    printf("embd: dtype=%d  attn_norm: dtype=%d  attn_q: dtype=%d\n",
           embd ? (int)embd->dtype : -1,
           attn_norm ? (int)attn_norm->dtype : -1,
           attn_q ? (int)attn_q->dtype : -1);

    /* --- Dequantize embedding of token 1 ------------------------------- */
    static float x[1024], xnorm[1024], q_neon[1024], q_scalar[1024];
    size_t rb = ((size_t)d / 32u) * 34u;  /* Q8_0 row bytes */
    cmol_dequant_row((const uint8_t *)embd->data + 1u * rb, x, d, embd->dtype);

    /* --- RMSNorm -------------------------------------------------------- */
    cmol_rms_norm(x, (float *)attn_norm->data, xnorm, d, eps);

    printf("xnorm[0..3]: %f %f %f %f\n", xnorm[0], xnorm[1], xnorm[2], xnorm[3]);

    /* --- Q projection: selected kernel vs scalar ----------------------- */
    m->kernels.matmul(q_neon,   attn_q->data, xnorm, d, d, 1, attn_q->dtype);
    cmol_matmul_scalar(q_scalar, attn_q->data, xnorm, d, d, 1, attn_q->dtype);

    printf("kernel: %s\n", m->kernels.name);
    printf("\nQ projection first 8 values:\n");
    printf("  %-10s  %-10s  %-12s\n", "NEON/sel", "scalar", "diff");
    float max_diff = 0.0f;
    for (i = 0; i < 8; i++) {
        float diff = q_neon[i] - q_scalar[i];
        if (fabsf(diff) > fabsf(max_diff)) max_diff = diff;
        printf("  %10.6f  %10.6f  %12.2e\n", q_neon[i], q_scalar[i], diff);
    }

    /* --- Max abs diff across all d outputs ----------------------------- */
    for (i = 0; i < d; i++) {
        float diff = q_neon[i] - q_scalar[i];
        if (fabsf(diff) > fabsf(max_diff)) max_diff = diff;
    }
    printf("\nMax |diff| across all %d Q outputs: %e\n", d, fabsf(max_diff));

    cmol_free(m);
    printf("test_kernel: done\n");
    return 0;
}
