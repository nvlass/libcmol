/*
 * test_embd.c — verify embedding lookup against raw GGUF bytes
 *
 * Checks token 1 (<|im_start|>) embedding:
 *   1. Dequantizes it via cmol_dequant_row
 *   2. Reads the raw Q8_0 bytes from the mmap and manually dequantizes
 *   3. Compares the two
 *
 * Also prints: output.weight existence, tie_embeddings flag,
 * and logit for token 57 ("I") vs 19556 ("Hello") using just the
 * EMBEDDING as the input (bypasses transformer, sanity-checks LM head).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "model.h"   /* pulls in cmol_internal.h → full struct definitions */
#include "quant.h"   /* cmol_dequant_row */

static float f16_to_f32_local(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t f32;
    if      (exp == 0u && mant == 0u) f32 = sign;
    else if (exp == 31u)              f32 = sign | (0xFFu << 23) | (mant << 13);
    else                              f32 = sign | ((exp + (127u-15u)) << 23) | (mant << 13);
    float r; memcpy(&r, &f32, sizeof r); return r;
}

int main(void) {
    const char *gguf = getenv("CMOL_TEST_GGUF");
    if (!gguf) { printf("no CMOL_TEST_GGUF — skip\n"); return 0; }

    cmol_config_t cfg = CMOL_DEFAULT_CONFIG;
    cmol_err_t err;
    cmol_model_t *m = cmol_load(gguf, &cfg, &err);
    if (!m) { fprintf(stderr, "load: %s\n", cmol_strerror(err)); return 1; }

    /* --- Find relevant tensors ----------------------------------------- */
    cmol_tensor_t *embd = NULL, *output_w = NULL;
    int i;
    for (i = 0; i < m->n_tensors; i++) {
        if (strcmp(m->tensors[i].name, "token_embd.weight") == 0) embd = &m->tensors[i];
        if (strcmp(m->tensors[i].name, "output.weight")     == 0) output_w = &m->tensors[i];
    }

    printf("output.weight present: %s  (tie_embeddings=%d)\n",
           output_w ? "YES" : "NO", m->hparams.tie_embeddings);
    printf("token_embd dtype=%d  shape=[%lld, %lld]\n",
           (int)embd->dtype,
           (long long)embd->shape[0], (long long)embd->shape[1]);
    if (output_w)
        printf("output.weight dtype=%d  shape=[%lld, %lld]\n",
               (int)output_w->dtype,
               (long long)output_w->shape[0], (long long)output_w->shape[1]);

    int d = m->hparams.d_model;  /* 960 */

    /* --- Dequantize embedding of token 1 via our API ------------------- */
    float embd1[1024];
    {
        /* row_bytes for Q8_0: (d/32)*34 */
        size_t rb = ((size_t)d / 32) * 34u;
        const uint8_t *row = (const uint8_t *)embd->data + 1u * rb;
        cmol_dequant_row(row, embd1, d, embd->dtype);
    }
    printf("\nEmbedding of token 1 (first 8 via cmol_dequant_row):\n");
    for (i = 0; i < 8; i++) printf("  embd1[%d] = %f\n", i, embd1[i]);

    /* --- Manual dequant of first Q8_0 block of token-1 row ------------- */
    printf("\nManual dequant of block 0 of token-1 row:\n");
    {
        size_t rb = ((size_t)d / 32) * 34u;
        const uint8_t *row = (const uint8_t *)embd->data + 1u * rb;
        /* Q8_0 block: uint16_t d, int8_t qs[32] */
        uint16_t raw_d;
        memcpy(&raw_d, row, 2);
        float scale = f16_to_f32_local(raw_d);
        const int8_t *qs = (const int8_t *)(row + 2);
        printf("  scale = %f\n", scale);
        for (i = 0; i < 8; i++) {
            float v = scale * (float)qs[i];
            printf("  manual[%d] = %f  qs=%d  (diff: %e)\n",
                   i, v, (int)qs[i], v - embd1[i]);
        }
    }

    /* --- Check logit for token 57 ("I") vs 19556 ("Hello") ------------- */
    /* We use the token-1 embedding as the "final hidden state" to test
     * just the LM-head matmul.  Since the embd IS the LM head (tied),
     * logit[i] = dot(embd1, embd_row_i). */
    printf("\nLM-head sanity (xnorm = embedding of token 1):\n");
    {
        cmol_tensor_t *lm = output_w ? output_w : embd;
        float logit_I = 0.0f, logit_Hello = 0.0f;
        float row_buf[1024];

        /* token 57 */
        size_t rb = ((size_t)d / (size_t)(lm->dtype == 8 ? 32 : 32))
                  * (size_t)(lm->dtype == 8 ? 34 : 22);
        /* For Q8_0: rb = (d/32)*34 */
        rb = ((size_t)d / 32u) * 34u;
        cmol_dequant_row((const uint8_t *)lm->data + 57u   * rb, row_buf, d, lm->dtype);
        for (i = 0; i < d; i++) logit_I     += row_buf[i] * embd1[i];

        cmol_dequant_row((const uint8_t *)lm->data + 19556u * rb, row_buf, d, lm->dtype);
        for (i = 0; i < d; i++) logit_Hello += row_buf[i] * embd1[i];

        printf("  logit(\"I\",   57):    %f\n", logit_I);
        printf("  logit(\"Hello\",19556): %f\n", logit_Hello);
    }

    cmol_free(m);
    printf("\ntest_embd: done\n");
    return 0;
}
