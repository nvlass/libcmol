/*
 * tokenizer.c — BPE tokenizer
 * Implemented in Phase 3.
 * Included by src/cmol.c (unity build); do not compile standalone.
 */

#include "tokenizer.h"

/* Phase 3 — TODO */

int cmol_tokenizer_encode(const cmol_tokenizer_t *tok,
                           const char *text,
                           int32_t    *out,
                           int         out_cap,
                           int         add_bos) {
    (void)tok; (void)text; (void)out; (void)out_cap; (void)add_bos;
    return CMOL_ERR_UNSUPPORTED;
}

const char *cmol_tokenizer_decode_token(const cmol_tokenizer_t *tok,
                                         int32_t token_id) {
    (void)tok; (void)token_id;
    return NULL;
}
