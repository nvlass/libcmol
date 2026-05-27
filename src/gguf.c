/*
 * gguf.c — GGUF file parser
 * Implemented in Phase 1.
 * Included by src/cmol.c (unity build); do not compile standalone.
 */

#include "gguf.h"

/* Phase 1 — TODO */

cmol_err_t cmol_gguf_parse(const cmol_mmap_t *mmap,
                             cmol_arena_t      *arena,
                             cmol_hparams_t    *hparams,
                             cmol_tensor_t    **tensors_out,
                             int               *n_tensors_out,
                             cmol_tokenizer_t  *tokenizer) {
    (void)mmap; (void)arena; (void)hparams;
    (void)tensors_out; (void)n_tensors_out; (void)tokenizer;
    return CMOL_ERR_UNSUPPORTED;
}

cmol_tensor_t *cmol_gguf_find_tensor(cmol_tensor_t *tensors, int n,
                                      const char *name) {
    (void)tensors; (void)n; (void)name;
    return NULL;
}

cmol_err_t cmol_gguf_peek(const char     *path,
                           cmol_hparams_t *hparams_out,
                           size_t         *n_tensors_out) {
    (void)path; (void)hparams_out; (void)n_tensors_out;
    return CMOL_ERR_UNSUPPORTED;
}
