/*
 * gguf.h — GGUF file parser
 * Implemented in Phase 1.
 * Internal header — not part of the public API.
 */

#ifndef CMOL_GGUF_H
#define CMOL_GGUF_H

#include "cmol_internal.h"

/*
 * cmol_gguf_parse — parse an mmap'd GGUF file.
 *
 * Populates hparams, tokenizer, and the tensor array.
 * All variable-length data (strings, vocab, merge rules) is written into
 * `arena`; the tensor data itself stays in the mmap region (no copies).
 *
 * Returns CMOL_OK on success.
 */
cmol_err_t cmol_gguf_parse(const cmol_mmap_t *mmap,
                             cmol_arena_t      *arena,
                             cmol_hparams_t    *hparams,
                             cmol_tensor_t    **tensors_out,
                             int               *n_tensors_out,
                             cmol_tokenizer_t  *tokenizer);

/*
 * cmol_gguf_find_tensor — look up a tensor by name.
 * Returns NULL if not found.
 */
cmol_tensor_t *cmol_gguf_find_tensor(cmol_tensor_t *tensors, int n,
                                      const char *name);

/*
 * cmol_gguf_peek — read only the GGUF header and architecture metadata
 * from `path`, without a full parse or mmap.
 *
 * Used by cmol_arena_estimate() to compute memory requirements before
 * committing to a full load.  Returns CMOL_OK on success.
 */
cmol_err_t cmol_gguf_peek(const char     *path,
                           cmol_hparams_t *hparams_out,
                           size_t         *n_tensors_out);

#endif /* CMOL_GGUF_H */
