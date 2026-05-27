/*
 * tokenizer.h — BPE tokenizer (loaded from GGUF metadata)
 * Implemented in Phase 3.
 * Internal header — not part of the public API.
 */

#ifndef CMOL_TOKENIZER_H
#define CMOL_TOKENIZER_H

#include "cmol_internal.h"

/*
 * cmol_tokenizer_encode — convert UTF-8 text to token IDs via BPE.
 *
 * Writes up to `out_cap` token IDs into `out`.
 * Returns token count (>= 0) on success, or a negative cmol_err_t.
 * Returns CMOL_ERR_TRUNC if out_cap was too small (partial result written).
 *
 * A BOS token is prepended when `add_bos` is non-zero.
 */
int cmol_tokenizer_encode(const cmol_tokenizer_t *tok,
                           const char *text,
                           int32_t    *out,
                           int         out_cap,
                           int         add_bos);

/*
 * cmol_tokenizer_decode_token — return the UTF-8 string for one token ID.
 * Pointer into the tokenizer's vocab (arena-owned); do not free.
 * Returns NULL for out-of-range IDs.
 */
const char *cmol_tokenizer_decode_token(const cmol_tokenizer_t *tok,
                                         int32_t token_id);

#endif /* CMOL_TOKENIZER_H */
