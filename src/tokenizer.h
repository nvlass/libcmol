/*
 * tokenizer.h — BPE tokenizer (SentencePiece variant)
 *
 * Internal header — not part of the public API.
 * Consumers: tokenizer.c (implementation), api.c, tests/test_tokenizer.c.
 *
 * Life-cycle:
 *   1. cmol_gguf_parse()        — fills raw fields (vocab, merge_left/right)
 *   2. cmol_tokenizer_build()   — builds runtime lookup tables in arena
 *   3. cmol_tokenizer_encode()  — BPE encode UTF-8 → token IDs
 *   4. cmol_tokenizer_decode_token() — token ID → UTF-8 piece
 */

#ifndef CMOL_TOKENIZER_H
#define CMOL_TOKENIZER_H

/* cmol_arena_t is defined in cmol_internal.h */
#include "cmol_internal.h"

/* =========================================================================
 * cmol_tokenizer_build
 *
 * Builds the four runtime lookup tables that encode/decode require:
 *
 *   decoded_vocab[vocab_size]   — decoded form of each token:
 *                                   ▁ prefix  → leading space
 *                                   <0xNN>    → single byte value
 *                                   other     → original vocab string
 *
 *   vocab_sort_idx[vocab_size]  — indices into vocab[], sorted by string;
 *                                 enables O(log V) string→token-ID lookup.
 *
 *   merge_result[n_merges]      — result token ID for merge rule i:
 *                                 vocab[merge_left[i]] + vocab[merge_right[i]]
 *                                 looked up via vocab_sort_idx.
 *
 *   msort_idx[n_merges]         — indices into merge_left/right[], sorted by
 *                                 pack64(left, right); enables O(log M)
 *                                 pair→rank (= merge priority) lookup.
 *
 * All allocations come from `arena`.  Must be called once before encode.
 * Returns CMOL_OK on success, CMOL_ERR_OOM / CMOL_ERR_INVALID on failure.
 * ====================================================================== */
cmol_err_t cmol_tokenizer_build(cmol_tokenizer_t *tok, cmol_arena_t *arena);

/* =========================================================================
 * cmol_tokenizer_encode
 *
 * BPE-encodes null-terminated UTF-8 `text` to token IDs.
 *
 * Pre-tokenization (CMOL_TOK_LLAMA / SentencePiece):
 *   - Inject one ▁ dummy prefix before the first character.
 *   - Replace each ' ' (space) with ▁.
 *   - Look up each UTF-8 character in the vocab; on miss emit byte-token
 *     fallback (<0xNN>) for each byte of the character.
 *
 * `add_bos`: prepend tok->bos_id as the first output token when non-zero.
 *
 * Returns the number of tokens written (≥ 0), or a negative cmol_err_t:
 *   CMOL_ERR_ARGS        — NULL pointer or out_cap ≤ 0
 *   CMOL_ERR_UNSUPPORTED — cmol_tokenizer_build() not yet called
 *   CMOL_ERR_TRUNC       — out_cap too small (partial result written)
 * ====================================================================== */
int cmol_tokenizer_encode(const cmol_tokenizer_t *tok,
                           const char             *text,
                           int32_t                *out,
                           int                     out_cap,
                           int                     add_bos);

/* =========================================================================
 * cmol_tokenizer_decode_token
 *
 * Returns the UTF-8 string for a single token ID.
 *
 * The returned pointer is into the arena (or into the original vocab string
 * literal for unchanged tokens) and is valid for the lifetime of the model.
 * Do not free.  Returns NULL for out-of-range IDs.
 *
 * If cmol_tokenizer_build() was not yet called, falls back to the raw
 * vocab string (▁ not converted, byte tokens not expanded).
 * ====================================================================== */
const char *cmol_tokenizer_decode_token(const cmol_tokenizer_t *tok,
                                         int32_t                 token_id);

#endif /* CMOL_TOKENIZER_H */
