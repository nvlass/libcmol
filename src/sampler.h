/*
 * sampler.h — token sampling (greedy / temperature / top-k / top-p)
 * Implemented in Phase 6.
 * Internal header — not part of the public API.
 */

#ifndef CMOL_SAMPLER_H
#define CMOL_SAMPLER_H

#include "cmol_internal.h"

/*
 * cmol_sample — sample a token ID from a logit vector.
 *
 * `logits` is modified in-place (temperature scaling, softmax).
 * `rng_state` is updated in-place; seed it once with cmol_rng_seed().
 *
 * With temperature == 0.0 the result is always argmax (greedy).
 */
int32_t cmol_sample(float                   *logits,
                     int                      vocab_size,
                     const cmol_gen_params_t *params,
                     uint64_t                *rng_state);

/* ---- xoshiro256** RNG ------------------------------------------------- */

/*
 * cmol_rng_seed — initialise the 256-bit state from a 32-bit seed.
 * seed == 0 uses a time-based seed for non-deterministic output.
 */
void cmol_rng_seed(uint64_t state[4], unsigned int seed);

/* cmol_rng_next — return the next 64-bit value and advance the state. */
uint64_t cmol_rng_next(uint64_t state[4]);

/*
 * cmol_apply_repeat_penalty — discount logits of recently-seen tokens.
 *
 * Applies the standard llama.cpp repetition penalty formula in-place,
 * before temperature scaling and softmax:
 *   logit > 0  →  logit /= penalty
 *   logit ≤ 0  →  logit *= penalty
 *
 * `tokens`   — ring buffer of the last N generated/prompt tokens (IDs).
 *              Any entry with id < 0 or id >= vocab_size is skipped.
 * `n_tokens` — number of valid entries in `tokens` (≤ CMOL_REPEAT_BUF).
 * `penalty`  — multiplier; values ≤ 1.0f are a no-op.
 */
#define CMOL_REPEAT_BUF 128

void cmol_apply_repeat_penalty(float *logits, int vocab_size,
                                const int32_t *tokens, int n_tokens,
                                float penalty);

#endif /* CMOL_SAMPLER_H */
