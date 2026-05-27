/*
 * sampler.c — token sampling
 * Implemented in Phase 6.
 * Included by src/cmol.c (unity build); do not compile standalone.
 */

#include "sampler.h"

/* Phase 6 — TODO: greedy, temperature, top-k, top-p, xoshiro256** RNG */

int32_t cmol_sample(float                   *logits,
                     int                      vocab_size,
                     const cmol_gen_params_t *params,
                     uint64_t                *rng_state) {
    (void)logits; (void)vocab_size; (void)params; (void)rng_state;
    return 0;
}

void cmol_rng_seed(uint64_t state[4], unsigned int seed) {
    (void)state; (void)seed;
}

uint64_t cmol_rng_next(uint64_t state[4]) {
    (void)state;
    return 0;
}
