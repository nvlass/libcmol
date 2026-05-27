/*
 * sampler.c — token sampling (greedy / temperature / top-k / top-p)
 * Included by src/cmol.c (unity build); do not compile standalone.
 *
 * Pipeline for cmol_sample():
 *   temperature == 0  →  greedy argmax, done
 *   scale logits by 1/temperature
 *   softmax → probabilities
 *   top-k   →  keep up to k tokens via min-heap (O(V log k))
 *   sort k tokens descending
 *   top-p   →  find smallest nucleus with cumulative prob ≥ top_p
 *   sample  →  CDF walk over the nucleus
 *
 * When both top-k and top-p are disabled the full vocabulary is sampled
 * via a direct CDF walk over all V probabilities.
 *
 * RNG:   xoshiro256** — 64-bit, period 2^256 - 1, excellent quality.
 * Seed:  splitmix64 expansion of a 32-bit seed; seed=0 uses time(NULL).
 */

#include "sampler.h"

#include <math.h>     /* expf               */
#include <string.h>   /* memcpy             */
#include <stdlib.h>   /* qsort              */
#include <time.h>     /* time()             */
#include <stdint.h>   /* uint64_t, uintptr_t */

/* =========================================================================
 * xoshiro256** — https://prng.di.unimi.it/xoshiro256starstar.c
 * ====================================================================== */

static inline uint64_t rotl64(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

uint64_t cmol_rng_next(uint64_t s[4]) {
    uint64_t result = rotl64(s[1] * 5u, 7) * 9u;
    uint64_t t      = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3]  = rotl64(s[3], 45);

    return result;
}

/* splitmix64 — one-way expansion of a 64-bit value */
static uint64_t splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

void cmol_rng_seed(uint64_t state[4], unsigned int seed) {
    uint64_t x;
    if (seed == 0) {
        /* Non-deterministic: XOR time, a counter, and the state pointer. */
        static uint64_t g_counter = 0;
        x  = (uint64_t)time(NULL);
        x ^= ++g_counter * 6364136223846793005ULL;
        x ^= (uint64_t)(uintptr_t)state;
        if (!x) x = 0xDEADBEEFCAFEBABEULL;
    } else {
        x = (uint64_t)seed;
    }
    state[0] = splitmix64(&x);
    state[1] = splitmix64(&x);
    state[2] = splitmix64(&x);
    state[3] = splitmix64(&x);
}

/* Random float in [0, 1) using the IEEE 754 mantissa trick. */
static float rng_f32(uint64_t state[4]) {
    uint64_t r    = cmol_rng_next(state);
    uint32_t bits = 0x3F800000u | (uint32_t)(r >> 41); /* exp=127, 23 random mantissa bits */
    float    f;
    memcpy(&f, &bits, sizeof f);
    return f - 1.0f;
}

/* =========================================================================
 * Internal softmax (static — avoids cross-module dependency on model.c)
 * ====================================================================== */

static void smpl_softmax(float *x, int n) {
    float mx = x[0], sm = 0.0f;
    int i;
    for (i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    for (i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sm += x[i]; }
    sm = 1.0f / sm;
    for (i = 0; i < n; i++) x[i] *= sm;
}

/* =========================================================================
 * (probability, token_id) pair used by the top-k heap
 * ====================================================================== */

typedef struct { float val; int32_t idx; } token_prob_t;

/* ---- Min-heap helpers (root = smallest val) --------------------------- */

static void heap_sift_down(token_prob_t *h, int i, int n) {
    for (;;) {
        int s = i, l = 2*i+1, r = 2*i+2;
        if (l < n && h[l].val < h[s].val) s = l;
        if (r < n && h[r].val < h[s].val) s = r;
        if (s == i) break;
        token_prob_t tmp = h[i]; h[i] = h[s]; h[s] = tmp;
        i = s;
    }
}

/* Push a new element onto the heap (heap must have space). */
static void heap_push(token_prob_t *h, int *n, token_prob_t v) {
    int i = (*n)++;
    h[i] = v;
    while (i > 0) {
        int p = (i - 1) / 2;
        if (h[p].val <= h[i].val) break;
        token_prob_t tmp = h[i]; h[i] = h[p]; h[p] = tmp;
        i = p;
    }
}

/* Replace the root (current minimum) and restore heap. */
static void heap_replace_root(token_prob_t *h, int n, token_prob_t v) {
    h[0] = v;
    heap_sift_down(h, 0, n);
}

/* qsort comparator: descending probability */
static int cmp_prob_desc(const void *a, const void *b) {
    float va = ((const token_prob_t *)a)->val;
    float vb = ((const token_prob_t *)b)->val;
    return (va < vb) ? 1 : (va > vb) ? -1 : 0;
}

/* =========================================================================
 * Top-k selection via a min-heap of size k
 *
 * Scans probs[0..V-1] and places the k largest into out[0..k-1]
 * (in min-heap order, not sorted).  Returns actual count placed.
 * ====================================================================== */

#define CMOL_TOPK_BUF 512   /* maximum supported top-k value */

static int topk_select(const float *probs, int V, int k,
                        token_prob_t *out) {
    int n = 0, i;
    if (k > CMOL_TOPK_BUF) k = CMOL_TOPK_BUF;

    for (i = 0; i < V; i++) {
        if (n < k) {
            heap_push(out, &n, (token_prob_t){probs[i], i});
        } else if (probs[i] > out[0].val) {
            heap_replace_root(out, k, (token_prob_t){probs[i], i});
        }
    }
    return n;
}

/* =========================================================================
 * cmol_sample
 * ====================================================================== */

int32_t cmol_sample(float                   *logits,
                     int                      vocab_size,
                     const cmol_gen_params_t *params,
                     uint64_t                *rng_state) {

    int i;
    if (!logits || vocab_size <= 0) return 0;

    /* ── Greedy (temperature == 0 or no params) ──────────────────────── */
    if (!params || params->temperature == 0.0f) {
        int32_t best = 0;
        for (i = 1; i < vocab_size; i++)
            if (logits[i] > logits[best]) best = i;
        return best;
    }

    /* ── Temperature scaling ──────────────────────────────────────────── */
    {
        float inv_t = 1.0f / params->temperature;
        for (i = 0; i < vocab_size; i++) logits[i] *= inv_t;
    }

    /* ── Softmax → probabilities (in-place) ────────────────────────────── */
    smpl_softmax(logits, vocab_size);

    /* ── Decide whether to use the heap path ──────────────────────────── */
    int use_topk = (params->top_k > 0 && params->top_k < vocab_size);
    int use_topp = (params->top_p > 0.0f && params->top_p < 1.0f);

    if (!use_topk && !use_topp) {
        /* ── Plain categorical sampling: CDF walk over all probs ─────── */
        if (!rng_state) {
            /* No RNG provided — return the argmax of probabilities */
            int32_t best = 0;
            for (i = 1; i < vocab_size; i++)
                if (logits[i] > logits[best]) best = i;
            return best;
        }
        float r = rng_f32(rng_state), acc = 0.0f;
        for (i = 0; i < vocab_size - 1; i++) {
            acc += logits[i];
            if (r < acc) return (int32_t)i;
        }
        return (int32_t)(vocab_size - 1);
    }

    /* ── Top-k selection ─────────────────────────────────────────────── */
    token_prob_t buf[CMOL_TOPK_BUF];
    int k = use_topk ? params->top_k : vocab_size; /* cap applied inside */
    int n = topk_select(logits, vocab_size, k, buf);

    /* Sort descending (required for nucleus scan) */
    qsort(buf, (size_t)n, sizeof(token_prob_t), cmp_prob_desc);

    /* ── Top-p nucleus ───────────────────────────────────────────────── */
    int nucleus = n;
    if (use_topp) {
        float cum = 0.0f;
        for (i = 0; i < n; i++) {
            cum += buf[i].val;
            if (cum >= params->top_p) { nucleus = i + 1; break; }
        }
        /* nucleus stays n if we never exceeded top_p (sum of top-k < top_p) */
    }

    /* ── Re-normalise and sample ─────────────────────────────────────── */
    float sum = 0.0f;
    for (i = 0; i < nucleus; i++) sum += buf[i].val;

    /* Guard: no RNG or degenerate distribution → deterministic top-1 */
    if (!rng_state || sum <= 0.0f) return buf[0].idx;

    float r = rng_f32(rng_state) * sum, acc = 0.0f;
    for (i = 0; i < nucleus; i++) {
        acc += buf[i].val;
        if (r < acc) return buf[i].idx;
    }
    return buf[nucleus - 1].idx;   /* rounding/fp safety */
}
