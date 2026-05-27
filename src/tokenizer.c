/*
 * tokenizer.c — BPE tokenizer (SentencePiece variant for SmolLM2/3)
 * Included by src/cmol.c (unity build); do not compile standalone.
 *
 * Supports CMOL_TOK_LLAMA (SentencePiece-style BPE), used by SmolLM2 and
 * SmolLM3.  CMOL_TOK_GPT2 (byte-level GPT-2 BPE) is stubbed for Phase 3
 * and will be fully implemented if needed for future model variants.
 *
 * Algorithm summary:
 *   1. Pre-tokenize: inject ▁ dummy prefix; replace ' ' with ▁; look up
 *      each UTF-8 character in the vocab (byte fallback on miss).
 *   2. BPE merge loop: O(w²) greedy scan; practical for w ≤ BPE_WORK_MAX.
 *   3. cmol_tokenizer_build() constructs three index arrays so that both
 *      string→ID and pair→rank lookups run in O(log n).
 */

#include "tokenizer.h"
#include "arena.h"     /* cmol_arena_alloc, cmol_arena_alloc_n */

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

/* =========================================================================
 * qsort comparator state — valid only during cmol_tokenizer_build().
 *
 * Not thread-safe for concurrent cmol_load() calls.  In practice, model
 * loading is single-threaded; document this limitation.
 * ====================================================================== */

static const char   **g_bld_vocab;   /* vocab pointer for vocab sort      */
static const int32_t *g_bld_ml;      /* merge_left  for merge sort        */
static const int32_t *g_bld_mr;      /* merge_right for merge sort        */

/* Sort vocab indices by the string they point to. */
static int cmp_vocab_idx(const void *a, const void *b) {
    int32_t ia = *(const int32_t *)a;
    int32_t ib = *(const int32_t *)b;
    return strcmp(g_bld_vocab[ia], g_bld_vocab[ib]);
}

/* Sort merge indices by packed (left<<32 | right). */
static int cmp_merge_idx(const void *a, const void *b) {
    int32_t ia = *(const int32_t *)a;
    int32_t ib = *(const int32_t *)b;
    uint64_t ka = ((uint64_t)(uint32_t)g_bld_ml[ia] << 32) | (uint32_t)g_bld_mr[ia];
    uint64_t kb = ((uint64_t)(uint32_t)g_bld_ml[ib] << 32) | (uint32_t)g_bld_mr[ib];
    if (ka < kb) return -1;
    if (ka > kb) return  1;
    return 0;
}

/* =========================================================================
 * Binary search helpers
 * ====================================================================== */

/*
 * find_token_id — return the vocab index whose string equals `s`, or -1.
 * Requires tok->vocab_sort_idx to be populated by cmol_tokenizer_build().
 */
static int32_t find_token_id(const cmol_tokenizer_t *tok, const char *s) {
    if (!tok->vocab_sort_idx || tok->vocab_size == 0) return -1;
    int lo = 0, hi = tok->vocab_size - 1;
    while (lo <= hi) {
        int     mid = (lo + hi) >> 1;
        int32_t idx = tok->vocab_sort_idx[mid];
        int     cmp = strcmp(tok->vocab[idx], s);
        if (cmp == 0) return idx;
        if (cmp  < 0) lo = mid + 1;
        else          hi = mid - 1;
    }
    return -1;
}

/*
 * find_merge_rank — return the rank (original merge index = priority) of
 * the merge rule (L, R), or -1 if the pair has no merge rule.
 * Rank 0 = highest priority (applied first).
 */
static int32_t find_merge_rank(const cmol_tokenizer_t *tok,
                                int32_t L, int32_t R) {
    if (!tok->msort_idx || tok->n_merges == 0 || L < 0 || R < 0) return -1;
    uint64_t target = ((uint64_t)(uint32_t)L << 32) | (uint32_t)R;
    int lo = 0, hi = tok->n_merges - 1;
    while (lo <= hi) {
        int     mid = (lo + hi) >> 1;
        int32_t idx = tok->msort_idx[mid];
        uint64_t key = ((uint64_t)(uint32_t)tok->merge_left[idx]  << 32)
                     |              (uint32_t)tok->merge_right[idx];
        if (key == target) return idx;   /* idx IS the priority rank */
        if (key  < target) lo = mid + 1;
        else               hi = mid - 1;
    }
    return -1;
}

/* =========================================================================
 * cmol_tokenizer_build
 * ====================================================================== */

cmol_err_t cmol_tokenizer_build(cmol_tokenizer_t *tok, cmol_arena_t *arena) {
    if (!tok || !arena)           return CMOL_ERR_ARGS;
    if (tok->vocab_size == 0)     return CMOL_ERR_INVALID;

    int V = tok->vocab_size;
    int M = tok->n_merges;

    /* ------------------------------------------------------------------
     * 1. vocab_sort_idx[V] — sorted by vocab string for string→ID lookup.
     * ------------------------------------------------------------------ */
    tok->vocab_sort_idx =
        (int32_t *)cmol_arena_alloc_n(arena, (size_t)V, sizeof(int32_t));
    if (!tok->vocab_sort_idx) return CMOL_ERR_OOM;

    for (int i = 0; i < V; i++) tok->vocab_sort_idx[i] = (int32_t)i;
    g_bld_vocab = tok->vocab;
    qsort(tok->vocab_sort_idx, (size_t)V, sizeof(int32_t), cmp_vocab_idx);

    /* ------------------------------------------------------------------
     * 2. msort_idx[M] — sorted by pack64(left, right) for pair→rank.
     * ------------------------------------------------------------------ */
    if (M > 0) {
        tok->msort_idx =
            (int32_t *)cmol_arena_alloc_n(arena, (size_t)M, sizeof(int32_t));
        if (!tok->msort_idx) return CMOL_ERR_OOM;

        for (int i = 0; i < M; i++) tok->msort_idx[i] = (int32_t)i;
        g_bld_ml = tok->merge_left;
        g_bld_mr = tok->merge_right;
        qsort(tok->msort_idx, (size_t)M, sizeof(int32_t), cmp_merge_idx);
    }

    /* ------------------------------------------------------------------
     * 3. merge_result[M] — result token ID for each merge rule.
     *    Concatenate vocab[left] + vocab[right], look up in sorted vocab.
     * ------------------------------------------------------------------ */
    if (M > 0) {
        tok->merge_result =
            (int32_t *)cmol_arena_alloc_n(arena, (size_t)M, sizeof(int32_t));
        if (!tok->merge_result) return CMOL_ERR_OOM;

        char concat[256];
        for (int i = 0; i < M; i++) {
            int32_t L = tok->merge_left[i];
            int32_t R = tok->merge_right[i];
            if (L < 0 || L >= V || R < 0 || R >= V) {
                tok->merge_result[i] = tok->unk_id;
                continue;
            }
            const char *ls = tok->vocab[L];
            const char *rs = tok->vocab[R];
            size_t ll = strlen(ls), rl = strlen(rs);
            if (ll + rl >= sizeof concat) {
                tok->merge_result[i] = tok->unk_id;
                continue;
            }
            memcpy(concat, ls, ll);
            memcpy(concat + ll, rs, rl);
            concat[ll + rl] = '\0';
            int32_t rid = find_token_id(tok, concat);
            tok->merge_result[i] = (rid >= 0) ? rid : tok->unk_id;
        }
    }

    /* ------------------------------------------------------------------
     * 4. decoded_vocab[V] — output form of each token:
     *      <0xNN> byte tokens    → single-byte string  "\xNN"
     *      ▁ (U+2581) prefix     → replace ▁ with ' '
     *      everything else       → point directly at vocab[i]
     * ------------------------------------------------------------------ */
    tok->decoded_vocab =
        (const char **)cmol_arena_alloc_n(arena, (size_t)V, sizeof(char *));
    if (!tok->decoded_vocab) return CMOL_ERR_OOM;

    for (int i = 0; i < V; i++) {
        const char *s = tok->vocab[i];

        /* Byte token?  Prefer the token_type flag; fall back to heuristic. */
        int is_byte = tok->token_type &&
                      (tok->token_type[i] == CMOL_TOKEN_BYTE);
        if (!is_byte) {
            /* Heuristic: exactly "<0x??>" — 6 chars */
            int n = (int)strlen(s);
            if (n == 6 && s[0]=='<' && s[1]=='0' && s[2]=='x' && s[5]=='>')
                is_byte = 1;
        }
        if (is_byte) {
            unsigned bv = 0;
            if (sscanf(s, "<0x%02x>", &bv) == 1) {
                char *ds = (char *)cmol_arena_alloc(arena, 2, 1);
                if (!ds) return CMOL_ERR_OOM;
                ds[0] = (char)(unsigned char)bv;
                ds[1] = '\0';
                tok->decoded_vocab[i] = ds;
                continue;
            }
        }

        /* ▁ prefix (U+2581 = 0xE2 0x96 0x81)? */
        if ((unsigned char)s[0] == 0xE2 &&
            (unsigned char)s[1] == 0x96 &&
            (unsigned char)s[2] == 0x81) {
            size_t rest = strlen(s + 3);
            char *ds = (char *)cmol_arena_alloc(arena, rest + 2, 1);
            if (!ds) return CMOL_ERR_OOM;
            ds[0] = ' ';
            if (rest) memcpy(ds + 1, s + 3, rest);
            ds[rest + 1] = '\0';
            tok->decoded_vocab[i] = ds;
            continue;
        }

        /* No transformation needed — alias original string. */
        tok->decoded_vocab[i] = s;
    }

    return CMOL_OK;
}

/* =========================================================================
 * UTF-8 helpers
 * ====================================================================== */

/* Returns the byte length of the UTF-8 sequence that starts with byte `c`. */
static int utf8_seqlen(unsigned char c) {
    if ((c & 0x80) == 0x00) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1; /* invalid / continuation byte — treat as 1 */
}

/* =========================================================================
 * cmol_tokenizer_encode
 * ====================================================================== */

/*
 * Maximum number of initial tokens (before merges) that the working buffer
 * can hold.  One token per UTF-8 byte in the worst case (all byte fallback).
 * Covers prompts up to ~4 KB; O(w²) BPE merge is fast enough for w ≤ 4096.
 */
#define BPE_WORK_MAX 4096

/* UTF-8 encoding of ▁ (U+2581 LOWER ONE EIGHTH BLOCK — SP word boundary) */
static const char SPIECE_PREFIX[4] = "\xe2\x96\x81";  /* 3 bytes + NUL */

int cmol_tokenizer_encode(const cmol_tokenizer_t *tok,
                           const char             *text,
                           int32_t                *out,
                           int                     out_cap,
                           int                     add_bos) {
    if (!tok || !text || !out || out_cap <= 0) return CMOL_ERR_ARGS;
    if (!tok->vocab_sort_idx || !tok->decoded_vocab)
        return CMOL_ERR_UNSUPPORTED; /* build() not called */

    int n = 0;          /* tokens written to `out`          */
    int overflowed = 0; /* BPE work buffer overflowed       */

    /* Optional BOS special token */
    if (add_bos && tok->bos_id >= 0) {
        if (n >= out_cap) return CMOL_ERR_TRUNC;
        out[n++] = tok->bos_id;
    }

    /* ----------------------------------------------------------------
     * Pre-tokenisation — build the initial token sequence in `work[]`.
     *
     * For CMOL_TOK_LLAMA (SentencePiece):
     *   • Inject one ▁ dummy prefix before the first input character.
     *   • Replace each ASCII space with ▁.
     *   • Look up each resulting UTF-8 character in the vocab via
     *     binary search.  On miss, emit one <0xNN> byte token per byte.
     *
     * For CMOL_TOK_GPT2: stub — treat like LLAMA for now.
     * ---------------------------------------------------------------- */
    int32_t work[BPE_WORK_MAX];
    int     w = 0;

    /* Emit one character (given as pointer + byte length) into work[]. */
    /* On vocab miss: emit byte fallback tokens. */
#define EMIT_CHAR(ptr, len)  do {                                       \
    char _cb[8];                                                        \
    memcpy(_cb, (ptr), (size_t)(len));                                  \
    _cb[(len)] = '\0';                                                  \
    int32_t _id = find_token_id(tok, _cb);                              \
    if (_id >= 0) {                                                     \
        if (w < BPE_WORK_MAX) work[w++] = _id;                         \
        else overflowed = 1;                                            \
    } else {                                                            \
        int _b;                                                         \
        for (_b = 0; _b < (len) && !overflowed; _b++) {                \
            char _bt[8];                                                \
            snprintf(_bt, sizeof _bt, "<0x%02x>",                      \
                     (unsigned char)(ptr)[_b]);                         \
            int32_t _bid = find_token_id(tok, _bt);                    \
            if (_bid < 0) _bid = tok->unk_id;                          \
            if (_bid >= 0) {                                            \
                if (w < BPE_WORK_MAX) work[w++] = _bid;                \
                else overflowed = 1;                                    \
            }                                                           \
        }                                                               \
    }                                                                   \
} while (0)

    const char *p = text;

    /* Inject ▁ dummy prefix for SentencePiece models (non-empty text). */
    if (tok->tok_model == CMOL_TOK_LLAMA && *p != '\0') {
        EMIT_CHAR(SPIECE_PREFIX, 3);
    }

    while (*p && !overflowed) {
        if (*p == ' ') {
            EMIT_CHAR(SPIECE_PREFIX, 3);
            p++;
        } else {
            int len = utf8_seqlen((unsigned char)*p);
            EMIT_CHAR(p, len);
            p += len;
        }
    }

#undef EMIT_CHAR

    /* ----------------------------------------------------------------
     * BPE merge loop
     *
     * Greedy O(w²) scan: find the adjacent pair with the lowest merge
     * rank (highest priority), apply it, repeat until no merges remain.
     * ---------------------------------------------------------------- */
    if (tok->n_merges > 0 && tok->merge_result && tok->msort_idx) {
        while (w > 1) {
            int     best_pos  = -1;
            int32_t best_rank = INT32_MAX;

            for (int i = 0; i < w - 1; i++) {
                int32_t rank = find_merge_rank(tok, work[i], work[i + 1]);
                if (rank >= 0 && rank < best_rank) {
                    best_rank = rank;
                    best_pos  = i;
                }
            }
            if (best_pos < 0) break; /* no mergeable pair remains */

            /* Replace pair with the merged token, shift array left. */
            work[best_pos] = tok->merge_result[best_rank];
            for (int i = best_pos + 1; i < w - 1; i++)
                work[i] = work[i + 1];
            w--;
        }
    }

    /* ----------------------------------------------------------------
     * Write merged tokens to `out`.
     * ---------------------------------------------------------------- */
    int trunc = overflowed;
    for (int i = 0; i < w; i++) {
        if (n >= out_cap) { trunc = 1; break; }
        out[n++] = work[i];
    }

    return trunc ? CMOL_ERR_TRUNC : n;
}

/* =========================================================================
 * cmol_tokenizer_decode_token
 * ====================================================================== */

const char *cmol_tokenizer_decode_token(const cmol_tokenizer_t *tok,
                                         int32_t                 token_id) {
    if (!tok || token_id < 0 || token_id >= tok->vocab_size) return NULL;
    /* Prefer decoded form (▁→space, byte tokens expanded). */
    if (tok->decoded_vocab) return tok->decoded_vocab[token_id];
    /* Fallback: raw vocab string (build() not yet called). */
    return tok->vocab ? tok->vocab[token_id] : NULL;
}
