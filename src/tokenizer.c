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

/* Parse one UTF-8 codepoint from *p, advancing *p past it. */
static unsigned int utf8_next_cp(const char **p) {
    const unsigned char *u = (const unsigned char *)*p;
    unsigned int cp; int seqlen, i;
    if (!u || !*u) return 0;
    if      (u[0] < 0x80u) { cp = u[0]; seqlen = 1; }
    else if (u[0] < 0xC0u) { (*p)++; return 0xFFFDu; } /* bare continuation */
    else if (u[0] < 0xE0u) { cp = u[0] & 0x1Fu; seqlen = 2; }
    else if (u[0] < 0xF0u) { cp = u[0] & 0x0Fu; seqlen = 3; }
    else                   { cp = u[0] & 0x07u; seqlen = 4; }
    for (i = 1; i < seqlen; i++) {
        if ((u[i] & 0xC0u) != 0x80u) break; /* truncated */
        cp = (cp << 6) | (u[i] & 0x3Fu);
    }
    *p += seqlen;
    return cp;
}

/* Encode Unicode codepoint cp as UTF-8 into buf[]; return byte length (1-4). */
static int cp_to_utf8(unsigned int cp, char buf[4]) {
    if (cp < 0x80u) {
        buf[0] = (char)cp; return 1;
    } else if (cp < 0x800u) {
        buf[0] = (char)(0xC0u | (cp >> 6));
        buf[1] = (char)(0x80u | (cp & 0x3Fu)); return 2;
    } else if (cp < 0x10000u) {
        buf[0] = (char)(0xE0u | (cp >> 12));
        buf[1] = (char)(0x80u | ((cp >> 6) & 0x3Fu));
        buf[2] = (char)(0x80u | (cp & 0x3Fu)); return 3;
    } else {
        buf[0] = (char)(0xF0u | (cp >> 18));
        buf[1] = (char)(0x80u | ((cp >> 12) & 0x3Fu));
        buf[2] = (char)(0x80u | ((cp >> 6)  & 0x3Fu));
        buf[3] = (char)(0x80u | (cp & 0x3Fu)); return 4;
    }
}

/* =========================================================================
 * GPT-2 byte-to-unicode mapping
 *
 * GPT-2 represents each byte 0x00-0xFF as a single Unicode codepoint so
 * that every byte sequence has a lossless text representation:
 *
 *   "Kept" bytes (map to themselves):
 *     0x21-0x7E  (printable ASCII: '!' to '~')
 *     0xA1-0xAC  (¡ to ¬)
 *     0xAE-0xFF  (® to ÿ)
 *
 *   "Extra" bytes (68 values, mapped to U+0100..U+0143 in byte-value order):
 *     0x00-0x20  → U+0100-U+0120  (U+0120 = Ġ = space 0x20)
 *     0x7F       → U+0121
 *     0x80-0xA0  → U+0122-U+0142
 *     0xAD       → U+0143
 * ====================================================================== */

/* Input byte → GPT-2 Unicode codepoint. */
static unsigned int gpt2_byte_to_cp(unsigned char b) {
    if (b <= 0x20u)              return 0x0100u + b;          /* 0x00-0x20 */
    if (b <= 0x7Eu)              return b;                    /* 0x21-0x7E */
    if (b == 0x7Fu)              return 0x0121u;
    if (b <= 0xA0u)              return 0x0122u + (b - 0x80u); /* 0x80-0xA0 */
    if (b == 0xADu)              return 0x0143u;
    return b;                                                  /* kept */
}

/* GPT-2 Unicode codepoint → original byte (inverse of gpt2_byte_to_cp). */
static unsigned char gpt2_cp_to_byte(unsigned int cp) {
    if (cp >= 0x0100u && cp <= 0x0120u) return (unsigned char)(cp - 0x0100u);
    if (cp == 0x0121u)                  return 0x7Fu;
    if (cp >= 0x0122u && cp <= 0x0142u) return (unsigned char)(0x80u + (cp - 0x0122u));
    if (cp == 0x0143u)                  return 0xADu;
    return (unsigned char)(cp & 0xFFu); /* kept bytes pass through */
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

        /* ------------------------------------------------------------------
         * GPT-2 byte-level BPE: every vocab token is a sequence of GPT-2
         * Unicode codepoints; decode each codepoint back to the original byte.
         * Control tokens (special markers) are kept as-is.
         * ------------------------------------------------------------------ */
        if (tok->tok_model == CMOL_TOK_GPT2) {
            int is_ctrl = tok->token_type &&
                          tok->token_type[i] == CMOL_TOKEN_CONTROL;
            if (!is_ctrl) {
                /* Walk the UTF-8 string, decode each codepoint → byte. */
                char dbuf[256];
                int  dlen = 0;
                const char *p = s;
                while (*p && dlen < (int)sizeof(dbuf) - 1) {
                    unsigned int cp = utf8_next_cp(&p);
                    if (!cp) break;
                    dbuf[dlen++] = (char)gpt2_cp_to_byte(cp);
                }
                dbuf[dlen] = '\0';
                if (dlen > 0) {
                    char *ds = (char *)cmol_arena_alloc(arena,
                                                        (size_t)dlen + 1, 1);
                    if (!ds) return CMOL_ERR_OOM;
                    memcpy(ds, dbuf, (size_t)dlen + 1);
                    tok->decoded_vocab[i] = ds;
                    continue;
                }
            }
            /* Control token or empty decode → alias raw string. */
            tok->decoded_vocab[i] = s;
            continue;
        }

        /* ------------------------------------------------------------------
         * SentencePiece / Llama-style BPE
         * ------------------------------------------------------------------ */

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

    if (tok->tok_model == CMOL_TOK_GPT2) {
        /* ----------------------------------------------------------------
         * GPT-2 byte-level pre-tokenisation:
         * Iterate byte by byte, convert each to its GPT-2 Unicode char,
         * encode as UTF-8, and look up as a single-codepoint vocab token.
         *
         * Control tokens (e.g. <|im_start|>) are matched verbatim first
         * and emitted directly, bypassing the byte-level encoding.
         * ---------------------------------------------------------------- */
        const char *p = text;
        while (*p && !overflowed) {
            /* Greedy longest control-token match at current position. */
            if (tok->token_type) {
                int32_t best_id  = -1;
                size_t  best_len = 0;
                int     v;
                for (v = 0; v < tok->vocab_size; v++) {
                    if (tok->token_type[v] != CMOL_TOKEN_CONTROL) continue;
                    const char *ts = tok->vocab[v];
                    size_t tl = strlen(ts);
                    if (tl > best_len && strncmp(p, ts, tl) == 0) {
                        best_len = tl;
                        best_id  = v;
                    }
                }
                if (best_id >= 0) {
                    if (w < BPE_WORK_MAX) work[w++] = best_id;
                    else overflowed = 1;
                    p += best_len;
                    continue;
                }
            }
            /* Normal byte: map to GPT-2 Unicode codepoint and look up. */
            char vtok[5];
            int vlen = cp_to_utf8(gpt2_byte_to_cp((unsigned char)*p++), vtok);
            vtok[vlen] = '\0';
            int32_t id = find_token_id(tok, vtok);
            if (id < 0) id = tok->unk_id;
            if (id >= 0) {
                if (w < BPE_WORK_MAX) work[w++] = (int32_t)id;
                else overflowed = 1;
            }
        }
    } else {
        /* ----------------------------------------------------------------
         * SentencePiece / Llama-style pre-tokenisation:
         * Inject ▁ dummy prefix; replace spaces with ▁; look up UTF-8 chars.
         *
         * Special token handling: before each character, check if any
         * CMOL_TOKEN_CONTROL token matches verbatim at the current position.
         * If so, emit it directly (bypassing BPE) — this handles ChatML
         * delimiters like <|im_start|> and <|im_end|>.
         * ---------------------------------------------------------------- */
        const char *p = text;

        /* Whether we have injected the ▁ prefix yet (only on first real char) */
        int prefix_done = 0;

        while (*p && !overflowed) {
            /* Try control-token match (greedy longest) at current position. */
            if (tok->token_type) {
                int32_t best_id  = -1;
                size_t  best_len = 0;
                int     v;
                for (v = 0; v < tok->vocab_size; v++) {
                    if (tok->token_type[v] != CMOL_TOKEN_CONTROL) continue;
                    const char *ts = tok->vocab[v];
                    size_t tl = strlen(ts);
                    if (tl > best_len && strncmp(p, ts, tl) == 0) {
                        best_len = tl;
                        best_id  = v;
                    }
                }
                if (best_id >= 0) {
                    /* Control token matched — emit directly, reset ▁ state */
                    if (w < BPE_WORK_MAX) work[w++] = best_id;
                    else overflowed = 1;
                    p += best_len;
                    prefix_done = 0; /* next real text needs a fresh ▁ prefix */
                    continue;
                }
            }

            /* Normal character: inject ▁ prefix on first character of segment */
            if (!prefix_done && *p != '\0') {
                EMIT_CHAR(SPIECE_PREFIX, 3);
                prefix_done = 1;
            }

            if (*p == ' ') {
                /* Space → ▁ (already handles segment boundary) */
                EMIT_CHAR(SPIECE_PREFIX, 3);
                p++;
            } else {
                int len = utf8_seqlen((unsigned char)*p);
                EMIT_CHAR(p, len);
                p += len;
            }
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
