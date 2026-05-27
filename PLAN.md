# libcmol — Implementation Plan

> A minimal, embeddable C inference library for SmolLM2/SmolLM3 (135M, 360M, 1.7B).
> Single-header release artifact (`cmol_amalgam.h` via `make amalgamate`).
> Zero allocations after `cmol_load()`. Runs on x86-64, ARM, and ARMv6 (RPi Zero W).

---

## Status Summary

| Phase | Status | Description |
|-------|--------|-------------|
| 0 | ✅ Done | Scaffold: directory structure, Makefile, `include/cmol.h`, `platform.h`, `cmol_internal.h` |
| 1 | ✅ Done | `platform.c` (POSIX mmap, CPUID/NEON detection) + `arena.c` |
| 2 | ✅ Done | `gguf.c`: KV metadata, tensor descriptors, BPE tokenizer data parsing |
| 3 | ✅ Done | `tokenizer.c`: SentencePiece BPE encode/decode, byte-token fallback |
| 4 | ✅ Done | `quant.c`: Q8\_0, Q4\_K, Q6\_K dequant + matmul, AVX2/AVX-512/NEON/scalar |
| 5 | ✅ Done | `model.c` + `attn.c`: RMSNorm, RoPE, GQA, SwiGLU, KV cache, NoPE, QK-norm |
| 6 | ✅ Done | `sampler.c`: xoshiro256\*\*, greedy, temperature, top-k, top-p |
| 7 | ⬜ Next | `api.c`: `cmol_load()`, `cmol_free()`, `cmol_generate()` |
| 8 | ⬜ Pending | `tests/test_generate.c` (end-to-end), `tests/test_threads.c` (concurrent sessions) |
| 9 | ⬜ Pending | `examples/` (`repl.c`, `oneshot.c`) — compile + smoke test |
| 10 | ⬜ Pending | `tools/amalgamate.py` → `cmol_amalgam.h` single-header release |

**Test suite (all passing):** `test_gguf`=47, `test_tokenizer`=39, `test_quant`=38, `test_model`=27, `test_sampler`=36 → **187 total**

---

## Design Decisions (locked)

| Decision | Choice | Notes |
|---|---|---|
| **Name** | `libcmol` | "smol" with a C |
| **Backend** | From scratch, GGUF format | No llama.cpp dependency; reuse GGUF for model portability |
| **Quantization** | Q4\_K\_M (priority) + Q8\_0 + Q6\_K | Q4\_K\_M is the sweet spot; Q8\_0 for 135M/360M when RAM allows |
| **Weights loading** | `mmap` | Platform-abstracted; Windows stubs deferred to post-v1 |
| **Memory** | Arena at init, zero alloc after | Single `malloc` in `cmol_load()`; sized from hparams + config |
| **Threading** | Shared weights, per-session KV cache from arena | Mutex pool for session acquisition |
| **Session exhausted** | Return `NULL` + `CMOL_ERR_NO_SESSION` | Caller retries; no hidden blocking |
| **Tokenizer** | BPE loaded from GGUF, built into library | Self-contained; caller passes strings, gets strings back |
| **Context window** | Up to 8192, set at `cmol_load()` | `cfg.max_ctx`; KV cache sized accordingly |
| **Prefill chunking** | Internal, `cfg.prefill_chunk` (default 512) | Scratch sized per chunk, not full context — critical for RPi |
| **Sampler params** | Per-call via `cmol_gen_params_t` | Temperature, top-p, top-k, seed passed to `cmol_generate()` |
| **SIMD dispatch** | Runtime, via function pointers set at load | AVX-512 → AVX2 → NEON → scalar; covers ARMv6 (no NEON on RPi Zero W) |
| **Dev structure** | Separate `.c` files, unity-built | All `#include`d into `src/cmol.c` in dependency order |
| **Shipped artifact** | Single-header amalgamation | `make amalgamate` generates `cmol_amalgam.h` |

---

## Hardware Targets

| Target | CPU | SIMD | RAM notes |
|---|---|---|---|
| Modern x86-64 | AVX-512 or AVX2 | AVX-512 → AVX2 | Comfortable |
| Old x86 laptop | No AVX2 guaranteed | Scalar fallback | ~2–4 GB budget |
| RPi 3 / 4 | ARMv8, Cortex-A53/A72 | NEON/ASIMD | 1 GB; 1.7B model borderline |
| RPi Zero W | ARMv6, ARM1176 | **No NEON — scalar only** | 512 MB; 135M variant only |
| Apple Silicon | ARMv8.x | NEON/ASIMD | Comfortable |

> `cmol_load()` must compute arena size and **fail with `CMOL_ERR_OOM`** if it exceeds available RAM,
> rather than silently crashing the OS. Especially important for RPi Zero W.

---

## Public API (final shape — locked)

```c
/* ---------- config ---------- */
typedef struct {
    int   max_ctx;          // max context window (up to 8192)
    int   max_sessions;     // number of concurrent sessions to pre-allocate
    int   prefill_chunk;    // internal prefill batch size (default 512)
} cmol_config_t;

typedef struct {
    float temperature;      // 0 = greedy
    float top_p;            // nucleus sampling cutoff (0 = disabled)
    int   top_k;            // 0 = disabled; clamped to 512 internally
    int   max_new_tokens;
    unsigned int seed;      // 0 = non-deterministic
} cmol_gen_params_t;

/* ---------- lifecycle ---------- */
cmol_model_t   *cmol_load(const char *gguf_path, const cmol_config_t *cfg);
void            cmol_free(cmol_model_t *m);
const char     *cmol_strerror(cmol_err_t err);

/* ---------- sessions ---------- */
cmol_session_t *cmol_session_acquire(cmol_model_t *m);   // thread-safe; NULL on pool exhaustion
void            cmol_session_release(cmol_session_t *s);
void            cmol_session_reset(cmol_session_t *s);   // clears KV cache, keeps session slot

/* ---------- inference ---------- */
typedef void (*cmol_token_cb_t)(const char *piece, size_t len, void *userdata);

cmol_err_t cmol_generate(cmol_session_t *s,
                          const char *prompt,
                          const cmol_gen_params_t *params,
                          cmol_token_cb_t on_token,
                          void *userdata);

/* ---------- tokenizer ---------- */
int         cmol_encode(cmol_model_t *m, const char *text,
                        int32_t *out, int out_cap);       // returns n_tokens or -err
const char *cmol_decode_token(cmol_model_t *m, int32_t token_id);
```

---

## File Structure

```
libcmol/
├── include/
│   └── cmol.h                  ← public API header
├── src/
│   ├── cmol.c                  ← unity build root (#includes all below in order)
│   ├── platform.c / platform.h ← mmap abstraction (#ifdef _WIN32 stubs)
│   ├── cmol_internal.h         ← shared internal types (not in public header)
│   ├── arena.c / arena.h       ← arena allocator
│   ├── gguf.c / gguf.h         ← GGUF parser
│   ├── tokenizer.c / tokenizer.h  ← SentencePiece BPE tokenizer
│   ├── quant.c / quant.h       ← Q8_0, Q4_K, Q6_K + SIMD dispatch
│   ├── model.c / model.h       ← transformer forward pass, RMSNorm, RoPE, SwiGLU
│   ├── attn.c / attn.h         ← attention, GQA, KV cache, QK-norm, NoPE
│   ├── sampler.c / sampler.h   ← greedy / temperature / top-k / top-p, xoshiro256**
│   └── api.c                   ← public API implementation
├── tests/
│   ├── test_gguf.c             ← 47 tests ✅
│   ├── test_tokenizer.c        ← 39 tests ✅
│   ├── test_quant.c            ← 38 tests ✅
│   ├── test_model.c            ← 27 tests ✅
│   ├── test_sampler.c          ← 36 tests ✅
│   ├── test_generate.c         ← ⬜ Phase 8
│   └── test_threads.c          ← ⬜ Phase 8
├── examples/
│   ├── repl.c                  ← ⬜ Phase 9 (interactive REPL, ~35 lines)
│   ├── oneshot.c               ← ⬜ Phase 9 (single prompt, exit, ~15 lines)
│   └── concurrent.c            ← ⬜ Phase 9 (N threads, N sessions, stress demo)
├── tools/
│   └── amalgamate.py           ← ⬜ Phase 10 (generates cmol_amalgam.h)
├── CLAUDE.md                   ← development guide for Claude Code sessions
└── Makefile
```

---

## Unity Build Order (CRITICAL)

```
platform.c → arena.c → gguf.c → tokenizer.c → quant.c
           → model.c → attn.c → sampler.c → api.c
```

`model.c` **must** precede `attn.c` — `attn.c` calls static helpers defined in `model.c`
(`cmol__find_blk`, `cmol__row_bytes`, `cmol_rms_norm`, `cmol_rope_apply`, `cmol_softmax`).

---

## Phase 0 — Scaffold ✅

- [x] Create directory structure
- [x] `Makefile`: targets `debug`, `release`, `test`, `examples`, `amalgamate`, `compdb`, `fetch-smol2-*`, `fetch-smol3-*`
- [x] `include/cmol.h` — full public API (structs, signatures, error codes, callback typedefs)
- [x] `src/platform.h` / `src/platform.c` — `cmol_mmap()` / `cmol_munmap()` with POSIX impl + Windows stubs
- [x] `src/cmol_internal.h` — internal types: `cmol_dtype_t`, `cmol_tensor_t`, `cmol_hparams_t`, `cmol_kvcache_t`, `cmol_arena_t`, `cmol_kernels_t`, `cmol_model`, `cmol_session`

---

## Phase 1 — Platform + Arena ✅

- [x] `src/platform.c`: `cmol_mmap()` / `cmol_munmap()` (POSIX), `cmol_detect_cpu()` (CPUID leaf 7 on x86; mandatory NEON on AArch64; `getauxval` on ARMv7; scalar fallback on ARMv6)
- [x] `src/arena.c`: `cmol_arena_init()`, `cmol_arena_alloc()`, `cmol_arena_alloc_n()`, `cmol_arena_reset()`, `cmol_arena_remaining()`
- [x] Bump-pointer allocator with alignment; returns NULL on overflow

---

## Phase 2 — GGUF Parser ✅

- [x] Parse GGUF magic + version (v3; v1/v2 not needed for SmolLM2/3)
- [x] Parse metadata KV store: arch hparams (`n_layers`, `n_heads`, `n_kv_heads`, `d_model`, `d_ffn`, `rope_freq_base`, `rms_norm_eps`, `context_length`, `no_rope_layer_interval`, `yarn_factor_x100`)
- [x] Parse tokenizer metadata: vocab strings, scores, BPE merge rules, `bos_token_id`, `eos_token_id`, token types
- [x] Parse tensor descriptors: name → `{ dtype, shape[], rank, byte_offset }` (no data copy — offsets into mmap)
- [x] `cmol_gguf_find_tensor()` — O(log n) binary search by name
- [x] `cmol_gguf_peek()` — lightweight pre-scan for arena sizing before full parse
- [x] GGUF alignment: `GGUF_DEFAULT_ALIGNMENT = 32` bytes padding between tensors
- [x] BPE merge resolution: "piece\_a piece\_b" strings → (left\_id, right\_id) via sorted-index binary search
- [x] Dynamic arch prefix matching via `kmatch()` (handles `llama.`, `smollm.`, etc.)
- [ ] Live test with real SmolLM2/3 GGUF (gated: `CMOL_TEST_GGUF=/path/to/model.gguf`)

---

## Phase 3 — Tokenizer ✅

- [x] `cmol_tokenizer_build()`: populate hash tables + sorted merge index from GGUF data (arena-allocated)
- [x] `cmol_tokenizer_encode()`: SentencePiece BPE with `▁` leading-space convention; byte-token fallback (`<0xNN>`)
- [x] `cmol_decode_token()`: array lookup into vocab; strips `▁` → space
- [x] O(log V) vocab lookup, O(log M) merge lookup via binary search
- [ ] Live encode/decode round-trip against Python `sentencepiece` reference

---

## Phase 4 — Quantization Kernels ✅

- [x] Block structs with `__attribute__((packed))` + compile-time size assertions
  - `q8_0_block_t`: 34 bytes (fp16 scale + 32 × int8)
  - `q4_k_block_t`: 144 bytes (fp16 d+dmin + 12-byte packed scales + 128-byte 4-bit weights)
  - `q6_k_block_t`: 210 bytes (128-byte lower 4-bit + 64-byte upper 2-bit + 16 × int8 scales + fp16 d)
- [x] `cmol_dequant_row()`: dispatches to per-type dequant for all supported dtypes (F32, F16, Q8\_0, Q4\_K, Q6\_K)
- [x] `cmol_matmul_scalar()`: scalar matmul handling all types; `float tmp[256]` stack temp (no heap)
- [x] `cmol_matmul_avx2()`: AVX2+FMA, 8-float FMA with `hsum_m256()` horizontal reduction
- [x] `cmol_matmul_avx512()`: AVX-512F, 16-float FMA with `_mm512_reduce_add_ps()`
- [x] `cmol_matmul_neon()`: ARM NEON, `vmlaq_f32` + `vpadd_f32` horizontal reduction
- [x] `cmol_kernels_select()`: reads `cmol_detect_cpu()`, sets `kn.matmul` once at load time
- [x] Q4\_K scale extraction: `q4k_get_scale_min()` matching llama.cpp `get_scale_min_k4()`
- [x] Q6\_K reconstruction: `lo | (hi << 4) - 32`, range `[-32, 31]`

---

## Phase 5 — Transformer Forward Pass ✅

- [x] `cmol_rms_norm()`: `out[i] = w[i] * x[i] / sqrt(mean(x²) + eps)`; safe when `x == out`
- [x] `cmol_softmax()`: numerically stable (subtract max first)
- [x] `cmol_swiglu()`: `silu(gate[i]) * up[i]`; safe in-place on gate
- [x] `cmol_rope_apply()`: LLaMA half-rotation style; θᵢ = pos / base^(2i/d); NoPE skip when `(layer+1) % interval == 0`
- [x] `cmol_attn_forward()`: Q/K/V matmuls → optional QK-norm (SmolLM3) → RoPE/NoPE → KV cache write → GQA attention → output projection
- [x] GQA: `h_kv = h * n_kv_heads / n_heads`
- [x] QK-norm: `blk.{i}.attn_q_norm.weight` / `attn_k_norm.weight` per head; absent → skip (SmolLM2 compat)
- [x] `cmol_model_forward()`: embedding lookup → N × (RMSNorm + attn + residual + RMSNorm + FFN + residual) → final norm → LM head → logits
- [x] Tied embeddings: absent `output.weight` → fall back to `token_embd.weight`
- [x] Scratch layout: `x + xnorm + q + k_buf + v_buf + scores + attn_out + ffn_gate + ffn_up + logits`
- [x] `q` buffer reuse: repurposed as temp for `wo` output projection after RoPE/KV-write
- [ ] YARN: `yarn_factor_x100` stored in hparams; application to RoPE freq deferred to Phase 7

---

## Phase 6 — Sampler ✅

- [x] `cmol_rng_seed()`: splitmix64 expansion; seed=0 uses `time(NULL) ^ ++counter ^ (uintptr_t)state`
- [x] `cmol_rng_next()`: xoshiro256\*\* — 64-bit, period 2^256−1
- [x] `rng_f32()`: IEEE 754 mantissa trick → `[0, 1)` with no division
- [x] Greedy: `argmax(logits)` when `temperature == 0` or `params == NULL`
- [x] Temperature scaling + softmax → probabilities
- [x] Top-k: O(V log k) min-heap, `CMOL_TOPK_BUF=512` stack-allocated; `top_k > 512` silently clamped
- [x] Top-p nucleus: walk sorted top-k descending; stop when cumulative prob ≥ top\_p
- [x] Re-normalise nucleus + CDF sample; `!rng_state` → deterministic top-1 fallback
- [x] Plain categorical sampling (no top-k/top-p): CDF walk over all V

---

## Phase 7 — API Layer (`api.c`) ⬜ NEXT

### `cmol_load()`

1. `cmol_gguf_peek()` to get counts (needed for arena sizing before full parse)
2. Compute total arena bytes:
   - KV caches: `max_sessions × 2 × n_layers × n_kv_heads × d_head × max_ctx × sizeof(float)`
   - Scratch per session: `(5*d_model + 2*kv_dim + max_ctx + 2*d_ffn + vocab_size) * sizeof(float)`
   - Tokenizer tables: vocab strings + scores + merge index (from `cmol_gguf_peek()` counts)
   - Session pool: `max_sessions × sizeof(cmol_session_t)` + mutex
3. Single `malloc()` → `cmol_arena_init()` → bump-allocate all sub-regions
4. `cmol_gguf_parse()` into the arena (tensors are pointers into mmap — no copy)
5. `cmol_tokenizer_build()` on arena
6. `cmol_kernels_select()` → store in `model->kn`
7. **Fix `cmol__arena_size()`** — current formula is a placeholder; use formula from step 2 above
8. Fail with `CMOL_ERR_OOM` if computed size exceeds available system RAM (check via `sysconf(_SC_PHYS_PAGES)` on Linux / `sysctl` on macOS)

### `cmol_generate()`

```
encode(prompt) → token_ids[n_prompt]
prefill: for each chunk of prefill_chunk tokens:
    cmol_model_forward() → discard logits (KV cache fill only)
generate: loop up to max_new_tokens:
    cmol_model_forward(last_token, pos) → logits
    cmol_sample(logits, vocab_size, params, rng_state) → next_token
    cmol_decode_token(next_token) → piece
    on_token(piece, len, userdata)
    if next_token == eos_token_id: break
```

### Other tasks

- [ ] `cmol_free()`: `cmol_munmap()` + `free(arena_buf)` + `pthread_mutex_destroy`
- [ ] `cmol_session_acquire/release/reset()`: thin wrappers (stubs exist, need implementation)
- [ ] YARN: apply `yarn_factor_x100 / 100.0f` to `freq_base` in `cmol_rope_apply()`
- [ ] `cmol_arena_estimate()`: fix scratch formula (currently placeholder in `api.c`)
- [ ] Linux `-lm`: add to `LDFLAGS` in Makefile (needed for `sqrtf`, `expf` on glibc)

---

## Phase 8 — Integration Tests ⬜

- [ ] `tests/test_generate.c`: end-to-end with real GGUF; gated by `CMOL_TEST_GGUF` env var
  - Load model, acquire session, generate ≥1 token, callback fires, no crash, EOS terminates
  - Greedy mode: known prompt → assert first token matches reference
- [ ] `tests/test_threads.c`: N threads, each acquire session, generate concurrently
  - Assert no data races (TSAN), no deadlocks, output per session is independent
  - Test session pool exhaustion path (N+1 threads → one gets NULL)

---

## Phase 9 — Examples ⬜

- [ ] `examples/repl.c`: interactive REPL (~35 lines) — `cmol_load`, loop `fgets` → `cmol_generate` → print
- [ ] `examples/oneshot.c`: single prompt from `argv[2]`, print, exit (~15 lines)
- [ ] `examples/concurrent.c`: N threads, each run a session; demo + stress test
- [ ] Verify all examples compile with `make examples` and link against `build/libcmol.a`

---

## Phase 10 — Amalgamation ⬜

- [ ] `tools/amalgamate.py`:
  - Read `src/cmol.c` unity-build root; recursively expand `#include "*.c"` and `#include "*.h"` in dependency order
  - Strip redundant `#include "*.h"` after first occurrence
  - Wrap implementation in `#ifdef CMOL_IMPLEMENTATION` guard
  - Prepend `include/cmol.h` public API
  - Output: `cmol_amalgam.h` — the single-header release artifact
- [ ] `make amalgamate` generates `cmol_amalgam.h` in repo root
- [ ] Smoke test: `cc -x c -DCMOL_IMPLEMENTATION cmol_amalgam.h -o /dev/null` compiles without errors
- [ ] `cmol_amalgam.h` committed to repo; `make clean` does NOT delete it

---

## Open Questions (deferred)

- **Windows mmap**: `platform.c` has `CreateFileMapping`/`MapViewOfFile` stubs; implement when Windows target available
- **`cmol_session_acquire()` blocking variant**: A `cmol_session_acquire_wait(timeout_ms)` could be added without breaking the current API
- **Chat template**: SmolLM3 uses `<|im_start|>` / `<|im_end|>` format — a `cmol_apply_chat_template()` helper would make the REPL example more useful without adding tokenizer complexity
- **INT4 fused matmul**: Currently dequantizes to float before matmul; a fused int4 matmul kernel (especially AVX-512 VNNI) would be faster but is a later optimization
- **YARN RoPE scaling**: `yarn_factor_x100` is parsed and stored; actual frequency interpolation deferred to Phase 7

---

*Last updated: 2026-05-27*
