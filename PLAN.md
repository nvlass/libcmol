# libcmol — Implementation Plan

> A minimal, embeddable C inference library for SmolLM3 and its smaller variants.
> Single-header release artifact (`cmol.h` with `#ifdef CMOL_IMPLEMENTATION`).
> Zero allocations after `cmol_load()`. Runs on x86-64, ARM, and ARMv6 (RPi Zero W).

---

## Design Decisions (locked)

| Decision | Choice | Notes |
|---|---|---|
| **Name** | `libcmol` | "smol" with a C — mirrors the C-shell origin of the project |
| **Backend** | From scratch, GGUF format | No llama.cpp dependency; reuse GGUF for model portability |
| **Quantization** | Q4_K_M (priority) + Q8_0 | Q4_K_M is the sweet spot; Q8_0 for 135M/360M when RAM allows |
| **Weights loading** | `mmap` | Platform-abstracted; Windows stubs deferred |
| **Memory** | Arena at init, zero alloc after | Single `malloc` in `cmol_load()`; sized from config |
| **Threading** | Shared weights, per-session KV cache from arena | Mutex pool for session acquisition |
| **Session exhausted** | Return `NULL` + `CMOL_ERR_NO_SESSION` | Caller retries; no hidden blocking |
| **Tokenizer** | BPE loaded from GGUF, included in library | Self-contained; caller passes strings, gets strings back |
| **Context window** | Up to 8192, set at `cmol_load()` | `cfg.max_ctx`; KV cache sized accordingly |
| **Prefill chunking** | In scope for v1, fully internal | `cfg.prefill_chunk` (default 512); scratch sized per chunk, not full context. Critical for RPi. |
| **Sampler params** | Per-call via `cmol_gen_params_t` | Temperature, top-p, top-k, seed passed to `cmol_generate()` |
| **SIMD dispatch** | Runtime, via function pointers set at load | AVX-512 → AVX2 → NEON → scalar. Covers ARMv6 (no NEON on RPi Zero W) |
| **Dev structure** | Separate `.c` files, unity-built | All `#include`d into `csmol.c` for compilation |
| **Shipped artifact** | Single-header amalgamation | `make amalgamate` generates release `cmol.h` |

---

## Hardware Targets

| Target | CPU | SIMD | RAM notes |
|---|---|---|---|
| Modern x86-64 | AVX-512 or AVX2 | AVX-512 → AVX2 | Comfortable |
| Old x86 laptop | No AVX2 guaranteed | Scalar fallback | ~2–4 GB budget |
| RPi 3 / 4 | ARMv8, Cortex-A53/A72 | NEON/ASIMD | 1 GB; 1.7B model borderline |
| RPi Zero W | ARMv6, ARM1176 | **No NEON — scalar only** | 512 MB; 135M variant only |
| Apple Silicon | ARMv8.x | NEON/ASIMD | Comfortable |

> `cmol_load()` must compute arena size and **fail with a clear error** if it exceeds available RAM,
> rather than silently OOM-crashing the OS. Especially important for RPi Zero W.

---

## Public API (target shape)

```c
/* ---------- config ---------- */
typedef struct {
    int   max_ctx;          // max context window (up to 8192)
    int   max_sessions;     // number of concurrent sessions to pre-allocate
    int   prefill_chunk;    // internal prefill batch size (default 512); not exposed further
} cmol_config_t;

typedef struct {
    float temperature;      // 0 = greedy
    float top_p;            // nucleus sampling cutoff
    int   top_k;            // 0 = disabled
    int   max_new_tokens;
    unsigned int seed;
} cmol_gen_params_t;

/* ---------- lifecycle ---------- */
cmol_model_t   *cmol_load(const char *gguf_path, const cmol_config_t *cfg);
void            cmol_free(cmol_model_t *m);
const char     *cmol_strerror(cmol_err_t err);

/* ---------- sessions ---------- */
cmol_session_t *cmol_session_acquire(cmol_model_t *m);   // thread-safe; NULL on pool exhaustion
void            cmol_session_release(cmol_session_t *s);
void            cmol_session_reset(smol_session_t *s);   // clears KV cache, keeps session slot

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

A REPL becomes:

```c
cmol_model_t   *m = cmol_load("smollm3-1.7b-q4_k_m.gguf", &cfg);
cmol_session_t *s = cmol_session_acquire(m);
char line[512];
while (fgets(line, sizeof line, stdin))
    cmol_generate(s, line, &params, print_cb, NULL);
```

---

## File Structure

```
libcmol/
├── include/
│   └── cmol.h                  ← public API header
├── src/
│   ├── cmol.c                  ← unity build root (#includes all below)
│   ├── platform.h              ← mmap abstraction (#ifdef _WIN32 stubs)
│   ├── cmol_internal.h         ← shared internal types
│   ├── gguf.c / gguf.h         ← GGUF parser
│   ├── arena.c / arena.h       ← arena allocator + session pool
│   ├── tokenizer.c / tokenizer.h  ← BPE tokenizer
│   ├── quant.c / quant.h       ← Q4_K_M + Q8_0 kernels + SIMD dispatch
│   ├── model.c / model.h       ← transformer forward pass
│   ├── attn.c / attn.h         ← attention, RoPE, GQA, KV cache
│   ├── sampler.c / sampler.h   ← greedy / temperature / top-p / top-k
│   └── api.c                   ← public API implementation
├── tests/
│   ├── test_gguf.c
│   ├── test_tokenizer.c
│   ├── test_quant.c
│   ├── test_forward.c
│   ├── test_generate.c
│   └── test_threads.c
├── examples/
│   ├── repl.c                  ← interactive REPL (~35 lines)
│   ├── oneshot.c               ← single prompt, exit (~15 lines)
│   └── concurrent.c            ← N threads, N sessions (stress demo)
├── tools/
│   └── amalgamate.py           ← generates single-header release artifact
└── Makefile
```

---

## Dependency Order

```
platform.h ──→ gguf.c ──→ arena.c ──→ quant.c
                    │                       │
                    └──→ tokenizer.c         └──→ model.c ──→ attn.c ──→ sampler.c ──→ api.c
```

Phases 1–3 (GGUF, arena, tokenizer) can be built and tested independently.
Phase 4 (quant) unblocks Phase 5 (model/attention). Everything converges at Phase 7 (API).

---

## Phase 0 — Scaffold

- [ ] Create directory structure above
- [ ] Write `Makefile`: targets `debug`, `release`, `test`, `amalgamate`
- [ ] Write `cmol.h` — full public API (structs, signatures, error codes, callback typedefs)
  - This is the design document; lock it before implementation starts
- [ ] Write `platform.h` — `cmol_mmap()` / `cmol_munmap()` with POSIX impl + Windows stubs
- [ ] Write `cmol_internal.h` — internal types not in the public header

---

## Phase 1 — GGUF Parser (`gguf.c`)

- [ ] Parse GGUF magic + version (target v3; v1/v2 compatibility optional)
- [ ] Parse metadata key-value store:
  - [ ] Architecture hyperparams: `n_layers`, `n_heads`, `n_kv_heads` (GQA), `d_model`, `d_ffn`, `rope_freq_base`, `rms_norm_eps`, `context_length`
  - [ ] Tokenizer metadata: vocab list, BPE merge rules, `bos_token_id`, `eos_token_id`, token types
- [ ] Parse tensor descriptors: name → `{ dtype, shape[], rank, byte_offset }`
  - Do **not** copy tensor data — record offsets into the mmap region
- [ ] `cmol_gguf_find_tensor(ctx, name)` — lookup by name, returns pointer into mmap
- [ ] `mmap` the whole file at load time; `munmap` on free
- [ ] ⚠️ Respect `GGUF_DEFAULT_ALIGNMENT` (32 bytes) padding between tensors
- [ ] Test: parse a real SmolLM3 GGUF, assert expected tensor names/shapes/dtypes

---

## Phase 2 — Arena Allocator (`arena.c`)

- [ ] `arena_init(buf, size)` — wraps a pre-allocated block
- [ ] `arena_alloc(arena, size, align)` — bump pointer with alignment; returns NULL on overflow
- [ ] Arena layout carved at `cmol_load()`:
  ```
  [ session_0 KV cache ][ session_1 KV cache ] ...
  [ session_0 scratch  ][ session_1 scratch  ] ...
  [ shared temp / decode buffer               ]
  ```
- [ ] Compute arena size formula:
  `n_sessions × (kv_cache_bytes + scratch_bytes) + temp_buf_bytes`
  - `kv_cache_bytes = 2 × n_layers × n_kv_heads × head_dim × max_ctx × sizeof(float)`
  - `scratch_bytes` sized for one prefill chunk of `cfg.prefill_chunk` tokens
- [ ] Fail in `cmol_load()` with `CMOL_ERR_OOM` if required arena exceeds system RAM
- [ ] Session pool: fixed array of `max_sessions` slots + `pthread_mutex_t` + free bitmask
- [ ] `cmol_session_acquire()` — lock, find free slot, mark taken, unlock; return NULL if none free
- [ ] `cmol_session_release()` — lock, mark slot free, unlock

---

## Phase 3 — Tokenizer (`tokenizer.c`)

- [ ] Load vocab (token strings + scores) from GGUF metadata into arena
- [ ] Load BPE merge rules from GGUF into arena — no heap
- [ ] `cmol_encode(text → token_ids)`:
  - [ ] Pre-tokenize: whitespace/punctuation split, leading-space convention
  - [ ] Byte fallback for unknown characters (`<0xNN>` tokens)
  - [ ] BPE merge loop: repeatedly apply highest-priority merge
  - [ ] Prepend BOS token if configured
- [ ] `cmol_decode_token(id → string)` — array lookup into vocab
- [ ] `cmol_decode(ids → text)` — concatenate pieces, strip leading-space artifacts
- [ ] ⚠️ This phase has the most edge cases — BPE corner cases, mixed-script, byte tokens
- [ ] Test: encode→decode round-trip lossless for ASCII; compare against Python `gguf` reference

---

## Phase 4 — Quantization Kernels (`quant.c`)

### Q8_0 (implement first — simpler)
- [ ] Block layout: 32 values → 1× fp16 scale + 32× int8 = 34 bytes/block (8.5 bpw effective)
- [ ] `dequantize_q8_0_row(blocks, float *out, n)`
- [ ] `dot_q8_0(a_blocks, b_float, n)` — dequant a, dot with float b

### Q4_K_M
- [ ] Block layout: 256 values → 2× fp16 (scale+min) + 12 bytes scales (6-bit packed) + 128 bytes weights = 144 bytes/block (4.5 bpw effective)
- [ ] `dequantize_q4_k_row(blocks, float *out, n)`
- [ ] `dot_q4_k(a_blocks, b_float, n)`

### SIMD dispatch
- [ ] `typedef float (*matmul_fn_t)(...)` — function pointer for hot inner loop
- [ ] Set at `cmol_load()` time via `__cpuid` (x86) / `getauxval(AT_HWCAP)` (ARM Linux) / compile-time fallback
- [ ] Implement: `_avx512`, `_avx2`, `_neon`, `_scalar`
- [ ] ⚠️ RPi Zero W (ARMv6): no NEON — scalar must be correct and will be the only path
- [ ] Only matmul inner loops need SIMD; RMSNorm, RoPE, SwiGLU are cheap scalar
- [ ] Test: dequantize known blocks, compare against Python `gguf` library reference values

---

## Phase 5 — Transformer Forward Pass (`model.c`, `attn.c`)

### Tensor ops (float buffers in scratch space)
- [ ] `rms_norm(x, weight, eps, n)` — no bias
- [ ] `rope_apply(q, k, pos, n_heads, head_dim, freq_base)` — in-place RoPE on Q and K
- [ ] `swiglu(gate, up, out, n)` — SiLU(gate) × up
- [ ] `softmax(x, n)`
- [ ] `matmul(out, a, b, m, k, n)` — dispatches via SIMD function pointer

### Attention (`attn.c`)
- [ ] Project input → Q, K, V (quantized matmul via quant layer)
- [ ] Apply RoPE to Q and K
- [ ] Write K and V into per-session KV cache at current position
- [ ] GQA: each KV head shared by `n_heads / n_kv_heads` Q heads
- [ ] Compute attention scores → softmax → weighted sum of V
- [ ] Output projection

### Full forward pass (`model.c`)
- [ ] Embedding lookup: token_id → `d_model` float vector
- [ ] Loop `n_layers`:
  - [ ] RMSNorm → Attention → residual add
  - [ ] RMSNorm → FFN (SwiGLU) → residual add
- [ ] Final RMSNorm → LM head → logits over vocab
- [ ] ⚠️ Tied embeddings: 135M/360M variants share LM head with embedding table — check for `output.weight` in GGUF; fall back to `token_embd.weight` if absent
- [ ] Prefill chunking (internal): loop over prompt in `cfg.prefill_chunk`-sized batches; scratch sized per chunk

---

## Phase 6 — Sampler (`sampler.c`)

- [ ] Greedy: `argmax(logits)`
- [ ] Temperature scaling: `logits[i] /= temp` before softmax
- [ ] Top-k: zero out all but top-k logits before sampling
- [ ] Top-p (nucleus): sort by probability, cumulative cutoff, sample from survivors
- [ ] RNG: xoshiro256** — small, fast, no stdlib dependency, seed from `cmol_gen_params_t`
- [ ] All ops on logit buffer in scratch space — no allocation

---

## Phase 7 — API Layer (`api.c`)

- [ ] `cmol_load()`:
  - Parse GGUF → extract hyperparams + tokenizer metadata
  - Compute arena size → single `malloc` → lay out arena
  - Detect SIMD capabilities → set function pointers
  - Initialize session pool (mutex + bitmask)
  - Return populated `cmol_model_t *`
- [ ] `cmol_free()`: `munmap` + `free(arena)` + `pthread_mutex_destroy`
- [ ] `cmol_session_acquire/release/reset()`: thin wrappers over arena pool
- [ ] `cmol_generate()`:
  ```
  encode prompt → token_ids
  prefill: loop chunks → forward pass (KV cache fill, no sampling)
  generate: loop {
      forward pass → logits
      sample (cmol_gen_params_t) → next_token
      decode_token → on_token callback
      stop if EOS or max_new_tokens reached
  }
  ```
- [ ] `cmol_err_t` return on every function; `cmol_strerror()` for human-readable messages

---

## Phase 8 — Tests (`tests/`)

- [ ] `test_gguf.c` — parse real SmolLM3 GGUF, assert tensor names/shapes/dtypes
- [ ] `test_tokenizer.c` — encode/decode round-trips; known edge cases (BPE, byte tokens, non-ASCII)
- [ ] `test_quant.c` — dequant correctness vs. Python reference values
- [ ] `test_forward.c` — single forward pass; compare logits against `llama.cpp` on same model + prompt
- [ ] `test_generate.c` — full end-to-end; assert no crash, terminates on EOS, output is non-empty
- [ ] `test_threads.c` — N threads, each acquire session, generate concurrently; valgrind/TSAN clean

---

## Phase 9 — Examples (`examples/`)

- [ ] `repl.c` — interactive REPL, ~35 lines
- [ ] `oneshot.c` — single prompt from argv, print, exit, ~15 lines
- [ ] `concurrent.c` — N threads, each run a session; demo + stress test

---

## Phase 10 — Amalgamation (`tools/amalgamate.py`)

- [ ] Topological sort of source files by dependency order
- [ ] Concatenate, stripping internal `#include "*.h"` directives
- [ ] Wrap in `#ifdef CMOL_IMPLEMENTATION` guard
- [ ] Prepend `cmol.h` public API
- [ ] Output: single `cmol.h` — the release artifact
- [ ] Add `make amalgamate` target; generated file committed to repo root

---

## Open Questions (deferred)

- **Windows mmap**: `platform.h` has stubs; implement `CreateFileMapping` / `MapViewOfFile` when Windows access is available
- **`cmol_session_acquire()` blocking variant**: A `cmol_session_acquire_wait(timeout_ms)` could be added without breaking the current API
- **Chat template**: SmolLM3 has a specific prompt format (`<|im_start|>` etc.) — a `cmol_apply_template()` helper would make the REPL example more useful
- **INT4 fused matmul**: Currently the plan dequantizes to float before matmul; a fused int4 matmul kernel would be faster but is a later optimization

---

*Last updated: 2026-05-27*
