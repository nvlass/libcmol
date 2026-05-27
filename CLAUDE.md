# libcmol — Development Guide for Claude Code

Minimal embeddable C99 inference library for SmolLM2/SmolLM3 (135M, 360M, 1.7B).
Zero allocations after `cmol_load()`. Single-header release via `make amalgamate`.

## Build & Test

```sh
make debug           # ASAN+UBSAN debug object → build/cmol_d.o
make release         # Optimised static library → build/libcmol.a
make test            # Run all tests (links each test against cmol_d.o)
make compdb          # Regenerate compile_commands.json for clangd
make amalgamate      # Generate cmol_amalgam.h single-header

# Fetch model files (requires curl, wget, or pip install huggingface_hub)
make fetch-smol2-135m        # ~90 MB  Q4_K_M — fits RPi Zero W
make fetch-smol2-360m        # ~230 MB Q4_K_M
make fetch-smol2-1.7b        # ~1.1 GB Q4_K_M
make fetch-smol3-1.7b        # ~1.1 GB Q4_K_M  (NoPE + QK-norm)
make fetch-smol2-1.7b QUANT=Q8_0   # override quantisation level

# Run live GGUF integration tests
CMOL_TEST_GGUF=models/SmolLM2-135M-Instruct-Q4_K_M.gguf make test
```

## Unity Build Order (CRITICAL)

`src/cmol.c` includes implementation files in this exact order:

```
platform.c → arena.c → gguf.c → tokenizer.c → quant.c
           → model.c → attn.c → sampler.c → api.c
```

`model.c` **MUST** precede `attn.c`: `attn.c` calls static helpers defined in `model.c`
(`cmol__find_blk`, `cmol__row_bytes`, `cmol_rms_norm`, `cmol_rope_apply`, `cmol_softmax`).

## Phase Status

| Phase | Status | Content |
|-------|--------|---------|
| 0  | ✅ Done    | Project skeleton, Makefile, public API (`include/cmol.h`) |
| 1  | ✅ Done    | `platform.c` (POSIX mmap, CPUID/NEON detection), `arena.c` |
| 2  | ✅ Done    | `gguf.c` (KV metadata, tensor descriptors, BPE tokenizer parsing) |
| 3  | ✅ Done    | `tokenizer.c` (SentencePiece BPE encode/decode, byte tokens) |
| 4  | ✅ Done    | `quant.c` (Q8\_0, Q4\_K, Q6\_K dequant + matmul, AVX2/AVX-512/NEON/scalar) |
| 5  | ✅ Done    | `model.c` + `attn.c` (RMSNorm, RoPE, GQA, SwiGLU, KV cache, NoPE, QK-norm) |
| 6  | ✅ Done    | `sampler.c` (xoshiro256\*\*, greedy, temperature, top-k, top-p) |
| 7  | ⬜ Next    | `api.c`: `cmol_load()`, `cmol_free()`, `cmol_generate()` |
| 8  | ⬜ Pending | `tests/test_generate.c`, `tests/test_threads.c` |
| 9  | ⬜ Pending | `examples/` (`repl.c`, `oneshot.c`) compilation verification |
| 10 | ⬜ Pending | `tools/amalgamate.py` → `cmol_amalgam.h` |

**Test counts:** `test_gguf`=47, `test_tokenizer`=39, `test_quant`=38, `test_model`=27, `test_sampler`=36 → **187 total**

Run all: `make test`

## Key Architecture

### Tensor Naming (matches llama.cpp GGUF output)

```
token_embd.weight          output_norm.weight         output.weight
blk.{i}.attn_norm.weight   blk.{i}.ffn_norm.weight
blk.{i}.attn_q.weight      blk.{i}.attn_k.weight      blk.{i}.attn_v.weight
blk.{i}.attn_output.weight
blk.{i}.attn_q_norm.weight blk.{i}.attn_k_norm.weight  (SmolLM3 only, optional)
blk.{i}.ffn_gate.weight    blk.{i}.ffn_up.weight      blk.{i}.ffn_down.weight
```

GGUF shape convention: `shape[0]` = `in_features` (innermost/fastest); `shape[1]` = `out_features`.
Row `i` starts at byte offset: `i * row_bytes(in_features, dtype)`.

### Scratch Buffer (`session->scratch`)

Minimum bytes: `(5*d_model + 2*n_kv_heads*d_head + max_ctx + 2*d_ffn + vocab_size) * 4`

Layout (all `float*`, contiguous):

```
x[d]  xnorm[d]  q[d]  k_buf[kv_dim]  v_buf[kv_dim]
scores[max_ctx]  attn_out[d]  ffn_gate[d_ffn]  ffn_up[d_ffn]  logits[vocab]
```

`q[d]` is reused as a temp buffer for the `wo` output projection after RoPE/KV-write.

### KV Cache

`k` and `v` are flat `float` arrays shared across all sessions per model:

```
index = ((layer * max_tokens + pos) * n_kv_heads + h_kv) * d_head
```

### SIMD Dispatch

`cmol_kernels_select()` called once in `cmol_load()`; sets `kn->matmul` function pointer.
Priority: `avx512f` > `avx2` > `neon` > `scalar`.
Each kernel uses `__attribute__((target(...)))` — no global `-march` required at compile time.

### SmolLM3 Features

- **NoPE**: `no_rope_layer_interval > 0` → skip RoPE when `(layer+1) % interval == 0`
- **QK-norm**: `blk.{i}.attn_q_norm.weight` / `attn_k_norm.weight`, shape `[d_head]`,
  applied per-head with shared weight; absent → silently skipped (SmolLM2 compatible)
- **YARN**: `yarn_factor_x100` parsed and stored in `hparams`; not yet applied in RoPE (deferred to Phase 7)

### Quantised Block Sizes

| Type  | Bytes/block | Values/block | Layout |
|-------|-------------|--------------|--------|
| Q8\_0  | 34          | 32           | `uint16_t d` + `int8_t qs[32]` |
| Q4\_K  | 144         | 256          | `uint16_t d, dmin` + `uint8_t scales[12]` + `uint8_t qs[128]` |
| Q6\_K  | 210         | 256          | `uint8_t ql[128]` + `uint8_t qh[64]` + `int8_t scales[16]` + `uint16_t d` |

Q4\_K scale extraction: `q4k_get_scale_min(j, scales, &sc, &mn)` — 6-bit packed, matches llama.cpp `get_scale_min_k4()`.
Q6\_K value reconstruction: `lo | (hi << 4) - 32`, range `[-32, 31]`.

### Sampler

- `CMOL_TOPK_BUF = 512` — stack-allocated min-heap; `top_k > 512` is silently clamped
- `top_p` is applied **after** `top_k` — nucleus may be smaller than `top_k`
- `seed=0` → non-deterministic: `time(NULL) ^ ++counter * constant ^ (uintptr_t)state`
- `!rng_state` → deterministic top-1 fallback (no sampling)

## Phase 7 Implementation Notes

### `cmol_load()`

1. `cmol_gguf_parse()` to get hparams, tokenizer strings/scores, tensor descriptors
2. Compute total arena size: KV caches (one per session) + scratch (one per session) + session pool + tokenizer tables
3. Single `malloc()` → `cmol_arena_init()` → bump-allocate everything
4. `cmol_tokenizer_build()` on the arena
5. `cmol_kernels_select()` → store in `model->kn`
6. Return `cmol_model_t*`

**Fix needed in `api.c`:** `cmol__arena_size()` scratch formula is a placeholder — replace with:
```c
size_t scratch = (5*d + 2*kv_dim + max_ctx + 2*d_ffn + vocab) * sizeof(float);
```

### `cmol_generate()`

1. `cmol_encode()` prompt → token IDs
2. Prefill loop: feed tokens in chunks of `prefill_chunk`, calling `cmol_model_forward()` for each
3. Generation loop: `cmol_model_forward()` → `cmol_sample()` → call `on_token` callback → stop on EOS or `max_new_tokens`

## Design Constraints

- **C99 only**: `-std=c99 -Wall -Wextra -Wpedantic`; no C11/C23 features, no VLAs
- **Zero-alloc after load**: everything lives in the single arena from `cmol_load()`
- **Thread safety**: sessions protected by `pool_lock` mutex; `cmol_load()` is single-threaded
- **Concurrent load**: `qsort` globals in `tokenizer.c` / `gguf.c` are NOT thread-safe for concurrent `cmol_load()` calls — document in API
- **Max sessions**: 32 (pool bitmask fits `uint32_t`; set by `cmol_config_t.max_sessions`)
- **Linux `-lm`**: add to `LDFLAGS` in Makefile (needed for `sqrtf`, `expf`, etc.)
- **Windows mmap**: stub in `platform.c` (`CreateFileMapping`/`MapViewOfFile`); implement when targeting Windows
- **`models/` gitignored**: re-download with `make fetch-*`; `clean` target intentionally leaves models intact
