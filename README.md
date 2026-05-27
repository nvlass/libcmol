# libcmol

Minimal, embeddable C inference library for [SmolLM3](https://huggingface.co/HuggingFaceTB/SmolLM3) and its smaller variants (135M, 360M, 1.7B).

## Goals

- Single-header release artifact — drop `cmol.h` into any C project
- Zero allocations after `cmol_load()` — fully arena-based
- GGUF model format — compatible with the broader llama.cpp ecosystem
- Runs on x86-64 (AVX2/AVX-512), ARM (NEON), and ARMv6 (Raspberry Pi Zero W — scalar)
- Thread-safe: shared weights, per-session KV cache, mutex session pool

## Quick start (REPL in ~10 lines)

```c
#define CMOL_IMPLEMENTATION
#include "cmol.h"

static int print_token(const char *p, size_t n, int eos, void *_) {
    if (!eos) fwrite(p, 1, n, stdout);
    return 0;
}

int main(void) {
    cmol_config_t cfg = CMOL_DEFAULT_CONFIG;
    cmol_model_t *m = cmol_load("smollm3.gguf", &cfg, NULL);
    cmol_session_t *s = cmol_session_acquire(m);
    cmol_gen_params_t p = CMOL_DEFAULT_PARAMS;
    char line[512];
    while (fgets(line, sizeof line, stdin))
        cmol_generate(s, line, &p, print_token, NULL);
    cmol_session_release(s);
    cmol_free(m);
}
```

## Building

```sh
make debug     # debug build with ASAN/UBSAN
make release   # optimised build + static library (build/libcmol.a)
make test      # build and run all tests
make amalgamate # generate single-header cmol_amalgam.h
```

## Status

Under active development. See [PLAN.md](PLAN.md) for the implementation roadmap.
