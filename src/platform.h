/*
 * platform.h — OS and CPU abstraction layer for libcmol
 *
 * Provides:
 *   - Memory-mapped file I/O  (POSIX mmap / Windows stubs)
 *   - CPU feature detection   (x86 CPUID / ARM HWCAP / AArch64 mandatory)
 *
 * Internal header — not part of the public API.
 */

#ifndef CMOL_PLATFORM_H
#define CMOL_PLATFORM_H

#include <stddef.h>
#include "../include/cmol.h"

/* =========================================================================
 * Memory-mapped file
 * ====================================================================== */

typedef struct {
    void  *data;   /* start of mapped region (read-only)  */
    size_t size;   /* total size in bytes                  */
} cmol_mmap_t;

/*
 * cmol_mmap_open — open `path` and map it into read-only memory.
 * On success, populates *out and returns CMOL_OK.
 * The file descriptor is closed before returning (mapping keeps the data).
 */
cmol_err_t cmol_mmap_open(const char *path, cmol_mmap_t *out);

/*
 * cmol_mmap_close — unmap a previously opened region.
 * Safe to call with a zeroed or already-closed cmol_mmap_t.
 */
void cmol_mmap_close(cmol_mmap_t *m);

/* =========================================================================
 * CPU feature detection
 *
 * Results are used in cmol_kernels_select() (quant.c) to set the SIMD
 * function pointer table once at cmol_load() time.
 *
 * Targets:
 *   x86-64   — CPUID leaf 7 subleaf 0 (AVX2 / AVX-512F)
 *   AArch64  — ASIMD is mandatory; neon is always 1
 *   ARMv7    — Linux AT_HWCAP / HWCAP_NEON
 *   ARMv6    — no NEON (Raspberry Pi Zero W); all fields 0
 *   Other    — all fields 0; scalar fallback used
 * ====================================================================== */

typedef struct {
    int avx512f; /* x86: AVX-512 Foundation                */
    int avx2;    /* x86: Advanced Vector Extensions 2      */
    int neon;    /* ARM: NEON (ARMv7) / ASIMD (AArch64)    */
} cmol_cpu_t;

/*
 * cmol_detect_cpu — query hardware capabilities.
 * Safe to call multiple times; result is idempotent.
 */
cmol_cpu_t cmol_detect_cpu(void);

#endif /* CMOL_PLATFORM_H */
