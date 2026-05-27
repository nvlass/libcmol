/*
 * platform.c — OS abstraction implementation
 * Included by src/cmol.c (unity build); do not compile standalone.
 */

#include "platform.h"

/* =========================================================================
 * Memory-mapped file
 * ====================================================================== */

#ifdef _WIN32
/* ---- Windows (stub — implement when Windows target is available) ------- */
#include <windows.h>

cmol_err_t cmol_mmap_open(const char *path, cmol_mmap_t *out) {
    (void)path; (void)out;
    /* TODO: CreateFile → CreateFileMapping → MapViewOfFile */
    return CMOL_ERR_UNSUPPORTED;
}

void cmol_mmap_close(cmol_mmap_t *m) {
    if (!m || !m->data) return;
    /* TODO: UnmapViewOfFile(m->data); CloseHandle(...) */
    m->data = NULL;
    m->size = 0;
}

#else
/* ---- POSIX --------------------------------------------------------------- */
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

cmol_err_t cmol_mmap_open(const char *path, cmol_mmap_t *out) {
    if (!path || !out) return CMOL_ERR_ARGS;

    int fd = open(path, O_RDONLY);
    if (fd < 0) return CMOL_ERR_IO;

    struct stat st;
    if (fstat(fd, &st) < 0) { close(fd); return CMOL_ERR_IO; }

    void *data = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd); /* fd can be closed immediately; mapping keeps the data alive */
    if (data == MAP_FAILED) return CMOL_ERR_IO;

    /* Hint to the OS: we'll read sequentially during the parse pass */
    madvise(data, (size_t)st.st_size, MADV_SEQUENTIAL);

    out->data = data;
    out->size = (size_t)st.st_size;
    return CMOL_OK;
}

void cmol_mmap_close(cmol_mmap_t *m) {
    if (!m || !m->data) return;
    munmap(m->data, m->size);
    m->data = NULL;
    m->size = 0;
}

#endif /* _WIN32 */

/* =========================================================================
 * CPU feature detection
 * ====================================================================== */

/* ---- x86 / x86-64 ------------------------------------------------------- */
#if defined(__x86_64__) || defined(_M_X64)

static void cmol__cpuid(unsigned int leaf, unsigned int subleaf,
                         unsigned int *eax, unsigned int *ebx,
                         unsigned int *ecx, unsigned int *edx) {
#if defined(_MSC_VER)
    int regs[4];
    __cpuidex(regs, (int)leaf, (int)subleaf);
    *eax = (unsigned int)regs[0]; *ebx = (unsigned int)regs[1];
    *ecx = (unsigned int)regs[2]; *edx = (unsigned int)regs[3];
#else
    __asm__ volatile(
        "cpuid"
        : "=a"(*eax), "=b"(*ebx), "=c"(*ecx), "=d"(*edx)
        : "a"(leaf), "c"(subleaf)
    );
#endif
}

cmol_cpu_t cmol_detect_cpu(void) {
    cmol_cpu_t cpu = {0};
    unsigned int eax, ebx, ecx, edx;

    /* Leaf 0: highest supported leaf */
    cmol__cpuid(0, 0, &eax, &ebx, &ecx, &edx);
    if (eax < 7) return cpu; /* no extended feature flags */

    /* Leaf 7, subleaf 0: extended features */
    cmol__cpuid(7, 0, &eax, &ebx, &ecx, &edx);
    cpu.avx2    = (ebx >> 5)  & 1; /* EBX bit  5: AVX2        */
    cpu.avx512f = (ebx >> 16) & 1; /* EBX bit 16: AVX-512F    */

    return cpu;
}

/* ---- AArch64 (Apple Silicon, RPi 3/4 64-bit) ----------------------------- */
#elif defined(__aarch64__)

cmol_cpu_t cmol_detect_cpu(void) {
    /* ASIMD (NEON equivalent) is architecturally mandatory on AArch64. */
    cmol_cpu_t cpu = {0};
    cpu.neon = 1;
    return cpu;
}

/* ---- ARMv7 (NEON optional — check at runtime via AT_HWCAP) --------------- */
#elif defined(__arm__) || defined(__ARM_ARCH)

#if defined(__linux__) && __has_include(<sys/auxv.h>)
#  include <sys/auxv.h>
#  ifndef HWCAP_NEON
#    define HWCAP_NEON (1 << 12)
#  endif
#endif

cmol_cpu_t cmol_detect_cpu(void) {
    cmol_cpu_t cpu = {0};
#if defined(__ARM_NEON)
    /* Compiled with -mfpu=neon: safe to assume NEON is present. */
    cpu.neon = 1;
#elif defined(__linux__) && __has_include(<sys/auxv.h>)
    unsigned long hwcap = getauxval(AT_HWCAP);
    cpu.neon = (hwcap & HWCAP_NEON) != 0;
#endif
    /* ARMv6 (RPi Zero W): __ARM_NEON not defined, getauxval returns 0
     * for HWCAP_NEON → cpu.neon stays 0 → scalar fallback selected. */
    return cpu;
}

/* ---- Everything else (scalar fallback) ----------------------------------- */
#else

cmol_cpu_t cmol_detect_cpu(void) {
    cmol_cpu_t cpu = {0};
    return cpu;
}

#endif
