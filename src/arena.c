/*
 * arena.c — bump-pointer arena allocator implementation
 * Included by src/cmol.c (unity build); do not compile standalone.
 */

#include "arena.h"
#include <string.h>

void cmol_arena_init(cmol_arena_t *a, void *buf, size_t size) {
    a->base = (uint8_t *)buf;
    a->size = size;
    a->used = 0;
}

void *cmol_arena_alloc(cmol_arena_t *a, size_t size, size_t align) {
    /* align must be a power of two */
    size_t start = (a->used + align - 1u) & ~(align - 1u);
    if (start + size > a->size) return NULL;
    void *ptr = a->base + start;
    a->used = start + size;
    memset(ptr, 0, size); /* zero on allocation — predictable behaviour */
    return ptr;
}

void *cmol_arena_alloc_n(cmol_arena_t *a, size_t count, size_t elem_size) {
    /* Check for overflow before multiplying */
    if (count && elem_size > (size_t)-1 / count) return NULL;
    return cmol_arena_alloc(a, count * elem_size, elem_size < 8 ? elem_size : 8);
}

void cmol_arena_reset(cmol_arena_t *a) {
    a->used = 0;
}

size_t cmol_arena_remaining(const cmol_arena_t *a) {
    return a->size - a->used;
}
