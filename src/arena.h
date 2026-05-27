/*
 * arena.h — bump-pointer arena allocator
 * Internal header — not part of the public API.
 */

#ifndef CMOL_ARENA_H
#define CMOL_ARENA_H

#include "cmol_internal.h"

/* Initialise an arena over an already-allocated buffer. */
void cmol_arena_init(cmol_arena_t *a, void *buf, size_t size);

/*
 * Allocate `size` bytes aligned to `align` from the arena.
 * `align` must be a power of two.
 * Returns NULL if the arena is exhausted.
 */
void *cmol_arena_alloc(cmol_arena_t *a, size_t size, size_t align);

/* Helper: allocate an array of `count` elements of `elem_size` bytes. */
void *cmol_arena_alloc_n(cmol_arena_t *a, size_t count, size_t elem_size);

/* Reset the arena, making all memory available again (does not zero). */
void cmol_arena_reset(cmol_arena_t *a);

/* Return the number of bytes still available. */
size_t cmol_arena_remaining(const cmol_arena_t *a);

#endif /* CMOL_ARENA_H */
