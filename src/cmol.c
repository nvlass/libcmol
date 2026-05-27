/*
 * cmol.c — unity build root for libcmol
 *
 * This is the only file the build system needs to compile.
 * All source files are included here in dependency order.
 *
 * To build the library:
 *   cc -std=c99 -O2 -I include -I src -c src/cmol.c -o build/cmol.o
 *   ar rcs build/libcmol.a build/cmol.o
 *
 * Or simply: make release
 */

/* Sub-files are included in topological dependency order.
 * Each .c file includes only its own .h; include guards prevent
 * double-inclusion of the shared headers (cmol_internal.h, platform.h). */

#include "platform.c"
#include "arena.c"
#include "gguf.c"
#include "tokenizer.c"
#include "quant.c"
#include "attn.c"
#include "model.c"
#include "sampler.c"
#include "api.c"
