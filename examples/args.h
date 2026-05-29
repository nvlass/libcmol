/*
 * args.h — shared CLI flag parser for repl.c and oneshot.c
 *
 * parse_gen_args(argc, argv, start, &params, &system)
 *   Scans argv[start..argc-1] for --flag value pairs and fills in params.
 *   system_out may be NULL (repl uses it; oneshot ignores it).
 *   Returns 0 on success, 1 on unknown flag or missing value.
 *
 * Supported flags (all optional):
 *   -t / --temp          <float>   sampling temperature  (0 = greedy)
 *   --top-k              <int>     top-k cutoff          (0 = disabled)
 *   --top-p              <float>   nucleus cutoff        (1.0 = disabled)
 *   --seed               <uint>    RNG seed              (0 = random)
 *   --rep-pen            <float>   repetition penalty    (1.0 = disabled)
 *   --rep-n              <int>     repeat window tokens
 *   -n / --max-tokens    <int>     max tokens to generate (-1 = until EOS)
 *   --system             <string>  system prompt override (repl only)
 */

#ifndef CMOL_EXAMPLE_ARGS_H
#define CMOL_EXAMPLE_ARGS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cmol.h"

static void print_gen_args_usage(void) {
    cmol_gen_params_t d = CMOL_DEFAULT_PARAMS;
    fprintf(stderr,
        "  -t / --temp         <float>  temperature (0=greedy, default %.1f)\n"
        "       --top-k        <int>    top-k       (0=off,    default %d)\n"
        "       --top-p        <float>  top-p       (1=off,    default %.2f)\n"
        "       --seed         <uint>   RNG seed    (0=random, default 0)\n"
        "       --rep-pen      <float>  repeat penalty           (default %.1f)\n"
        "       --rep-n        <int>    repeat window            (default %d)\n"
        "  -n / --max-tokens   <int>    max new tokens (-1=EOS,  default %d)\n",
        (double)d.temperature, d.top_k, (double)d.top_p,
        (double)d.repeat_penalty, d.repeat_last_n, d.max_new_tokens);
}

static int parse_gen_args(int argc, char **argv, int start,
                           cmol_gen_params_t *p, const char **system_out) {
    int i;
    for (i = start; i < argc; i++) {
        const char *f = argv[i];
        int has_next  = (i + 1 < argc);

#define NEED_VAL(name) \
    if (!has_next) { fprintf(stderr, "%s requires a value\n", name); return 1; }

        if (!strcmp(f, "-t") || !strcmp(f, "--temp")) {
            NEED_VAL(f); p->temperature    = (float)atof(argv[++i]);
        } else if (!strcmp(f, "--top-k")) {
            NEED_VAL(f); p->top_k          = atoi(argv[++i]);
        } else if (!strcmp(f, "--top-p")) {
            NEED_VAL(f); p->top_p          = (float)atof(argv[++i]);
        } else if (!strcmp(f, "--seed")) {
            NEED_VAL(f); p->seed           = (unsigned int)atoi(argv[++i]);
        } else if (!strcmp(f, "--rep-pen")) {
            NEED_VAL(f); p->repeat_penalty = (float)atof(argv[++i]);
        } else if (!strcmp(f, "--rep-n")) {
            NEED_VAL(f); p->repeat_last_n  = atoi(argv[++i]);
        } else if (!strcmp(f, "-n") || !strcmp(f, "--max-tokens")) {
            NEED_VAL(f); p->max_new_tokens = atoi(argv[++i]);
        } else if (!strcmp(f, "--system")) {
            NEED_VAL(f);
            if (system_out) *system_out = argv[++i];
            else { fprintf(stderr, "--system not supported here\n"); return 1; }
        } else {
            fprintf(stderr, "unknown flag: %s\n", f);
            return 1;
        }

#undef NEED_VAL
    }
    return 0;
}

#endif /* CMOL_EXAMPLE_ARGS_H */
