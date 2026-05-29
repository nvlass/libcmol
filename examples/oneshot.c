/*
 * oneshot.c — single prompt, print result, exit
 *
 * Usage:
 *   build/examples/oneshot <model.gguf> <prompt> [options]
 *
 * Options:
 *   -t / --temp         <float>  temperature (0 = greedy, default 0.8)
 *        --top-k        <int>    top-k cutoff
 *        --top-p        <float>  nucleus cutoff
 *        --seed         <uint>   RNG seed (0 = random)
 *        --rep-pen      <float>  repetition penalty (default 1.1)
 *        --rep-n        <int>    repetition window size
 *   -n / --max-tokens   <int>    max tokens to generate (-1 = until EOS)
 */

#include <stdio.h>
#include <string.h>
#include "cmol.h"
#include "args.h"

static int on_token(const char *piece, size_t len, int is_eos, void *_) {
    (void)_; (void)is_eos;
    fwrite(piece, 1, len, stdout);
    fflush(stdout);
    return 0;
}

int main(int argc, char **argv) {
    if (argc > 1 && (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h"))) {
        fprintf(stderr, "usage: %s <model.gguf> <prompt> [options]\noptions:\n",
                argv[0]);
        print_gen_args_usage();
        return 0;
    }
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <prompt> [options]\n", argv[0]);
        return 1;
    }

    cmol_config_t     cfg    = CMOL_DEFAULT_CONFIG;
    cmol_gen_params_t params = CMOL_DEFAULT_PARAMS;

    if (parse_gen_args(argc, argv, 3, &params, NULL))
        return 1;

    cmol_err_t    err;
    cmol_model_t *m = cmol_load(argv[1], &cfg, &err);
    if (!m) { fprintf(stderr, "%s\n", cmol_strerror(err)); return 1; }

    char prompt[4096];
    cmol_format_chatml(NULL, argv[2], prompt, sizeof prompt);

    cmol_session_t *s = cmol_session_acquire(m);
    cmol_err_t r = cmol_generate(s, prompt, &params, on_token, NULL);
    putchar('\n');

    cmol_session_release(s);
    cmol_free(m);
    return r == CMOL_OK ? 0 : 1;
}
