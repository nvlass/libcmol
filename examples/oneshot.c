/*
 * oneshot.c — single prompt, print result, exit
 *
 * Usage:
 *   build/examples/oneshot path/to/model.gguf "Your prompt here"
 */

#include <stdio.h>
#include "cmol.h"

static int on_token(const char *piece, size_t len, int is_eos, void *_) {
    (void)_; (void)is_eos;
    fwrite(piece, 1, len, stdout);
    fflush(stdout);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <prompt>\n", argv[0]);
        return 1;
    }

    cmol_config_t    cfg    = CMOL_DEFAULT_CONFIG;
    cmol_gen_params_t params = CMOL_DEFAULT_PARAMS;

    cmol_err_t    err;
    cmol_model_t *m = cmol_load(argv[1], &cfg, &err);
    if (!m) { fprintf(stderr, "%s\n", cmol_strerror(err)); return 1; }

    cmol_session_t *s = cmol_session_acquire(m);
    cmol_err_t r = cmol_generate(s, argv[2], &params, on_token, NULL);
    putchar('\n');

    cmol_session_release(s);
    cmol_free(m);
    return r == CMOL_OK ? 0 : 1;
}
