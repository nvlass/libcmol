/*
 * repl.c — minimal interactive REPL using libcmol
 *
 * Usage:
 *   build/examples/repl path/to/model.gguf
 */

#include <stdio.h>
#include <string.h>
#include "cmol.h"

static int on_token(const char *piece, size_t len, int is_eos, void *_) {
    (void)_;
    if (!is_eos) fwrite(piece, 1, len, stdout);
    else         putchar('\n');
    fflush(stdout);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    cmol_config_t    cfg    = CMOL_DEFAULT_CONFIG;
    cmol_gen_params_t params = CMOL_DEFAULT_PARAMS;

    cmol_err_t    err;
    cmol_model_t *m = cmol_load(argv[1], &cfg, &err);
    if (!m) {
        fprintf(stderr, "cmol_load: %s\n", cmol_strerror(err));
        return 1;
    }

    cmol_session_t *s = cmol_session_acquire(m);

    char line[2048];
    char prompt[4096];
    int  turn = 0;
    printf("libcmol %s  —  type a prompt, empty line to quit\n\n",
           cmol_version());

    while (printf("> "), fflush(stdout), fgets(line, sizeof line, stdin)) {
        line[strcspn(line, "\n")] = '\0';
        if (!*line) break;

        /* Wrap in ChatML — first turn includes system message, later turns
           append only the new user+assistant markers (KV cache holds context). */
        if (turn == 0)
            cmol_format_chatml(NULL, line, prompt, sizeof prompt);
        else
            cmol_format_chatml_turn(line, prompt, sizeof prompt);
        turn++;

        cmol_err_t r = cmol_generate(s, prompt, &params, on_token, NULL);
        if (r != CMOL_OK)
            fprintf(stderr, "generate: %s\n", cmol_strerror(r));
    }

    cmol_session_release(s);
    cmol_free(m);
    return 0;
}
