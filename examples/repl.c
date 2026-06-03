/*
 * repl.c — interactive multi-turn chat REPL using libcmol
 *
 * Usage:
 *   build/examples/repl <model.gguf> [options]
 *
 * Options:
 *   -t / --temp         <float>  temperature (0 = greedy, default 0.8)
 *        --top-k        <int>    top-k cutoff
 *        --top-p        <float>  nucleus cutoff
 *        --seed         <uint>   RNG seed (0 = random)
 *        --rep-pen      <float>  repetition penalty (default 1.1)
 *        --rep-n        <int>    repetition window size
 *   -n / --max-tokens   <int>    max tokens per reply (-1 = until EOS)
 *        --system       <str>    override system prompt ("" = omit)
 */

#include <stdio.h>
#include <string.h>
#include "cmol.h"
#include "args.h"

static int on_token(const char *piece, size_t len, int is_eos, void *_) {
    (void)_;
    if (!is_eos) fwrite(piece, 1, len, stdout);
    else         putchar('\n');
    fflush(stdout);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2 || !strcmp(argv[1], "--help") || !strcmp(argv[1], "-h")) {
        fprintf(stderr, "usage: %s <model.gguf> [options]\noptions:\n", argv[0]);
        print_gen_args_usage();
        fprintf(stderr, "       --system  <str>   system prompt override\n");
        return argc < 2 ? 1 : 0;
    }

    cmol_config_t     cfg    = CMOL_DEFAULT_CONFIG;
    cmol_gen_params_t params = CMOL_DEFAULT_PARAMS;
    const char       *system = NULL;  /* NULL = omit system turn */

    if (parse_gen_args(argc, argv, 2, &params, &system))
        return 1;

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

    printf("libcmol %s  —  type a prompt, empty line to quit\n", cmol_version());
    if (params.temperature == 0.0f)
        printf("(greedy decoding)\n");
    printf("\n");

    while (printf("> "), fflush(stdout), fgets(line, sizeof line, stdin)) {
        line[strcspn(line, "\n")] = '\0';
        if (!*line) break;

        /* First turn: full ChatML (system turn included if --system was given).
           Later turns: append user+assistant headers only;
           KV cache holds the prior conversation. */
        if (turn == 0)
            cmol_format_chatml(system, line, prompt, sizeof prompt);
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
