/* Valgrind harness: engine init/infer/cleanup loop (no Python). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void* engine_init(void);
extern int engine_infer(void* h, const char* prompt, int max_tokens, char* out, int out_len);
extern void engine_cleanup(void* h);

int main(void) {
    char out[4096];
    for (int i = 0; i < 200; ++i) {
        void* h = engine_init();
        if (!h) { fprintf(stderr, "engine_init failed\n"); return 1; }
        char prompt[64];
        snprintf(prompt, sizeof(prompt), "valgrind cycle %d", i);
        engine_infer(h, prompt, 8, out, sizeof(out));
        engine_cleanup(h);
    }
    printf("OK 200 cycles\n");
    return 0;
}
