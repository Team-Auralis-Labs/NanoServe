/* Extended Valgrind harness — cycles from VALGRIND_CYCLES env (default 1000). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void* engine_init(void);
extern int engine_infer(void* h, const char* prompt, int max_tokens, char* out, int out_len);
extern void engine_cleanup(void* h);

int main(void) {
    const char* env = getenv("VALGRIND_CYCLES");
    int cycles = env ? atoi(env) : 1000;
    char out[4096];
    for (int i = 0; i < cycles; ++i) {
        void* h = engine_init();
        if (!h) {
            fprintf(stderr, "engine_init failed at %d\n", i);
            return 1;
        }
        char prompt[64];
        snprintf(prompt, sizeof(prompt), "valgrind cycle %d", i);
        engine_infer(h, prompt, 8, out, sizeof(out));
        engine_cleanup(h);
    }
    printf("OK %d cycles\n", cycles);
    return 0;
}
