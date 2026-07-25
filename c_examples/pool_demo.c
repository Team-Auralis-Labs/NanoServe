/* Minimal C consumer of the Rust buddy allocator.
 *
 * Demonstrates the safety pattern discussed in the design notes: a pool
 * handle wrapped in GCC/Clang's __attribute__((cleanup)) behaves like a
 * Rust/C++ destructor -- it is released automatically when the variable
 * goes out of scope, even on an early `return`, so you never have to
 * remember a matching manual free call.
 *
 * Build:  gcc -std=c17 -O2 -o pool_demo pool_demo.c \
 *             -L../allocator/target/release -lbuddy_alloc -lpthread -ldl
 * Run:    LD_LIBRARY_PATH=../allocator/target/release ./pool_demo
 */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

extern void* pool_create(size_t size);
extern void* pool_acquire(const void* pool);
extern void* pool_allocate(void* pool, size_t req_size);
extern void  pool_free(void* pool, void* ptr, size_t size);
extern void  pool_release(void* pool);

#if defined(__GNUC__) || defined(__clang__)
#define POOL_AUTO __attribute__((cleanup(pool_auto_release)))
static inline void pool_auto_release(void** p) {
    if (*p) { pool_release(*p); *p = NULL; }
}
#else
#define POOL_AUTO
#endif

int main(void) {
    /* handle is released automatically on scope exit -- no leak possible */
    void* pool POOL_AUTO = pool_create(1024 * 1024); /* 1 MiB */
    if (!pool) { fprintf(stderr, "pool_create failed\n"); return 1; }

    uint8_t* buf = pool_allocate(pool, 256);
    if (!buf) { fprintf(stderr, "allocation failed\n"); return 1; }
    memset(buf, 0xAB, 256);
    printf("Allocated 256 bytes, buf[0] = 0x%02X\n", buf[0]);
    pool_free(pool, buf, 256);

    /* a second, independent owner of the SAME arena (refcounted via Arc) */
    void* pool2 POOL_AUTO = pool_acquire(pool);
    uint8_t* buf2 = pool_allocate(pool2, 128);
    printf("Second handle allocated from the same arena: %p\n", (void*)buf2);
    pool_free(pool2, buf2, 128);

    printf("Leaving scope: both handles released, arena freed once last one drops.\n");
    return 0;
}
