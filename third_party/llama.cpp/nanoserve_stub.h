# llama.cpp kernel subset placeholder for Phase 01.
# Full kernel vendoring deferred; native ops in engine/src/transformer_gpt2.cpp.

#ifndef NANOSERVE_LLAMA_CPP_STUB_H
#define NANOSERVE_LLAMA_CPP_STUB_H
static inline int nanoserve_llama_kernels_enabled(void) { return 1; }
#endif
