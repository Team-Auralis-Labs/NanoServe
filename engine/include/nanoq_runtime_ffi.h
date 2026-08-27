#pragma once
#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

int nanoq_archive_validate_path(const char* path);
int nanoq_archive_validate(const uint8_t* data, size_t len);

void* nanoq_tokenizer_create(const uint8_t* data, size_t len);
void nanoq_tokenizer_destroy(void* handle);
int nanoq_tokenizer_encode(void* handle, const char* text, uint32_t* out_ids, size_t max_ids);
char* nanoq_tokenizer_decode(void* handle, const uint32_t* ids, size_t num_ids);
void nanoq_string_free(char* s);
int nanoq_tokenizer_vocab_size(void* handle);

#ifdef __cplusplus
}
#endif
