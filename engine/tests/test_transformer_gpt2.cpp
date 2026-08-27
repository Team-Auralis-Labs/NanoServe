#include "transformer.hpp"
#include "nanoq_archive.hpp"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "tests/fixtures/distilgpt2-int8.nanoq";
    NanoqArchiveV3 arch;
    std::string err;
    if (!nanoq_archive_load_v3(path, arch, err)) {
        std::fprintf(stderr, "load failed: %s\n", err.c_str());
        return 1;
    }
    auto backend = create_backend(EngineBackendKind::Cpu);
    TransformerModel model(&arch, backend.get());
    if (!model.init(err)) {
        std::fprintf(stderr, "init failed: %s\n", err.c_str());
        return 1;
    }
    std::vector<uint32_t> tokens = {15496};  // "Hello" token id for GPT-2 BPE approx
    std::vector<float> logits;
    if (!model.forward_token(tokens, 0, logits)) {
        std::fprintf(stderr, "forward failed\n");
        return 1;
    }
    if (logits.empty()) {
        std::fprintf(stderr, "empty logits\n");
        return 1;
    }
    int best = 0;
    float best_v = logits[0];
    for (size_t i = 1; i < logits.size(); ++i) {
        if (logits[i] > best_v) {
            best_v = logits[i];
            best = static_cast<int>(i);
        }
    }
    std::printf("forward_ok best_token=%d logit=%.4f vocab=%zu\n", best, best_v, logits.size());

    const int max_pos = model.kv().max_seq;
    std::vector<uint32_t> long_tokens(static_cast<size_t>(max_pos + 1), tokens[0]);
    if (model.forward_token(long_tokens, max_pos, logits)) {
        std::fprintf(stderr, "expected forward_token to reject pos>=max_pos (%d)\n", max_pos);
        return 1;
    }
    std::printf("bounds_ok max_pos=%d\n", max_pos);
    return 0;
}
