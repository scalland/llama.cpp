// Contract checks for turbo builds: session file version and a q8_0 K/V context.
//
// usage: test-turbo-session-contract <model.gguf>
// Exits with 77 (ctest SKIP) when no model path is given.
#include "common.h"
#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

static int failed = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        (void) fprintf(stderr, "FAILED: %s (%s:%d)\n", msg, __FILE__, __LINE__); \
        failed++; \
    } else { \
        (void) printf("OK: %s\n", msg); \
    } \
} while (0)

static bool decode_prompt(llama_context * ctx, const std::vector<llama_token> & tokens) {
    common_batch batch(ctx);
    for (size_t i = 0; i < tokens.size(); i++) {
        batch.add(tokens[i], (llama_pos) i, 0, i + 1 == tokens.size());
    }
    return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
}

// read the magic and version words of a session file (host byte order, same as the writer)
static bool read_header(const std::string & path, uint32_t & magic, uint32_t & version) {
    std::ifstream f(path, std::ios::in | std::ios::binary);
    if (!f) {
        return false;
    }
    f.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    f.read(reinterpret_cast<char *>(&version), sizeof(version));
    return (bool) f;
}

static llama_model * load_model(const char * path) {
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    return llama_model_load_from_file(path, mparams);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        (void) fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 77;
    }

    llama_backend_init();

    llama_model * model = load_model(argv[1]);
    if (!model) {
        (void) fprintf(stderr, "failed to load model: %s\n", argv[1]);
        llama_backend_free();
        return 1;
    }

    // (a) session files are written as the current version and load back
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 256;
    cparams.n_batch = 256;
    llama_context * ctx = llama_init_from_model(model, cparams);
    CHECK(ctx != nullptr, "f16 context created");

    if (ctx) {
        const std::vector<llama_token> prompt = { 1, 1, 1, 1 };
        CHECK(decode_prompt(ctx, prompt), "prompt decoded into the context");

        const std::string sess = "test-turbo-session-contract.bin";
        CHECK(llama_state_save_file(ctx, sess.c_str(), prompt.data(), prompt.size()), "session saved");

        uint32_t magic = 0;
        uint32_t version = 0;
        CHECK(read_header(sess, magic, version), "session header read");
        CHECK(magic == LLAMA_SESSION_MAGIC, "session magic is the current magic");
        CHECK(version == LLAMA_SESSION_VERSION, "session is written as the current version");

        llama_context * fresh = llama_init_from_model(model, cparams);
        CHECK(fresh != nullptr, "fresh context created");
        if (fresh) {
            std::vector<llama_token> out(prompt.size() + 8);
            size_t n_out = 0;
            const bool ok = llama_state_load_file(fresh, sess.c_str(), out.data(), out.size(), &n_out);
            CHECK(ok, "current-version session loads into a fresh context");
            CHECK(n_out == prompt.size(), "loaded session returns the saved tokens");
            llama_free(fresh);
        }

        (void) std::remove(sess.c_str());
        llama_free(ctx);
    }

    // (b) q8_0 K and V cache with flash attention enabled
    llama_context_params qparams = llama_context_default_params();
    qparams.n_ctx           = 256;
    qparams.n_batch         = 256;
    qparams.type_k          = GGML_TYPE_Q8_0;
    qparams.type_v          = GGML_TYPE_Q8_0;
    qparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context * qctx = llama_init_from_model(model, qparams);
    CHECK(qctx != nullptr, "q8_0 K and V context created");
    if (qctx) {
        llama_free(qctx);
    }

    llama_model_free(model);
    llama_backend_free();

    (void) printf("%d checks failed\n", failed);
    return failed != 0;
}
