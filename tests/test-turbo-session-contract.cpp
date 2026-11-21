// Contract checks for turbo builds: session file version and a q8_0 K/V context.
//
// usage: test-turbo-session-contract [--save <session> | --load <session>] <model.gguf>
// Exits with 77 (ctest SKIP) when no model path is given.
// --save writes a session from the model, --load checks a session file loads back.
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

static const std::vector<llama_token> k_prompt = { 1, 1, 1, 1 };

static llama_context_params f16_params() {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 256;
    cparams.n_batch = 256;
    return cparams;
}

// write a session file from the model, used by the baseline build
static int run_save(llama_model * model, const char * path) {
    llama_context * ctx = llama_init_from_model(model, f16_params());
    if (!ctx) {
        (void) fprintf(stderr, "FAILED: f16 context not created\n");
        return 1;
    }
    const bool ok = decode_prompt(ctx, k_prompt) &&
        llama_state_save_file(ctx, path, k_prompt.data(), k_prompt.size());
    llama_free(ctx);
    if (!ok) {
        (void) fprintf(stderr, "FAILED: session not saved: %s\n", path);
        return 1;
    }
    (void) printf("saved session: %s\n", path);
    return 0;
}

// load a session file into a fresh context and check the saved tokens
static int run_load(llama_model * model, const char * path) {
    llama_context * ctx = llama_init_from_model(model, f16_params());
    if (!ctx) {
        (void) fprintf(stderr, "FAILED: f16 context not created\n");
        return 1;
    }
    std::vector<llama_token> out(k_prompt.size() + 8);
    size_t n_out = 0;
    const bool ok = llama_state_load_file(ctx, path, out.data(), out.size(), &n_out);
    llama_free(ctx);

    CHECK(ok, "session file loads into a fresh f16 context");
    CHECK(n_out == k_prompt.size(), "loaded session returns the saved token count");
    CHECK(n_out <= out.size() && std::equal(k_prompt.begin(), k_prompt.end(), out.begin()),
          "loaded session returns the saved tokens");
    (void) printf("%d checks failed\n", failed);
    return failed != 0;
}

// VmRSS in kB from /proc/self/status, -1 if unavailable
static long read_vmrss_kb() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            return std::strtol(line.c_str() + 6, nullptr, 10);
        }
    }
    return -1;
}

// one save and load round with the given cache types; false on any failure
static bool soak_round(llama_model * model, ggml_type type_k, ggml_type type_v) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = 256;
    cparams.n_batch         = 256;
    cparams.type_k          = type_k;
    cparams.type_v          = type_v;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

    const std::string sess = "test-turbo-soak.bin";
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        return false;
    }
    bool ok = decode_prompt(ctx, k_prompt) &&
        llama_state_save_file(ctx, sess.c_str(), k_prompt.data(), k_prompt.size());
    llama_free(ctx);

    if (ok) {
        llama_context * fresh = llama_init_from_model(model, cparams);
        ok = fresh != nullptr;
        if (fresh) {
            std::vector<llama_token> out(k_prompt.size() + 8);
            size_t n_out = 0;
            ok = llama_state_load_file(fresh, sess.c_str(), out.data(), out.size(), &n_out) &&
                n_out == k_prompt.size();
            llama_free(fresh);
        }
    }

    (void) std::remove(sess.c_str());
    return ok;
}

// --soak mode: loop turbo3 and turbo4 K/V save and load; print VmRSS every 30 s
static int run_soak(llama_model * model, long seconds) {
    static const ggml_type kv_types[2][2] = {
        { GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0 },
        { GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0 },
    };

    const auto t0 = std::chrono::steady_clock::now();
    long elapsed = 0;
    long next_sample = 0;
    long rounds = 0;
    int rc = 0;
    while (elapsed < seconds) {
        if (elapsed >= next_sample) {
            (void) printf("rss t=%ld kb=%ld\n", next_sample, read_vmrss_kb());
            (void) fflush(stdout);
            next_sample += 30;
        }
        const ggml_type * kv = kv_types[rounds % 2];
        if (!soak_round(model, kv[0], kv[1])) {
            (void) fprintf(stderr, "FAILED: soak round %ld\n", rounds);
            rc = 1;
            break;
        }
        rounds++;
        elapsed = (long) std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - t0).count();
    }

    (void) printf("rss_end t=%ld kb=%ld\n", elapsed, read_vmrss_kb());
    (void) printf("soak rounds=%ld rc=%d\n", rounds, rc);
    return rc;
}

int main(int argc, char ** argv) {
    const char * mode = nullptr;
    const char * sess = nullptr;
    int arg = 1;
    if (argc >= 3 && (std::strcmp(argv[1], "--save") == 0 || std::strcmp(argv[1], "--load") == 0)) {
        mode = argv[1];
        sess = argv[2];
        arg = 3;
    }

    const bool soak = argc == 4 && std::strcmp(argv[1], "--soak") == 0;
    if (soak) {
        arg = 2;
    }

    if (argc <= arg) {
        (void) fprintf(stderr, "usage: %s [--save <session> | --load <session>] <model.gguf> | --soak <model.gguf> <seconds>\n", argv[0]);
        return 77;
    }

    llama_backend_init();

    llama_model * model = load_model(argv[arg]);
    if (!model) {
        (void) fprintf(stderr, "failed to load model: %s\n", argv[arg]);
        llama_backend_free();
        return 1;
    }

    if (soak) {
        const int rc = run_soak(model, std::atol(argv[3]));
        llama_model_free(model);
        llama_backend_free();
        return rc;
    }

    if (mode) {
        const int rc = std::strcmp(mode, "--save") == 0 ? run_save(model, sess) : run_load(model, sess);
        llama_model_free(model);
        llama_backend_free();
        return rc;
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
