// Deterministic session mutation test for turbo builds.
//
// Mutated session files (byte changes and truncation) must not crash the loader.
// Every mutation the loader rejects must leave the context unchanged.
//
// usage: test-turbo-session-mutate <model.gguf>
// Exits with 77 (ctest SKIP) when no model path is given.
#include "common.h"
#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <random>
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

static const unsigned n_mutations = 2000;
static const unsigned seed        = 1234;

static bool read_bytes(const std::string & path, std::vector<char> & data) {
    std::ifstream f(path, std::ios::in | std::ios::binary);
    if (!f) {
        return false;
    }
    data.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    return true;
}

static bool write_bytes(const std::string & path, const std::vector<char> & data) {
    std::ofstream f(path, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!f) {
        return false;
    }
    f.write(data.data(), (std::streamsize) data.size());
    return (bool) f;
}

// the full state of the context, used to prove that a rejected load changed nothing
static std::vector<uint8_t> snapshot(llama_context * ctx) {
    std::vector<uint8_t> buf(llama_state_get_size(ctx));
    const size_t n = llama_state_get_data(ctx, buf.data(), buf.size());
    buf.resize(n);
    return buf;
}

static bool decode_prompt(llama_context * ctx, const std::vector<llama_token> & tokens) {
    common_batch batch(ctx);
    for (size_t i = 0; i < tokens.size(); i++) {
        batch.add(tokens[i], (llama_pos) i, 0, i + 1 == tokens.size());
    }
    return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
}

static bool load(llama_context * ctx, const std::string & path, size_t n_tokens) {
    std::vector<llama_token> out(n_tokens + 8);
    size_t n_out = 0;
    return llama_state_load_file(ctx, path.c_str(), out.data(), out.size(), &n_out);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        (void) fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 77;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (!model) {
        (void) fprintf(stderr, "failed to load model: %s\n", argv[1]);
        llama_backend_free();
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 64;
    cparams.n_batch = 64;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        (void) fprintf(stderr, "failed to create context\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    const std::vector<llama_token> prompt = { 1, 1, 1, 1 };
    CHECK(decode_prompt(ctx, prompt), "live state: prompt decoded into the context");

    const std::string good = "test-turbo-session-mutate-good.bin";
    const std::string mut  = "test-turbo-session-mutate-mut.bin";

    CHECK(llama_state_save_file(ctx, good.c_str(), prompt.data(), prompt.size()), "good session saved");

    std::vector<char> good_bytes;
    CHECK(read_bytes(good, good_bytes) && !good_bytes.empty(), "good session read back");

    CHECK(load(ctx, good, prompt.size()), "good session loads");
    const std::vector<uint8_t> base = snapshot(ctx);

    std::mt19937 rng(seed);
    unsigned n_rejected = 0;
    unsigned n_accepted = 0;

    for (unsigned i = 0; i < n_mutations; i++) {
        std::vector<char> data = good_bytes;

        if (rng() % 4 == 0) {
            data.resize(rng() % good_bytes.size());
        } else {
            const unsigned n_changes = 1 + rng() % 4;
            for (unsigned k = 0; k < n_changes && !data.empty(); k++) {
                const size_t pos = rng() % data.size();
                data[pos] = (char) (data[pos] ^ (char) (1 + rng() % 255));
            }
        }

        if (!write_bytes(mut, data)) {
            failed++;
            (void) fprintf(stderr, "FAILED: mutation %u could not be written\n", i);
            continue;
        }

        if (!load(ctx, mut, prompt.size())) {
            n_rejected++;
            if (snapshot(ctx) != base) {
                failed++;
                (void) fprintf(stderr, "FAILED: mutation %u rejected but changed the context\n", i);
            }
        } else {
            n_accepted++;
            // restore the good state so the next mutation starts from the same context
            if (!load(ctx, good, prompt.size()) || snapshot(ctx) != base) {
                failed++;
                (void) fprintf(stderr, "FAILED: could not restore the good state after mutation %u\n", i);
            }
        }
    }

    (void) printf("mutations: %u, rejected: %u, accepted: %u\n", n_mutations, n_rejected, n_accepted);
    CHECK(n_rejected > 0, "some mutations are rejected by the loader");

    (void) std::remove(good.c_str());
    (void) std::remove(mut.c_str());

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    (void) printf("%d checks failed\n", failed);
    return failed != 0;
}
