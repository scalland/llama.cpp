// Session version and magic rejection checks for turbo builds.
//
// A session file or sequence state with a stale version or a wrong magic must be rejected
// by the loader, and a rejected load must leave a context that holds live state unchanged.
//
// usage: test-turbo-session-version <model.gguf>
// Exits with 77 (ctest SKIP) when no model path is given.
#include "common.h"
#include "llama.h"

#include <cstdint>
#include <cstdio>
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

// byte offset into a file, wrapped so it cannot be mixed up with a value
struct file_offset {
    long pos;
};

// overwrite a 32-bit field of a file in place (host byte order, same as the writer)
static bool patch_u32(const std::string & path, file_offset offset, uint32_t value) {
    std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
    if (!f) {
        return false;
    }
    f.seekp(offset.pos);
    f.write(reinterpret_cast<const char *>(&value), sizeof(value));
    return (bool) f;
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

// put some tokens into the context so that it holds live state
static bool decode_prompt(llama_context * ctx, const std::vector<llama_token> & tokens) {
    common_batch batch(ctx);
    for (size_t i = 0; i < tokens.size(); i++) {
        batch.add(tokens[i], (llama_pos) i, 0, i + 1 == tokens.size());
    }
    return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
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
    cparams.n_ctx   = 256;
    cparams.n_batch = 256;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        (void) fprintf(stderr, "failed to create context\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // token 1 is valid for every vocabulary the test model can have
    const std::vector<llama_token> prompt = { 1, 1, 1, 1 };
    CHECK(decode_prompt(ctx, prompt), "live state: prompt decoded into the context");

    const char * sess  = "test-turbo-session-version-session.bin";
    const char * empty = "test-turbo-session-version-empty.bin";

    // (a) version mismatch: a real session saved with the current writer, version patched to 10
    CHECK(llama_state_save_file(ctx, sess, prompt.data(), prompt.size()), "session saved with the current version");
    CHECK(patch_u32(sess, file_offset{ 4 }, 10), "session version patched to 10");

    {
        const auto before = snapshot(ctx);
        std::vector<llama_token> out(prompt.size() + 8);
        size_t n_out = 0;
        const bool ok = llama_state_load_file(ctx, sess, out.data(), out.size(), &n_out);
        CHECK(!ok, "session version 10 is rejected");
        CHECK(snapshot(ctx) == before, "context unchanged after session version 10 is rejected");
    }

    // (b) wrong magic: a real session with the magic word zeroed
    CHECK(llama_state_save_file(ctx, sess, prompt.data(), prompt.size()), "session re-saved for the magic case");
    CHECK(patch_u32(sess, file_offset{ 0 }, 0), "session magic zeroed");
    {
        const auto before = snapshot(ctx);
        std::vector<llama_token> out(prompt.size() + 8);
        size_t n_out = 0;
        const bool ok = llama_state_load_file(ctx, sess, out.data(), out.size(), &n_out);
        CHECK(!ok, "session with wrong magic is rejected");
        CHECK(snapshot(ctx) == before, "context unchanged after wrong magic is rejected");
    }

    // (b) zero-length file
    CHECK(write_bytes(empty, {}), "zero-length file written");
    {
        const auto before = snapshot(ctx);
        std::vector<llama_token> out(prompt.size() + 8);
        size_t n_out = 0;
        const bool ok = llama_state_load_file(ctx, empty, out.data(), out.size(), &n_out);
        CHECK(!ok, "zero-length session file is rejected");
        CHECK(snapshot(ctx) == before, "context unchanged after zero-length file is rejected");
    }

    (void) std::remove(sess);
    (void) std::remove(empty);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    (void) printf("%d checks failed\n", failed);
    return failed != 0;
}
