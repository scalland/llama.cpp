// Header bounds of the turbo session pre-pass (D-T5, steps 1 to 3 of D12).
//
// Calls llama_session_prepass_plain_kv directly on an in-memory full-session
// file. The file is built to the layout that llama_state_save_file writes
// (src/llama-context.cpp, src/llama-kv-cache.cpp), so no model is needed.
//
// Cases:
//   positive: a file written in the save layout passes the header checks
//   c: token count above the caller capacity
//   d: cell count above the context maximum
//   n: architecture string, n_layer, n_rot_k and n_rot_v mismatch
//
// Each malformed case must return false AND give a reason in error that is not
// the placeholder text of the unimplemented stub. The stub rejects everything,
// so without that second check the malformed cases would pass for the wrong reason.
//
// Per-layer row size (e) is covered by the bounds test for that case; the file
// size cap (o) is covered elsewhere.

#include "llama.h"
#include "llama-session-prepass.h"
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr uint32_t k_n_cells_max = 8;
constexpr uint32_t k_n_layer     = 1;
constexpr int32_t  k_n_rot       = 64;
constexpr uint64_t k_row_size    = 16;
constexpr int32_t  k_n_vocab     = 32;
const char *       k_arch        = "llama";
const char *       k_stub_error  = "session pre-pass is not implemented";

// Byte offsets of the header fields the malformed cases patch in place.
struct file_offsets {
    size_t arch_bytes = 0;
    size_t n_rot_k    = 0;
    size_t n_rot_v    = 0;
};

// Parameters of the file that the builder writes.
struct file_config {
    uint32_t    n_token = 2;     // tokens written after the count
    uint32_t    n_cells = 2;     // cells written in the stream
    uint32_t    n_layer = k_n_layer;
    std::string arch    = k_arch;
    int32_t     n_rot_k = k_n_rot;
    int32_t     n_rot_v = k_n_rot;
};

void append_bytes(std::vector<uint8_t> & out, const void * src, size_t n) {
    const size_t at = out.size();
    out.resize(at + n);
    if (n > 0) {
        std::memcpy(out.data() + at, src, n);
    }
}

template <typename T>
void append(std::vector<uint8_t> & out, T v) {
    append_bytes(out, &v, sizeof(T));
}

// Overwrite sizeof(T) bytes at off with v. The caller guarantees off is in range.
template <typename T>
void patch(std::vector<uint8_t> & buf, size_t off, T v) {
    std::memcpy(buf.data() + off, &v, sizeof(T));
}

// Build a full-session file in the save layout for one stream.
std::vector<uint8_t> build_file(const file_config & cfg, file_offsets & off) {
    std::vector<uint8_t> out;

    append<uint32_t>(out, LLAMA_SESSION_MAGIC);
    append<uint32_t>(out, LLAMA_SESSION_VERSION);

    // token count and tokens (llama_token is int32)
    append<uint32_t>(out, cfg.n_token);
    for (uint32_t i = 0; i < cfg.n_token; ++i) {
        append<int32_t>(out, i % k_n_vocab);
    }

    // architecture string: u32 length, then bytes
    append<uint32_t>(out, static_cast<uint32_t>(cfg.arch.size()));
    off.arch_bytes = out.size();
    append_bytes(out, cfg.arch.data(), cfg.arch.size());

    // one stream
    append<uint32_t>(out, 1);

    // cell count, then per-cell meta: pos i32, n_seq_id u32, seq_id i32 x n_seq_id
    append<uint32_t>(out, cfg.n_cells);
    for (uint32_t i = 0; i < cfg.n_cells; ++i) {
        append<int32_t>(out, static_cast<int32_t>(i));
        append<uint32_t>(out, 1);
        append<int32_t>(out, 0);
    }

    // KV data header: v_trans, n_layer, n_rot_k, n_rot_v
    append<uint32_t>(out, 0);
    append<uint32_t>(out, cfg.n_layer);
    off.n_rot_k = out.size();
    append<int32_t>(out, cfg.n_rot_k);
    off.n_rot_v = out.size();
    append<int32_t>(out, cfg.n_rot_v);

    const uint64_t payload = k_row_size * cfg.n_cells;
    for (uint32_t il = 0; il < cfg.n_layer; ++il) {
        // K: type i32, row size u64, payload
        append<int32_t>(out, static_cast<int32_t>(GGML_TYPE_F16));
        append<uint64_t>(out, k_row_size);
        out.resize(out.size() + payload, 0);

        // V (v_trans false): type i32, row size u64, payload
        append<int32_t>(out, static_cast<int32_t>(GGML_TYPE_F16));
        append<uint64_t>(out, k_row_size);
        out.resize(out.size() + payload, 0);
    }

    return out;
}

// Layout of the live context that every case is checked against.
llama_session_prepass_layout make_layout() {
    llama_session_prepass_layout layout;
    layout.n_cells_max = k_n_cells_max;
    layout.n_vocab     = k_n_vocab;
    layout.n_seq_max   = 1;
    layout.n_layer     = k_n_layer;
    layout.n_rot_k     = k_n_rot;
    layout.n_rot_v     = k_n_rot;
    layout.v_trans     = false;
    layout.arch        = k_arch;

    llama_session_prepass_layer ly;
    ly.k_type     = GGML_TYPE_F16;
    ly.k_row_size = static_cast<size_t>(k_row_size);
    ly.v_type     = GGML_TYPE_F16;
    ly.v_row_size = static_cast<size_t>(k_row_size);
    layout.layers.push_back(ly);

    return layout;
}

// Run the pre-pass on the full-session path. Returns the error text in error.
bool run_prepass(const std::vector<uint8_t> & buf, std::string & error) {
    return llama_session_prepass_plain_kv(buf.data(), buf.size(), make_layout(), false, error);
}

int report(const char * name, bool pass, const std::string & detail) {
    printf("%-40s %s%s%s\n", name, pass ? "OK" : "FAILED", detail.empty() ? "" : " ", detail.c_str());
    return pass ? 0 : 1;
}

// A file that must pass the header checks.
int test_valid_passes() {
    file_config cfg;
    file_offsets off;
    const std::vector<uint8_t> buf = build_file(cfg, off);

    std::string error;
    const bool ok = run_prepass(buf, error);
    return report("save-layout file passes", ok, ok ? "" : "error: " + error);
}

// A malformed file must be rejected with a reason of its own.
int expect_rejected(const char * name, const std::vector<uint8_t> & buf) {
    std::string error;
    const bool ok = run_prepass(buf, error);
    const bool pass = !ok && !error.empty() && error != k_stub_error;
    return report(name, pass, pass ? "" : "returned " + std::string(ok ? "true" : "false") + ", error: '" + error + "'");
}

int test_token_count_above_capacity() {
    file_config cfg;
    cfg.n_token = 4096; // far above any caller capacity used by the loader
    file_offsets off;
    return expect_rejected("c: token count above capacity", build_file(cfg, off));
}

int test_cell_count_above_max() {
    file_config cfg;
    cfg.n_cells = k_n_cells_max + 1; // one cell more than the context maximum
    file_offsets off;
    return expect_rejected("d: cell count above context maximum", build_file(cfg, off));
}

int test_arch_mismatch() {
    file_config cfg;
    file_offsets off;
    std::vector<uint8_t> buf = build_file(cfg, off);
    // same length as "llama", so no other field moves
    std::memcpy(buf.data() + off.arch_bytes, "gemma", 5);
    return expect_rejected("n: architecture string mismatch", buf);
}

int test_n_layer_mismatch() {
    file_config cfg;
    cfg.n_layer = k_n_layer + 1; // the file carries two layers; the context has one
    file_offsets off;
    return expect_rejected("n: n_layer mismatch", build_file(cfg, off));
}

int test_n_rot_k_mismatch() {
    file_config cfg;
    file_offsets off;
    std::vector<uint8_t> buf = build_file(cfg, off);
    patch<int32_t>(buf, off.n_rot_k, k_n_rot / 2);
    return expect_rejected("n: n_rot_k mismatch", buf);
}

int test_n_rot_v_mismatch() {
    file_config cfg;
    file_offsets off;
    std::vector<uint8_t> buf = build_file(cfg, off);
    patch<int32_t>(buf, off.n_rot_v, k_n_rot / 2);
    return expect_rejected("n: n_rot_v mismatch", buf);
}

} // namespace

int main() {
    int failed = 0;
    failed += test_valid_passes();
    failed += test_token_count_above_capacity();
    failed += test_cell_count_above_max();
    failed += test_arch_mismatch();
    failed += test_n_layer_mismatch();
    failed += test_n_rot_k_mismatch();
    failed += test_n_rot_v_mismatch();
    printf("%d tests failed\n", failed);
    return failed != 0;
}
