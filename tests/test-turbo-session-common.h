// Shared helpers for the turbo session tests (phase 1 and later).
//
// Holds no test cases. Provides:
//   - turbo_session_setup:   create a model and a context (as in tests/test-llama-context.cpp)
//   - turbo_session_save:    write a session file through llama_state_save_file (the save path)
//   - turbo_session_read_file / turbo_session_write_file: move a file to and from one buffer
//   - turbo_session_walk:    locate the header, per-cell and per-layer fields of a full-session file
//   - turbo_session_put_*, turbo_session_truncate, turbo_session_append: byte patching helpers
//
// Layout of a full-session file, as written by the save path:
//   magic u32, version u32                              src/llama-context.cpp:3974-3975
//   n_token_count u32, tokens (4 bytes each)            src/llama-context.cpp:3978-3979
//   arch string: u32 length, then bytes                 src/llama-context.cpp:4070, src/llama-io.cpp:5-10
//   n_stream u32                                        src/llama-kv-cache.cpp:2245
//   per stream: cell_count u32                          src/llama-kv-cache.cpp:2295
//     if cell_count > 0, per-cell KV meta:              src/llama-kv-cache.cpp:2385-2416
//       pos i32, n_seq_id u32, ext (only when has_cell_ext), seq_id i32 x n_seq_id
//     KV data header: v_trans u32, n_layer u32,
//       n_rot_k u32, n_rot_v u32                        src/llama-kv-cache.cpp:2424-2427
//     per layer K: type i32, row size u64, payload      src/llama-kv-cache.cpp:2440-2448
//       payload = row size * cell_count
//     per layer V, v_trans false: type i32, row u64,
//       payload = row size * cell_count                 src/llama-kv-cache.cpp:2467-2475
//     per layer V, v_trans true: type i32, element size u32,
//       GQA embedding size u32, payload = n_embd_v_gqa * cell_count * element size
//                                                        src/llama-kv-cache.cpp:2496-2510
//
// The seq session file (llama_state_seq_save_file) has a different header and is not walked here.

#pragma once

#include "llama.h"
#include "llama-cpp.h"
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// Field offsets of one full-session file. Filled by turbo_session_walk.
struct turbo_session_cell {
    size_t     pos_off      = 0; // src/llama-kv-cache.cpp:2403 (pos, i32)
    llama_pos  pos          = 0;
    size_t     n_seq_id_off = 0; // src/llama-kv-cache.cpp:2404 (u32)
    size_t   ext_off       = 0; // src/llama-kv-cache.cpp:2407-2408 (only when has_ext)
    size_t   seq_ids_off   = 0; // src/llama-kv-cache.cpp:2411-2412 (i32 each)
    uint32_t n_seq_id      = 0;
};

struct turbo_session_layer {
    size_t k_type_off   = 0; // src/llama-kv-cache.cpp:2440 (i32)
    size_t k_row_off    = 0; // src/llama-kv-cache.cpp:2444 (u64)
    size_t k_data_off   = 0;
    size_t k_data_len   = 0;
    size_t v_type_off   = 0; // src/llama-kv-cache.cpp:2467 or :2496 (i32)
    size_t v_row_off    = 0; // non-trans: u64 row size, src/llama-kv-cache.cpp:2471
    size_t v_el_off     = 0; // trans only: u32 element size, src/llama-kv-cache.cpp:2500
    size_t v_gqa_off    = 0; // trans only: u32 GQA embedding size, src/llama-kv-cache.cpp:2503
    size_t v_data_off   = 0;
    size_t v_data_len   = 0;
};

struct turbo_session_stream {
    size_t   cell_count_off = 0; // src/llama-kv-cache.cpp:2295 (u32)
    uint32_t n_cells        = 0;
    std::vector<turbo_session_cell> cells;
    size_t   v_trans_off    = 0; // src/llama-kv-cache.cpp:2424 (u32)
    size_t   n_layer_off    = 0; // src/llama-kv-cache.cpp:2425 (u32)
    size_t   n_rot_k_off    = 0; // src/llama-kv-cache.cpp:2426 (u32)
    size_t   n_rot_v_off    = 0; // src/llama-kv-cache.cpp:2427 (u32)
    bool     v_trans        = false;
    std::vector<turbo_session_layer> layers;
};

struct turbo_session_layout {
    size_t   n_token_off    = 0; // src/llama-context.cpp:3978 (u32)
    uint32_t n_token        = 0;
    size_t   tokens_off     = 0; // src/llama-context.cpp:3979
    size_t   arch_len_off   = 0; // src/llama-context.cpp:4070 (u32 string length)
    uint32_t arch_len       = 0;
    size_t   n_stream_off   = 0; // src/llama-kv-cache.cpp:2245 (u32)
    uint32_t n_stream       = 0;
    std::vector<turbo_session_stream> streams;
    size_t   end            = 0; // offset one past the last byte the layout covers
};

// K and V cache types. A struct keeps the two same-typed values from being swapped at call sites.
struct turbo_session_kv_types {
    ggml_type type_k;
    ggml_type type_v;
};

// Create a model and a context. Returns false when the model cannot be loaded.
// Callers pass cache types and flash attention explicitly, as the turbo tests need.
static inline bool turbo_session_setup(
        llama_model_ptr & model, // NOLINT(bugprone-easily-swappable-parameters): distinct smart-pointer types (llama_model_ptr, llama_context_ptr), no implicit conversion between them
        llama_context_ptr & ctx,
        const std::string & model_path,
        uint32_t n_ctx,
        turbo_session_kv_types kv,
        llama_flash_attn_type flash_attn) {
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    model.reset(llama_model_load_from_file(model_path.c_str(), mparams));
    if (!model) {
        return false;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = n_ctx;
    cparams.n_batch         = n_ctx;
    cparams.type_k          = kv.type_k;
    cparams.type_v          = kv.type_v;
    cparams.flash_attn_type = flash_attn;

    ctx.reset(llama_init_from_model(model.get(), cparams));
    return ctx != nullptr;
}

// Write a session file through the save path. Returns false on failure.
static inline bool turbo_session_save(
        llama_context * ctx,
        const std::string & path, // NOLINT(bugprone-easily-swappable-parameters): path and token list are different types
        const std::vector<llama_token> & tokens) {
    return llama_state_save_file(ctx, path.c_str(), tokens.data(), tokens.size());
}

// Read a whole file into one buffer. Returns false when the file cannot be opened or read,
// or when it is larger than max_bytes. The size is checked before the buffer is allocated.
static inline bool turbo_session_read_file(const std::string & path, std::vector<uint8_t> & buf, size_t max_bytes) {
    buf.clear();

    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }

    bool ok = std::fseek(f, 0, SEEK_END) == 0;
    const long end = ok ? std::ftell(f) : -1;
    ok = ok && end >= 0 && std::fseek(f, 0, SEEK_SET) == 0;
    if (!ok || static_cast<unsigned long long>(end) > max_bytes) {
        // Read-only handle and the result is already false: the close status does not change the outcome.
        (void) std::fclose(f);
        return false;
    }

    buf.resize(static_cast<size_t>(end));
    const size_t n = buf.empty() ? 0 : std::fread(buf.data(), 1, buf.size(), f);
    ok = n == buf.size() && std::ferror(f) == 0;
    const bool closed = std::fclose(f) == 0;
    return ok && closed;
}

// Write a buffer to a file, replacing it. Returns false on failure.
static inline bool turbo_session_write_file(const std::string & path, const std::vector<uint8_t> & buf) { // NOLINT(bugprone-easily-swappable-parameters): path and buffer are different types
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) {
        return false;
    }

    const size_t n = buf.empty() ? 0 : std::fwrite(buf.data(), 1, buf.size(), f);
    const bool ok = n == buf.size();
    return std::fclose(f) == 0 && ok;
}

// Read a little-endian-host value at an offset. Returns false when it does not fit in the buffer.
template <typename T>
static inline bool turbo_session_get(const std::vector<uint8_t> & buf, size_t off, T & out) { // NOLINT(bugprone-easily-swappable-parameters): buffer, offset and output differ in type
    if (off > buf.size() || sizeof(T) > buf.size() - off) {
        return false;
    }
    std::memcpy(&out, buf.data() + off, sizeof(T));
    return true;
}

// Overwrite bytes at an offset. Returns false when the write would pass the end of the buffer.
static inline bool turbo_session_put_bytes(std::vector<uint8_t> & buf, size_t off, const void * src, size_t n) {
    if (off > buf.size() || n > buf.size() - off) {
        return false;
    }
    std::memcpy(buf.data() + off, src, n);
    return true;
}

template <typename T>
static inline bool turbo_session_put(std::vector<uint8_t> & buf, size_t off, const T & value) {
    return turbo_session_put_bytes(buf, off, &value, sizeof(T));
}

// Drop bytes from the end of the buffer. Returns false when new_size is larger than the buffer.
static inline bool turbo_session_truncate(std::vector<uint8_t> & buf, size_t new_size) {
    if (new_size > buf.size()) {
        return false;
    }
    buf.resize(new_size);
    return true;
}

// Add bytes at the end of the buffer.
static inline void turbo_session_append(std::vector<uint8_t> & buf, const void * src, size_t n) {
    const auto * p = static_cast<const uint8_t *>(src);
    buf.insert(buf.end(), p, p + n);
}

// Walk a full-session file written by the save path and record the offset of every
// header, per-cell and per-layer field. has_ext: the model has cell ext records
// (llama_kv_cache::has_cell_ext); ext_size: sizeof(llama_kv_cell_ext).
// Returns false when the file is too short for the layout it describes.
// Reads only; the buffer is not changed.
static inline bool turbo_session_walk(
        const std::vector<uint8_t> & buf,
        bool has_ext,
        size_t ext_size,
        turbo_session_layout & out) {
    out = turbo_session_layout{};

    size_t off = 0;

    // magic and version: checked by the loader, not walked
    off += 8;

    out.n_token_off = off;
    if (!turbo_session_get(buf, off, out.n_token)) {
        return false;
    }
    off += 4;

    out.tokens_off = off;
    const size_t tokens_len = size_t(out.n_token) * sizeof(llama_token);
    if (tokens_len > buf.size() - off) {
        return false;
    }
    off += tokens_len;

    out.arch_len_off = off;
    if (!turbo_session_get(buf, off, out.arch_len)) {
        return false;
    }
    off += 4;
    if (out.arch_len > buf.size() - off) {
        return false;
    }
    off += out.arch_len;

    out.n_stream_off = off;
    if (!turbo_session_get(buf, off, out.n_stream)) {
        return false;
    }
    off += 4;

    for (uint32_t s = 0; s < out.n_stream; ++s) {
        turbo_session_stream st;

        st.cell_count_off = off;
        if (!turbo_session_get(buf, off, st.n_cells)) {
            return false;
        }
        off += 4;

        if (st.n_cells == 0) {
            out.streams.push_back(st);
            continue;
        }

        // per-cell KV meta: every cell needs at least pos, n_seq_id and the ext record when present
        const size_t min_cell_bytes = sizeof(llama_pos) + 4 + (has_ext ? ext_size : 0);
        if (size_t(st.n_cells) > (buf.size() - off) / min_cell_bytes) {
            return false;
        }
        st.cells.resize(st.n_cells);
        for (uint32_t c = 0; c < st.n_cells; ++c) {
            turbo_session_cell & cell = st.cells[c];

            cell.pos_off = off;
            if (!turbo_session_get(buf, off, cell.pos)) {
                return false;
            }
            off += sizeof(llama_pos);

            cell.n_seq_id_off = off;
            if (!turbo_session_get(buf, off, cell.n_seq_id)) {
                return false;
            }
            off += 4;

            if (has_ext) {
                cell.ext_off = off;
                if (ext_size > buf.size() - off) {
                    return false;
                }
                off += ext_size;
            }

            cell.seq_ids_off = off;
            const size_t ids_len = size_t(cell.n_seq_id) * sizeof(llama_seq_id);
            if (ids_len > buf.size() - off) {
                return false;
            }
            off += ids_len;
        }

        // KV data header
        st.v_trans_off = off;
        uint32_t v_trans = 0;
        uint32_t n_layer = 0;
        if (!turbo_session_get(buf, off, v_trans)) {
            return false;
        }
        off += 4;
        st.v_trans = v_trans != 0;

        st.n_layer_off = off;
        if (!turbo_session_get(buf, off, n_layer)) {
            return false;
        }
        off += 4;

        st.n_rot_k_off = off;
        off += 4;
        st.n_rot_v_off = off;
        off += 4;
        if (off > buf.size()) {
            return false;
        }

        // K data for every layer
        for (uint32_t il = 0; il < n_layer; ++il) {
            turbo_session_layer ly;

            ly.k_type_off = off;
            off += 4;

            uint64_t k_row = 0;
            ly.k_row_off = off;
            if (!turbo_session_get(buf, off, k_row)) {
                return false;
            }
            off += 8;

            ly.k_data_off = off;
            if (k_row > (buf.size() - off) / st.n_cells) {
                return false;
            }
            ly.k_data_len = size_t(k_row) * st.n_cells;
            off += ly.k_data_len;

            st.layers.push_back(ly);
        }

        // V data for every layer
        for (uint32_t il = 0; il < n_layer; ++il) {
            turbo_session_layer & ly = st.layers[il];

            ly.v_type_off = off;
            off += 4;

            if (!st.v_trans) {
                uint64_t v_row = 0;
                ly.v_row_off = off;
                if (!turbo_session_get(buf, off, v_row)) {
                    return false;
                }
                off += 8;

                ly.v_data_off = off;
                if (v_row > (buf.size() - off) / st.n_cells) {
                    return false;
                }
                ly.v_data_len = size_t(v_row) * st.n_cells;
            } else {
                uint32_t v_el = 0;
                uint32_t v_gqa = 0;
                ly.v_el_off = off;
                if (!turbo_session_get(buf, off, v_el)) {
                    return false;
                }
                off += 4;

                ly.v_gqa_off = off;
                if (!turbo_session_get(buf, off, v_gqa)) {
                    return false;
                }
                off += 4;

                ly.v_data_off = off;
                const size_t v_cells = size_t(v_gqa) * st.n_cells;
                if (v_el != 0 && v_cells > (buf.size() - off) / v_el) {
                    return false;
                }
                ly.v_data_len = v_cells * v_el;
            }

            if (ly.v_data_len > buf.size() - off) {
                return false;
            }
            off += ly.v_data_len;
        }

        out.streams.push_back(st);
    }

    out.end = off;
    return true;
}
