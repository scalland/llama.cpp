#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Read-only pre-pass over a plain KV cache session file (D-T5).
//
// The pre-pass validates the whole session file in one buffer before any
// mutation of the context. It must not write to the context or to the buffer.

// Per-layer live layout of the context that the file is checked against.
struct llama_session_prepass_layer {
    ggml_type k_type;      // live K cache type of the layer
    size_t    k_row_size;  // ggml_row_size(k_type, n_embd_k_gqa(il))
    ggml_type v_type;      // live V cache type of the layer
    size_t    v_row_size;  // ggml_row_size(v_type, n_embd_v_gqa(il)); element size when v_trans
};

// Live layout of the context, the reference for every check in the pre-pass.
struct llama_session_prepass_layout {
    uint32_t n_cells_max;     // maximum cell count of the context (cells.size())
    int32_t  n_vocab;         // vocabulary size, bound for ext.tok
    uint32_t n_seq_max;       // bound for seq_id
    uint32_t n_layer;         // layers.size()
    int32_t  n_rot_k;         // live n_rot_k
    int32_t  n_rot_v;         // live n_rot_v
    bool     v_trans;         // live v_trans flag
    std::string arch;         // live architecture string
    std::vector<llama_session_prepass_layer> layers;
};

// Checks the session file held in buffer [data, data + size).
//
// seq_path: true for the sequence restore path (llama_state_seq_load_file),
// false for the full-session path (llama_state_load_file).
//
// Returns true when every check passes. On failure returns false and sets
// error to a short reason; the caller must leave the context unchanged.
bool llama_session_prepass_plain_kv(
        const uint8_t * data,
        size_t size,
        const llama_session_prepass_layout & layout,
        bool seq_path,
        std::string & error);
