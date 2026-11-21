#pragma once

// Read stub for the turbo session load tests.
//
// Every read returns 0 bytes and is counted, so a load that reads from this
// stream fails and the test can read back how many reads were attempted.
// Test files that include this header need ${PROJECT_SOURCE_DIR}/src on their
// include path (llama-io.h lives there).

#include "llama-io.h"

#include <cstddef>

class llama_io_read_stub : public llama_io_read_i {
public:
    llama_io_read_stub() = default;

    void read(void * /*dst*/, size_t /*size*/) override {
        n_read_calls++;
    }

    void read_tensor(ggml_tensor * /*tensor*/, size_t /*offset*/, size_t /*size*/) override {
        n_read_tensor_calls++;
    }

    size_t n_bytes() override {
        return 0;
    }

    // number of read() calls seen so far
    size_t get_read_calls() const {
        return n_read_calls;
    }

    // number of read_tensor() calls seen so far
    size_t get_read_tensor_calls() const {
        return n_read_tensor_calls;
    }

private:
    size_t n_read_calls        = 0;
    size_t n_read_tensor_calls = 0;
};
