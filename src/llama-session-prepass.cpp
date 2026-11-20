#include "llama-session-prepass.h"

// Stub: the checks of D-T5 are added in later cards. No caller is wired yet.
bool llama_session_prepass_plain_kv(
        const uint8_t * data,
        size_t size,
        const llama_session_prepass_layout & layout,
        bool seq_path,
        std::string & error) {
    (void) data;
    (void) size;
    (void) layout;
    (void) seq_path;

    error = "session pre-pass is not implemented";
    return false;
}
