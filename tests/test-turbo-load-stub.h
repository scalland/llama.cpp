#pragma once

struct llama_context;

// Test stub for the full-session load path. It always returns false and does not
// touch the context. Phase 5 (D17) replaces each stub call site with the real load.
bool turbo_stub_full_session_load(struct llama_context * ctx, const char * path_session);
