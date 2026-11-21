#include "test-turbo-load-stub.h"

bool turbo_stub_full_session_load(struct llama_context * ctx, const char * path_session) {
    (void) ctx;
    (void) path_session;

    return false;
}
