// TurboQuant codec check. The codec stores vectors in the WHT-rotated domain,
// so the dequantized values are compared against W*x, where W is the turbo WHT.
#include "ggml.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int test_codec(ggml_type type, double max_nmse) {
    const int64_t n = 128 * 256;

    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x(n), wx(n), y(n);
    for (auto & v : x) {
        v = nd(rng);
    }

    ggml_init_params params = { 64*1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * tx = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, n/128);
    memcpy(tx->data, x.data(), n*sizeof(float));
    ggml_tensor * w = ggml_turbo_wht(ctx, tx, 0, 0, nullptr);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, w);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    memcpy(wx.data(), w->data, n*sizeof(float));
    ggml_free(ctx);

    std::vector<char> buf(ggml_row_size(type, n));
    ggml_get_type_traits_cpu(type)->from_float(x.data(), buf.data(), n);
    ggml_get_type_traits(type)->to_float(buf.data(), y.data(), n);

    double err = 0, ref = 0;
    for (int64_t i = 0; i < n; i++) {
        err += (wx[i] - y[i]) * (wx[i] - y[i]);
        ref += wx[i] * wx[i];
    }
    const double nmse = err / ref;
    const bool ok = nmse <= max_nmse;

    printf("%-10s NMSE vs W*x = %.5f (max %.5f) %s\n", ggml_type_name(type), nmse, max_nmse, ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}

int main() {
    int failed = 0;
    // measured 0.1185, 0.0335, 0.0091 on Gaussian data; limits leave ~10% margin
    failed += test_codec(GGML_TYPE_TURBO2_0, 0.13);
    failed += test_codec(GGML_TYPE_TURBO3_0, 0.037);
    failed += test_codec(GGML_TYPE_TURBO4_0, 0.0105);
    printf("%d tests failed\n", failed);
    return failed != 0;
}
