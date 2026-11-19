# Phase 1 results: turbo2, turbo3, turbo4 on the TheTom base

## Build
- Base: TheTom llama-cpp-turboquant, commit 4185b90a0.
- Tested commit: 1f962792b (branch tt/T1), which drops the workbox[hash].js requirement in tools/ui/embed.cpp so the UI embeds from the b12498 bundle.
- Build command (GB10 node 192.168.1.112, CUDA arch 121):
  `cmake -S . -B build-ui -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_UI=ON -DLLAMA_USE_PREBUILT_UI=ON`
  with `HF_UI_VERSION=b12498`, then `cmake --build build-ui -j 20 --target llama-server llama-cli test-backend-ops test-turbo-quant test-quantize-fns`
- Result: exit 0, 193 s.

## Test commands and pass counts
| Command | Result | Log |
|---|---|---|
| `bash tests/kv-dtypes/test_ctk_flags.sh build-ui/bin/llama-cli` | 8/8 PASS, exit 0 | t1-final-flags.log |
| `./build-ui/bin/test-turbo-quant` | exit 0; 35 "ok" lines in the chunked-vs-whole-row dequant test, no failures. The log has no summary line. | t1-final-turbo.log |
| `./build-ui/bin/test-quantize-fns` | exit 0. turbo2, turbo3, turbo4, turbo5 and turbo6 are reported "skipped: rotated-domain KV quant". No quantization pass count is claimed for turbo2-4. | t1-final-quant.log |

## Notes
- The flag test script (tests/kv-dtypes/test_ctk_flags.sh) is committed on tt/T1 at 84643f49a, not on this branch.
- Turbo2, turbo3 and turbo4 are accepted by `-ctk` and `-ctv`, and turbo9 is rejected with "Unsupported cache type: turbo9" (exit 1).
- The quantize test does not exercise turbo2-4 quantization, so Phase 1 makes no quantization claim beyond the round-trip checks above.
