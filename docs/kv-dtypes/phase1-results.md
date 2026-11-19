# Phase 1 results: turbo2, turbo3, turbo4 on the TheTom base

## Build
- Base: TheTom llama-cpp-turboquant, commit 4185b90a0.
- Tested commit: 1f962792b (branch tt/T1), which drops the workbox[hash].js requirement in tools/ui/embed.cpp so the UI embeds from the b12498 bundle.
- Build command (GB10 node 192.168.1.112, CUDA arch 121):
  `cmake -S . -B build-ui -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_UI=ON -DLLAMA_USE_PREBUILT_UI=ON`
  with `HF_UI_VERSION=b12498`, then `cmake --build build-ui -j 20 --target llama-server llama-cli test-backend-ops test-turbo-quant test-quantize-fns`
- Result: BUILD_EXIT=0, BUILD_SECONDS=193 (log: build-T1-final.log).

## Test commands and pass counts
| Command | Result | Log |
|---|---|---|
| `bash tests/kv-dtypes/test_ctk_flags.sh build-ui/bin/llama-cli` | 8/8 PASS, EXIT=0 | t1-exit-flags.log |
| `./build-ui/bin/test-turbo-quant` | EXIT=0; 35 "ok" lines in the chunked-vs-whole-row dequant test, no failures. The log has no summary line. | t1-exit-turbo.log |
| `./build-ui/bin/test-quantize-fns` | EXIT=0. turbo2, turbo3, turbo4, turbo5 and turbo6 are reported "skipped: rotated-domain KV quant". No quantization pass count is claimed for turbo2-4. | t1-exit-quant.log |

## Notes
- The flag test script (tests/kv-dtypes/test_ctk_flags.sh) is committed on tt/T1 at 84643f49a, not on this branch.
- Turbo2, turbo3 and turbo4 are accepted by `-ctk` and `-ctv`. turbo9 is rejected: `llama-cli -ctk turbo9` prints "Unsupported cache type: turbo9" (t1-exit-flags.log).
- The quantize test does not exercise turbo2-4 quantization, so Phase 1 makes no quantization claim beyond the round-trip checks above.
