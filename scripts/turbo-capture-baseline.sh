#!/usr/bin/env bash
#
# Capture the f16 default-path llama-bench baseline for the turbo KV port.
#
# The baseline is commit f8caa5ea3, the parent of cca3c1e96 (D-T11). The
# script checks it out in a detached git worktree, builds a CPU tree and a
# CUDA tree with the same Release flags as the quality matrix build, runs
# llama-bench 5 times per build, and writes a run record. It does not commit
# or push anything.
#
# Usage:  scripts/turbo-capture-baseline.sh <model.gguf>
# Env:    JOBS          parallel build jobs (default 8)
#         BASELINE_DIR  output directory (default <repo root>/baseline)
#
# Output:
#   <BASELINE_DIR>/src          detached worktree at f8caa5ea3
#   <BASELINE_DIR>/cpu          CPU build (GGML_CUDA=OFF)
#   <BASELINE_DIR>/cuda         CUDA build (GGML_CUDA=ON)
#   <BASELINE_DIR>/run-record.json

set -euo pipefail

BASE_COMMIT=f8caa5ea3
REPS=5
PROMPT_TOKENS=512
GEN_TOKENS=128
SPREAD_LIMIT=0.05
JOBS=${JOBS:-8}

die() {
    echo "error: $*" >&2
    exit 1
}

if [ "$#" -ne 1 ]; then
    echo "usage: $0 <model.gguf>" >&2
    exit 2
fi

MODEL=$(readlink -f -- "$1") || die "cannot resolve model path: $1"
[ -f "$MODEL" ] || die "model file not found: $1"

for tool in git cmake jq; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool not found: $tool"
done

REPO_ROOT=$(git rev-parse --show-toplevel) || die "not inside a git repository"
cd "$REPO_ROOT"

if ! git cat-file -e "${BASE_COMMIT}^{commit}" 2>/dev/null; then
    die "commit ${BASE_COMMIT} not found in this repository"
fi
BASE_FULL=$(git rev-parse "${BASE_COMMIT}^{commit}")

BASE_DIR=$(readlink -m -- "${BASELINE_DIR:-$REPO_ROOT/baseline}")
SRC_DIR="$BASE_DIR/src"
mkdir -p "$BASE_DIR"

# Fresh detached worktree at the baseline commit.
if [ -e "$SRC_DIR" ]; then
    git worktree remove --force "$SRC_DIR" 2>/dev/null || rm -rf "$SRC_DIR"
    git worktree prune
fi
git worktree add --detach "$SRC_DIR" "$BASE_FULL" >/dev/null

# Same Release flags as the quality matrix build (BUILD section of the card).
build_tree() {
    local name=$1 cuda=$2
    local bdir="$BASE_DIR/$name"
    local flags=(
        -DCMAKE_BUILD_TYPE=Release
        -DGGML_CUDA="$cuda"
        -DLLAMA_BUILD_TESTS=ON
        -DLLAMA_BUILD_EXAMPLES=ON
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    )
    echo "building $name (GGML_CUDA=$cuda) at $BASE_FULL" >&2
    cmake -S "$SRC_DIR" -B "$bdir" "${flags[@]}" >"$bdir.configure.log" 2>&1 \
        || die "configure failed for $name (see $bdir.configure.log)"
    cmake --build "$bdir" -j "$JOBS" --target llama-bench >"$bdir.build.log" 2>&1 \
        || die "build failed for $name (see $bdir.build.log)"
    [ -x "$bdir/bin/llama-bench" ] || die "llama-bench missing after build: $name"
}

# Run llama-bench REPS times per test and summarise each test.
bench_tree() {
    local name=$1 cuda=$2
    local bdir="$BASE_DIR/$name"
    local bin="$bdir/bin/llama-bench"
    echo "benchmarking $name" >&2
    "$bin" -m "$MODEL" -p "$PROMPT_TOKENS" -n "$GEN_TOKENS" -r "$REPS" -o json \
        >"$bdir/bench.json" 2>"$bdir/bench.log" \
        || die "llama-bench failed for $name (see $bdir/bench.log)"

    jq --arg name "$name" --arg cuda "$cuda" --arg bin "$bin" \
       --arg flags "CMAKE_BUILD_TYPE=Release GGML_CUDA=$cuda LLAMA_BUILD_TESTS=ON LLAMA_BUILD_EXAMPLES=ON" \
       --argjson limit "$SPREAD_LIMIT" '
        def med:
            sort as $s
            | length as $n
            | if $n == 0 then null
              elif $n % 2 == 1 then $s[($n / 2) | floor]
              else ($s[($n / 2) - 1] + $s[$n / 2]) / 2
              end;
        {
            build: $name,
            gpu_offload: ($cuda == "ON"),
            binary: $bin,
            build_flags: $flags,
            tests: [ .[] | (.samples_ts) as $s
                | ($s | med) as $m
                | {
                    test: (if .n_gen == 0 then "pp\(.n_prompt)" else "tg\(.n_gen)" end),
                    repetitions: ($s | length),
                    median_ts: $m,
                    min_ts: ($s | min),
                    max_ts: ($s | max),
                    spread: ((($s | max) - ($s | min)) / $m),
                    spread_above_limit: ((($s | max) - ($s | min)) / $m > $limit),
                    samples_ts: $s
                  }
              ]
        }' "$bdir/bench.json" >"$bdir/summary.json" \
        || die "could not summarise llama-bench output for $name"
}

build_tree cpu OFF
build_tree cuda ON
bench_tree cpu OFF
bench_tree cuda ON

GIT_DESCRIBE=$(git -C "$SRC_DIR" log -1 --format='%H %s' "$BASE_FULL")

jq -n \
    --arg commit "$BASE_FULL" \
    --arg subject "$GIT_DESCRIBE" \
    --arg model "$MODEL" \
    --arg captured_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --argjson reps "$REPS" \
    --argjson prompt_tokens "$PROMPT_TOKENS" \
    --argjson gen_tokens "$GEN_TOKENS" \
    --argjson spread_limit "$SPREAD_LIMIT" \
    --slurpfile cpu "$BASE_DIR/cpu/summary.json" \
    --slurpfile cuda "$BASE_DIR/cuda/summary.json" '
    {
        commit: $commit,
        commit_subject: $subject,
        model: $model,
        captured_at: $captured_at,
        repetitions: $reps,
        prompt_tokens: $prompt_tokens,
        gen_tokens: $gen_tokens,
        spread_limit: $spread_limit,
        cache_flags: "none (f16 default path)",
        builds: [ $cpu[0], $cuda[0] ]
    }' >"$BASE_DIR/run-record.json" \
    || die "could not write run record"

echo "baseline recorded at $BASE_DIR/run-record.json" >&2
echo "  commit: $BASE_FULL" >&2
echo "  cpu binary:  $BASE_DIR/cpu/bin/llama-bench" >&2
echo "  cuda binary: $BASE_DIR/cuda/bin/llama-bench" >&2
