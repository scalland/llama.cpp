#!/usr/bin/env bash
# Gate Q30: a session file written by the baseline build must load on the candidate build.
# usage: bash tests/contract/session-compat.sh <build-dir under ~/gate-fix> <baseline-commit>
set -u

BUILD_DIR=${1:?usage: session-compat.sh <build-dir> <baseline-commit>}
BASE=${2:?usage: session-compat.sh <build-dir> <baseline-commit>}

NODE=${NODE:-preeti@192.168.1.112}
MODEL=${MODEL:-/home/preeti/llm/models/ggml-org/Qwen3-0.6B-Base-GGUF/ggml-org/Qwen3-0.6B-Base-GGUF/Qwen3-0.6B-Base-Q8_0.gguf}
ROOT=gate-fix                       # node paths, relative to home
SRC=$ROOT/src                       # candidate source (synced from this worktree)
CAND=$ROOT/$BUILD_DIR               # candidate build
BASE_SRC=$ROOT/compat-base-src      # baseline source from git archive
BASE_BUILD=$ROOT/build-compat-base  # baseline build, kept across runs
WORK=$ROOT/q30                      # scratch for session files
TEST=test-turbo-session-contract
LOG_REMOTE=gate-logs/Q30/Q30.log

rs() { ssh -o BatchMode=yes "$NODE" "$@"; }

step_ok()   { echo "STEP $1 PASS: $2"; }
step_fail() { echo "STEP $1 FAIL: $2"; exit 1; }

main() {
    REPO=$(git rev-parse --show-toplevel) || step_fail 0 "not inside a git worktree"
    cd "$REPO" || step_fail 0 "cannot enter $REPO"

    echo "== sync candidate source to $NODE:$SRC =="
    rsync -a --delete --exclude .git ./ "$NODE:$SRC/" || step_fail 0 "rsync failed"

    echo "== step 1: include/llama.h unchanged from $BASE =="
    if git diff --exit-code "$BASE" -- include/llama.h; then
        step_ok 1 "include/llama.h matches $BASE"
    else
        step_fail 1 "include/llama.h differs from $BASE"
    fi

    echo "== step 2: baseline build of $BASE =="
    if ! rs "test -d $BASE_SRC/include"; then
        rs "rm -rf $BASE_SRC && mkdir -p $BASE_SRC" || step_fail 2 "cannot create $BASE_SRC"
        git archive "$BASE" | ssh -o BatchMode=yes "$NODE" "tar -x -C $BASE_SRC" \
            || step_fail 2 "git archive of $BASE failed"
    fi
    rs "cd $BASE_SRC && mkdir -p tests" || step_fail 2 "baseline tree missing"
    scp -q tests/$TEST.cpp "$NODE:$BASE_SRC/tests/$TEST.cpp" || step_fail 2 "copy test source failed"
    if ! rs "grep -q '$TEST' $BASE_SRC/tests/CMakeLists.txt"; then
        printf '%s\n' "llama_build_and_test($TEST.cpp  LABEL \"main behavior\"  LINK llama-common)" \
            | ssh -o BatchMode=yes "$NODE" "cat >> $BASE_SRC/tests/CMakeLists.txt" \
            || step_fail 2 "cannot add test target to baseline"
    fi
    if ! rs "test -f $BASE_BUILD/CMakeCache.txt"; then
        rs "cmake -S $BASE_SRC -B $BASE_BUILD -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF" \
            || step_fail 2 "baseline configure failed"
    fi
    rs "cmake --build $BASE_BUILD --target $TEST -j 16" \
        || step_fail 2 "baseline does not build $TEST (baseline may lack a required API)"
    step_ok 2 "baseline $TEST built from $BASE"

    echo "== step 3: baseline writes a session file =="
    rs "mkdir -p $WORK && rm -f $WORK/baseline-session.bin && $BASE_BUILD/bin/$TEST --save $WORK/baseline-session.bin $MODEL" \
        || step_fail 3 "baseline --save failed"
    rs "test -s $WORK/baseline-session.bin" || step_fail 3 "session file missing"
    step_ok 3 "baseline session written to $WORK/baseline-session.bin"

    echo "== step 4: candidate loads the baseline session =="
    rs "cmake --build $CAND --target $TEST -j 16" || step_fail 4 "candidate build failed"
    rs "$CAND/bin/$TEST --load $WORK/baseline-session.bin $MODEL" \
        || step_fail 4 "candidate --load of baseline session failed"
    step_ok 4 "candidate loaded the baseline session"

    echo "== step 5: candidate creates a q8_0 K and V context =="
    local out
    out=$(rs "cd ~/$WORK && ~/$CAND/bin/$TEST $MODEL") || step_fail 5 "candidate default mode failed"
    echo "$out"
    if echo "$out" | grep -qF "OK: q8_0 K and V context created"; then
        step_ok 5 "q8_0 K and V context created through the C API"
    else
        step_fail 5 "q8_0 K and V context check line not found"
    fi

    echo "ALL STEPS PASS"
    return 0
}

TMP_LOG=$(mktemp)
main 2>&1 | tee "$TMP_LOG"
rc=${PIPESTATUS[0]}
rs "mkdir -p gate-logs/Q30 && cat > $LOG_REMOTE" < "$TMP_LOG" || echo "warning: could not save $LOG_REMOTE on node"
rm -f "$TMP_LOG"
exit "$rc"
