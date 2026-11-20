#!/usr/bin/env bash
#
# End-to-end checks for the turbo KV cache types (turbo2, turbo3, turbo4).
#
# Usage: turbo-e2e.sh <llama-cli binary> <model path>
#
# The script does not build anything. It runs the built llama-cli binary on
# the given model and exits non-zero if any case fails.

set -u

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <llama-cli binary> <model path>" >&2
    exit 2
fi

CLI="$1"
MODEL="$2"

if [ ! -x "$CLI" ]; then
    echo "error: llama-cli binary not found or not executable: $CLI" >&2
    exit 2
fi

if [ ! -f "$MODEL" ]; then
    echo "error: model file not found: $MODEL" >&2
    exit 2
fi

WORK_DIR="$(mktemp -d)" || exit 2
trap 'rm -rf "$WORK_DIR"' EXIT

PROMPT="The quick brown fox"
SUPPORTED_SET="turbo2 turbo3 turbo4"

FAILURES=0
CASES=0

pass() {
    echo "PASS $1"
}

fail() {
    echo "FAIL $1: $2"
    FAILURES=$((FAILURES + 1))
}

# Fails the case if the combined output and log contain nan or inf as whole words.
check_no_nan_inf() {
    local file="$1"
    if grep -q -i -w -E 'nan|inf' "$file"; then
        return 1
    fi
    return 0
}

# Runs llama-cli with the given arguments. Output and log go to one file.
# Sets RUN_RC and RUN_OUT.
run_cli() {
    local name="$1"
    shift
    RUN_OUT="$WORK_DIR/$name.log"
    "$CLI" -m "$MODEL" -p "$PROMPT" -n 64 -no-cnv "$@" >"$RUN_OUT" 2>&1 </dev/null
    RUN_RC=$?
}

# Case: a 64-token generation with one turbo type as K (f16 V), flash attention on.
case_turbo_k() {
    local type="$1"
    local name="turbo_k_${type}"
    CASES=$((CASES + 1))
    run_cli "$name" -fa on -ctk "$type" -ctv f16
    if [ "$RUN_RC" -ne 0 ]; then
        fail "$name" "exit code $RUN_RC (expected 0)"
    elif ! check_no_nan_inf "$RUN_OUT"; then
        fail "$name" "nan or inf found in output or log"
    else
        pass "$name"
    fi
}

# Case: a 64-token generation with one turbo type as V (f16 K), flash attention on.
case_turbo_v() {
    local type="$1"
    local name="turbo_v_${type}"
    CASES=$((CASES + 1))
    run_cli "$name" -fa on -ctk f16 -ctv "$type"
    if [ "$RUN_RC" -ne 0 ]; then
        fail "$name" "exit code $RUN_RC (expected 0)"
    elif ! check_no_nan_inf "$RUN_OUT"; then
        fail "$name" "nan or inf found in output or log"
    else
        pass "$name"
    fi
}

# Case: turbo5 is refused at the CLI, and stderr names the supported set.
case_turbo5_refused() {
    local name="turbo5_refused"
    CASES=$((CASES + 1))
    run_cli "$name" -fa on -ctk turbo5 -ctv f16
    if [ "$RUN_RC" -eq 0 ]; then
        fail "$name" "exit code 0 (expected non-zero)"
        return
    fi
    local t
    for t in $SUPPORTED_SET; do
        if ! grep -q -w "$t" "$RUN_OUT"; then
            fail "$name" "stderr does not name supported type $t"
            return
        fi
    done
    pass "$name"
}

# Case: turbo K with flash attention off exits 0 and logs the warning.
case_turbo_k_fa_off() {
    local name="turbo_k_fa_off"
    CASES=$((CASES + 1))
    run_cli "$name" -fa off -ctk turbo3 -ctv f16
    if [ "$RUN_RC" -ne 0 ]; then
        fail "$name" "exit code $RUN_RC (expected 0)"
    elif ! grep -q "TurboQuant K cache without flash_attn" "$RUN_OUT"; then
        fail "$name" "warning not found in log"
    else
        pass "$name"
    fi
}

# Case: turbo V with flash attention off exits non-zero.
case_turbo_v_fa_off() {
    local name="turbo_v_fa_off"
    CASES=$((CASES + 1))
    run_cli "$name" -fa off -ctk f16 -ctv turbo3
    if [ "$RUN_RC" -eq 0 ]; then
        fail "$name" "exit code 0 (expected non-zero)"
    else
        pass "$name"
    fi
}

for type in turbo2 turbo3 turbo4; do
    case_turbo_k "$type"
    case_turbo_v "$type"
done
case_turbo5_refused
case_turbo_k_fa_off
case_turbo_v_fa_off

echo "cases: $CASES, failures: $FAILURES"
if [ "$FAILURES" -ne 0 ]; then
    exit 1
fi
exit 0
