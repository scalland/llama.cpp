#!/usr/bin/env bash
# Checks that -ctk/-ctv accept turbo2, turbo3, turbo4 and reject unknown names.
# Usage: test_ctk_flags.sh [path/to/llama-cli]   (or set LLAMA_BIN)

set -u

script_dir=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$script_dir/../.." && pwd)
bin=${1:-${LLAMA_BIN:-$root/build-ci/bin/llama-cli}}

if [ ! -x "$bin" ]; then
    echo "FAIL: binary not found or not executable: $bin"
    exit 1
fi

fails=0

# --help exits 0 only when argument parsing succeeded
check_accept() {
    local flag=$1 type=$2 out rc
    out=$("$bin" "$flag" "$type" --help 2>&1)
    rc=$?
    if [ $rc -eq 0 ]; then
        echo "PASS: $flag $type accepted"
    else
        echo "FAIL: $flag $type rejected (exit $rc)"
        echo "$out" | tail -n 3
        fails=$((fails + 1))
    fi
}

check_reject() {
    local flag=$1 type=$2 out rc
    out=$("$bin" "$flag" "$type" --help 2>&1)
    rc=$?
    if [ $rc -ne 0 ] && printf '%s\n' "$out" | grep -qF "Unsupported cache type: $type"; then
        echo "PASS: $flag $type rejected (exit $rc)"
    else
        echo "FAIL: $flag $type not rejected as expected (exit $rc)"
        echo "$out" | tail -n 3
        fails=$((fails + 1))
    fi
}

for flag in -ctk -ctv; do
    for type in turbo2 turbo3 turbo4; do
        check_accept "$flag" "$type"
    done
    check_reject "$flag" turbo9
done

if [ $fails -ne 0 ]; then
    echo "FAILED: $fails check(s) failed"
    exit 1
fi
echo "OK: all -ctk/-ctv checks passed"
exit 0
