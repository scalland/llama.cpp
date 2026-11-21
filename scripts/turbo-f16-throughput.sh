#!/usr/bin/env bash
#
# Throughput comparison for the default f16 KV cache path (goal G6, decision D-T11).
#
# usage: LLAMA_BENCH_MODEL=/path/to/model.gguf \
#        scripts/turbo-f16-throughput.sh <candidate-build-dir> <baseline-build-dir>
#
# Runs llama-bench (pp512 and tg128, 5 repetitions) from each build directory
# with the f16 K and V cache types set explicitly. Prints the median and spread
# ((max - min) / median, in percent) for each build and each test, and the four
# spread values. Exit status:
#   0  candidate median >= 0.95 x baseline median for pp512 and tg128, and all
#      four spreads <= 5 percent (after at most one rerun of both sets)
#   1  the ratio check failed, or a spread is still above 5 percent after the rerun
#   2  usage error, missing build directory or missing llama-bench binary

set -u -o pipefail

readonly REPS=5
readonly PP_TOKENS=512
readonly TG_TOKENS=128
readonly MIN_RATIO=0.95
readonly MAX_SPREAD=5

usage() {
    echo "usage: LLAMA_BENCH_MODEL=<model.gguf> $0 <candidate-build-dir> <baseline-build-dir>" >&2
}

if [ $# -ne 2 ]; then
    usage
    exit 2
fi

candidate_dir=$1
baseline_dir=$2

if [ -z "${LLAMA_BENCH_MODEL:-}" ]; then
    echo "error: LLAMA_BENCH_MODEL is not set" >&2
    usage
    exit 2
fi

if [ ! -f "$LLAMA_BENCH_MODEL" ]; then
    echo "error: model file not found: $LLAMA_BENCH_MODEL" >&2
    exit 2
fi

for dir in "$candidate_dir" "$baseline_dir"; do
    if [ ! -d "$dir" ]; then
        echo "error: build directory not found: $dir" >&2
        exit 2
    fi
    if [ ! -x "$dir/bin/llama-bench" ]; then
        echo "error: llama-bench not found or not executable: $dir/bin/llama-bench" >&2
        exit 2
    fi
done

work_dir=$(mktemp -d "${TMPDIR:-/tmp}/turbo-f16-throughput.XXXXXX") || exit 2
trap 'rm -rf "$work_dir"' EXIT

# Run llama-bench once for one build directory. Writes JSON lines to $2.
run_bench() {
    local dir=$1 out=$2
    "$dir/bin/llama-bench" \
        -m "$LLAMA_BENCH_MODEL" \
        -p "$PP_TOKENS" -n "$TG_TOKENS" \
        -r "$REPS" \
        -ctk f16 -ctv f16 \
        -o jsonl > "$out"
}

# Turn llama-bench JSON lines into "<test> <sample> <sample> ..." lines.
# Prompt-only rows (n_gen 0) are named ppN, generation-only rows tgN.
extract_samples() {
    awk '{
        np = -1; ng = -1; s = $0
        if (match(s, /"n_prompt": *[0-9]+/)) { t = substr(s, RSTART, RLENGTH); sub(/.*: */, "", t); np = t + 0 }
        if (match(s, /"n_gen": *[0-9]+/))    { t = substr(s, RSTART, RLENGTH); sub(/.*: */, "", t); ng = t + 0 }
        if (match(s, /"samples_ts": *\[[^]]*\]/)) {
            t = substr(s, RSTART, RLENGTH)
            sub(/.*\[/, "", t); sub(/\].*/, "", t); gsub(/,/, " ", t)
            name = (ng == 0) ? "pp" np : "tg" ng
            print name, t
        }
    }' "$1"
}

# Reduce "<test> <samples...>" lines to "<test> <median> <spread_percent>".
summarise() {
    awk '{
        n = NF - 1
        for (i = 1; i <= n; i++) {
            x = $(i + 1) + 0
            j = i - 1
            while (j >= 1 && a[j] > x) { a[j + 1] = a[j]; j-- }
            a[j + 1] = x
        }
        if (n % 2 == 1) med = a[(n + 1) / 2]
        else            med = (a[n / 2] + a[n / 2 + 1]) / 2
        spread = (a[n] - a[1]) / med * 100
        printf "%s %.4f %.4f\n", $1, med, spread
        delete a
    }' "$1"
}

# Run both builds and write summaries to $1 (base) and $2 (cand).
# Arguments: $3 tag used for file names.
run_set() {
    local tag=$1
    run_bench "$baseline_dir" "$work_dir/base-$tag.jsonl" || return 1
    run_bench "$candidate_dir" "$work_dir/cand-$tag.jsonl" || return 1
    extract_samples "$work_dir/base-$tag.jsonl" > "$work_dir/base-$tag.samples"
    extract_samples "$work_dir/cand-$tag.jsonl" > "$work_dir/cand-$tag.samples"
    summarise "$work_dir/base-$tag.samples" > "$work_dir/base-$tag.sum"
    summarise "$work_dir/cand-$tag.samples" > "$work_dir/cand-$tag.sum"
    if [ ! -s "$work_dir/base-$tag.sum" ] || [ ! -s "$work_dir/cand-$tag.sum" ]; then
        echo "error: no samples parsed from llama-bench output" >&2
        return 1
    fi
}

# Print the figures for one tag and set the global result variables.
report() {
    local tag=$1
    echo "== run: $tag (${REPS} repetitions each) =="
    echo "-- baseline: $baseline_dir"
    while read -r name med spread; do
        printf '   %-8s median %12s tok/s  spread %8s %%\n' "$name" "$med" "$spread"
    done < "$work_dir/base-$tag.sum"
    echo "-- candidate: $candidate_dir"
    while read -r name med spread; do
        printf '   %-8s median %12s tok/s  spread %8s %%\n' "$name" "$med" "$spread"
    done < "$work_dir/cand-$tag.sum"
}

# Return 0 when all four spreads are at most MAX_SPREAD for this tag.
spreads_ok() {
    local tag=$1
    awk -v max="$MAX_SPREAD" '{ if ($3 + 0 > max) bad = 1 } END { exit bad ? 1 : 0 }' \
        "$work_dir/base-$tag.sum" "$work_dir/cand-$tag.sum"
}

# Print the four spread values for this tag.
print_spreads() {
    local tag=$1
    while read -r name med spread; do
        echo "spread baseline $name: $spread %"
    done < "$work_dir/base-$tag.sum"
    while read -r name med spread; do
        echo "spread candidate $name: $spread %"
    done < "$work_dir/cand-$tag.sum"
}

# Return 0 when candidate median >= MIN_RATIO x baseline median for every test.
ratio_ok() {
    local tag=$1
    awk -v min="$MIN_RATIO" '
        FNR == NR { base[$1] = $2; next }
        {
            ratio = (base[$1] > 0) ? $2 / base[$1] : 0
            printf "ratio candidate/baseline %s: %.4f (need >= %s)\n", $1, ratio, min
            if (ratio < min) bad = 1
            seen++
        }
        END { if (seen != 2) bad = 1; exit bad ? 1 : 0 }' \
        "$work_dir/base-$tag.sum" "$work_dir/cand-$tag.sum"
}

if ! run_set first; then
    echo "error: llama-bench run failed" >&2
    exit 2
fi
report first
print_spreads first

if spreads_ok first; then
    final=first
else
    echo "one or more spreads above ${MAX_SPREAD} percent: rerunning both sets once"
    if ! run_set rerun; then
        echo "error: llama-bench rerun failed" >&2
        exit 2
    fi
    report rerun
    print_spreads rerun
    if ! spreads_ok rerun; then
        echo "NOT PASSED: spread still above ${MAX_SPREAD} percent after the rerun"
        exit 1
    fi
    final=rerun
fi

if ratio_ok "$final"; then
    echo "PASSED: candidate median is at least ${MIN_RATIO} x baseline median and all spreads are within ${MAX_SPREAD} percent"
    exit 0
fi

echo "NOT PASSED: candidate median is below ${MIN_RATIO} x baseline median"
exit 1
