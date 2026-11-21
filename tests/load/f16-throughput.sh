#!/usr/bin/env bash
# Gate Q21: f16 throughput vs baseline (pp512 and tg128, 5 reps, median).
# usage: f16-throughput.sh <build-dir> <baseline-dir> <cpu|cuda>
# With F16_RECORD=1 the build's samples are written to <baseline-dir> instead.
set -u

BUILD=$1
BASE=$2
BACKEND=$3
MODEL=${F16_MODEL:-/home/preeti/llm/models/ggml-org/Qwen3-0.6B-Base-GGUF/ggml-org/Qwen3-0.6B-Base-GGUF/Qwen3-0.6B-Base-Q8_0.gguf}
BENCH="$BUILD/bin/llama-bench"

# run 5 reps of one test; print one tok/s sample per line
bench_samples() {
    local args=$1
    local out
    if [ "$BACKEND" = cpu ]; then
        out=$(taskset -c 0-9 "$BENCH" -m "$MODEL" $args -t 10 -ngl 0 -r 5 -o json) || return 1
    else
        out=$("$BENCH" -m "$MODEL" $args -ngl 99 -r 5 -o json) || return 1
    fi
    printf '%s' "$out" | jq -r '.[].samples_ts[]'
}

# print "median spread%" for a sample file; spread = (max - min) / median * 100
stats() {
    sort -g "$1" | awk 'NF {a[++n] = $1}
        END {
            if (n == 0) exit 1
            m = (n % 2) ? a[(n + 1) / 2] : (a[n / 2] + a[n / 2 + 1]) / 2
            printf "%.2f %.2f\n", m, (a[n] - a[1]) / m * 100
        }'
}

# CUDA prompt run is 4096 tokens: 512 tokens gives 12-16% spread on GB10, 4096 about 1%
PP=512
[ "$BACKEND" = cuda ] && PP=4096
TESTS=("pp${PP}|-p $PP -n 0" "tg128|-p 0 -n 128")

if [ "${F16_RECORD:-0}" = 1 ]; then
    mkdir -p "$BASE"
    for t in "${TESTS[@]}"; do
        name=${t%%|*}; args=${t#*|}
        bench_samples "$args" > "$BASE/$name.samples" || { echo "record $name failed"; exit 1; }
        read -r med spr < <(stats "$BASE/$name.samples")
        echo "record $BACKEND $name median=$med spread=$spr%"
    done
    exit 0
fi

fail=0
for t in "${TESTS[@]}"; do
    name=${t%%|*}; args=${t#*|}
    if [ ! -s "$BASE/$name.samples" ]; then
        echo "Q21 $BACKEND $name: NOT PASSED (no baseline in $BASE)"
        fail=1
        continue
    fi
    read -r base_med _ < <(stats "$BASE/$name.samples")

    cand="$BASE/.cand-$BACKEND-$name.samples"
    rerun=no
    bench_samples "$args" > "$cand" || { echo "bench $name failed"; exit 1; }
    read -r cand_med spr < <(stats "$cand")
    if awk -v s="$spr" 'BEGIN {exit !(s > 5)}'; then
        rerun=yes
        bench_samples "$args" > "$cand" || { echo "bench $name failed"; exit 1; }
        read -r cand_med spr < <(stats "$cand")
    fi
    rm -f "$cand"

    ratio=$(awk -v c="$cand_med" -v b="$base_med" 'BEGIN {printf "%.4f", c / b}')
    if awk -v s="$spr" 'BEGIN {exit !(s > 5)}'; then
        verdict="NOT PASSED (not evaluable: spread above 5% after rerun)"
        fail=1
    elif awk -v r="$ratio" 'BEGIN {exit !(r >= 0.95)}'; then
        verdict=PASS
    else
        verdict=FAIL
        fail=1
    fi
    echo "Q21 $BACKEND $name median=$cand_med baseline=$base_med ratio=$ratio spread=$spr% rerun=$rerun verdict=$verdict"
done

if [ "$fail" = 0 ]; then
    echo "Q21 $BACKEND: PASS"
else
    echo "Q21 $BACKEND: FAIL"
fi
exit $fail
