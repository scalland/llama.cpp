#!/usr/bin/env bash
# Gate Q23: turbo session save/load soak with an RSS check.
# usage: turbo-soak.sh <build-dir> <model.gguf> <seconds>
# Fails if the run fails, or if RSS at the end is more than 5% above RSS at 5 min.
set -u

BUILD=$1
MODEL=$2
SECS=$3
BIN="$BUILD/bin/test-turbo-session-contract"

if [ ! -x "$BIN" ]; then
    echo "turbo-soak: missing $BIN"
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "turbo-soak: missing model $MODEL"
    exit 1
fi

log=$(mktemp)
"$BIN" --soak "$MODEL" "$SECS" > "$log" 2>&1
rc=$?
cat "$log"

start_kb=$(awk '$1 == "rss" && $2 == "t=300" {sub("kb=", "", $3); print $3}' "$log")
end_kb=$(awk '$1 == "rss_end" {sub("kb=", "", $3); print $3}' "$log")
rm -f "$log"

rss_ok=0
if [ -z "$start_kb" ] || [ -z "$end_kb" ]; then
    echo "turbo-soak: RSS samples missing (run shorter than 300 s?)"
elif awk -v s="$start_kb" -v e="$end_kb" 'BEGIN {exit !(e <= s * 1.05)}'; then
    rss_ok=1
fi
growth=$(awk -v s="${start_kb:-0}" -v e="${end_kb:-0}" 'BEGIN {if (s > 0) printf "%.2f", (e - s) / s * 100; else print "n/a"}')
echo "turbo-soak: RSS at 5 min = ${start_kb:-n/a} kB, at end = ${end_kb:-n/a} kB, growth = ${growth}%"

if [ "$rc" -ne 0 ] || [ "$rss_ok" -ne 1 ]; then
    echo "turbo-soak: FAIL (run exit $rc, rss_ok=$rss_ok)"
    exit 1
fi
echo "turbo-soak: PASS (run exit $rc)"
exit 0
