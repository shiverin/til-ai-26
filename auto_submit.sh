#!/usr/bin/env bash
# auto_submit.sh — Re-submit a pre-built TIL Docker image on a fixed interval.
#
# Usage:
#   ./auto_submit.sh <challenge> <tag> <interval>
#
#   <challenge>  TIL challenge name passed to `til submit` (ae, cv, nlp, asr).
#   <tag>        Docker tag already built locally via `til build <challenge> <tag>`.
#   <interval>   Sleep between submissions. Anything `sleep` accepts:
#                "30m", "1h", "90", "2h30m".
#
# Behaviour:
#   - Fires the first submission immediately, then sleeps <interval>, repeat.
#   - Logs every action to stdout with an ISO timestamp.
#   - Continues on failure (logs the exit code, sleeps, retries).
#   - Ctrl-C exits cleanly and reports the total number of submissions made.
#
# To run unattended:
#   nohup ./auto_submit.sh ae neural-smoke-20260524-1830 30m \
#     > auto_submit.log 2>&1 &
#   tail -f auto_submit.log
#
# To stop:
#   Foreground: Ctrl-C.
#   Background: `kill <pid>` — the trap still fires and prints the summary.

set -u

if [[ $# -ne 3 ]]; then
    echo "usage: $0 <challenge> <tag> <interval>" >&2
    echo "example: $0 ae neural-smoke-20260524-1830 30m" >&2
    exit 1
fi
CHALLENGE=$1
TAG=$2
INTERVAL=$3

export TIL_FOLDER="${TIL_FOLDER:-/home/jupyter/til-ai-26}"

count=0
trap 'echo "[$(date -Iseconds)] stopped after $count submission(s)"; exit 0' INT TERM

echo "[$(date -Iseconds)] auto_submit: challenge=$CHALLENGE tag=$TAG interval=$INTERVAL TIL_FOLDER=$TIL_FOLDER"
while :; do
    count=$((count + 1))
    echo "[$(date -Iseconds)] submission #$count: til submit $CHALLENGE $TAG"
    if til submit "$CHALLENGE" "$TAG"; then
        echo "[$(date -Iseconds)] submission #$count OK"
    else
        rc=$?
        echo "[$(date -Iseconds)] submission #$count FAILED (exit $rc), continuing"
    fi
    echo "[$(date -Iseconds)] sleeping $INTERVAL"
    sleep "$INTERVAL"
done
