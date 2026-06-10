#!/usr/bin/env bash
# Tail the live stdout of whatever `train.py` is currently running.
# Auto-discovers the log file from the running process's open stdout fd, so
# you don't need to know which transient /var/tmp/.../output file is in use.
#
# Usage:
#   ./tail_training.sh
set -euo pipefail

# Find the train.py PID (prefer the parent, not the rfdetr DataLoader workers).
PID=$(pgrep -of "train\.py" || true)
if [[ -z "$PID" ]]; then
    echo "[tail] no running train.py found" >&2
    echo "[tail] did the training crash or finish? check weights/base/" >&2
    exit 1
fi

# stdout (fd 1) symlinks to the actual log file under /var/tmp/.../tasks/...
LOG=$(readlink -f "/proc/$PID/fd/1" 2>/dev/null || true)
if [[ -z "$LOG" || ! -f "$LOG" ]]; then
    echo "[tail] could not resolve stdout of PID $PID (process may have exited)" >&2
    exit 1
fi

echo "[tail] train.py PID=$PID  log=$LOG"
echo "[tail] tail -F (Ctrl-C to stop) — terminal control sequences will be stripped"
echo
# tail -F (capital) follows by name in case the file is rotated.
# Strip ANSI escapes and \r so tqdm/rich tables render plainly when read on a
# terminal that may not be a tty.
tail -F "$LOG" | tr -d '\r' | sed -u 's/\x1b\[[0-9;]*[a-zA-Z]//g'
