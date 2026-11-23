#!/usr/bin/env bash
# bash glue_matched/monitor.sh [seconds]   (Ctrl-C quits the view; ONCE=1 for a single snapshot)
cd "$(dirname "$0")/.."
while true; do
  clear 2>/dev/null || true; echo "=========== glue_matched | $(date '+%F %T') ==========="
  command -v nvidia-smi > /dev/null && nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader | sed 's/^/GPU /'
  grep FAILED glue_matched/logs/_queue.log 2>/dev/null | tail -n 5
  "${PY:-python}" glue_matched/summarize.py 2>/dev/null | head -n "${LINES_MAX:-80}"
  [[ "${ONCE:-0}" == 1 ]] && break; sleep "${1:-60}"
done
