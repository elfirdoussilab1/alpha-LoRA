#!/usr/bin/env bash
# bash glue_matched/monitor.sh [seconds]   (Ctrl-C quits the view; ONCE=1 for a single snapshot)
# RES_DIR=glue_matched/results_all bash glue_matched/monitor.sh   # the all-methods run (run_glue_all.sh)
cd "$(dirname "$0")/.."
RES_DIR="${RES_DIR:-glue_matched/results}"; LOG_DIR="${RES_DIR/results/logs}"
while true; do
  clear 2>/dev/null || true; echo "=========== glue_matched | $(date '+%F %T') ==========="
  command -v nvidia-smi > /dev/null && nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader | sed 's/^/GPU /'
  grep FAILED $LOG_DIR/_queue.log 2>/dev/null | tail -n 5
  "${PY:-python}" glue_matched/summarize.py "$RES_DIR" 2>/dev/null | head -n "${LINES_MAX:-80}"
  [[ "${ONCE:-0}" == 1 ]] && break; sleep "${1:-60}"
done
