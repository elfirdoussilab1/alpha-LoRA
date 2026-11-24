#!/usr/bin/env bash
# Live view: queue progress, GPU usage, failures, running jobs and the current result tables.
#   bash commonsense_reasoning/monitor.sh main        # refresh every 60 s (Ctrl-C quits the view, jobs keep running)
#   bash commonsense_reasoning/monitor.sh ablation 20
#   ONCE=1 bash commonsense_reasoning/monitor.sh scarcity
#   bash commonsense_reasoning/monitor.sh all         # progress + headline numbers (Avg, gain over LoRA) of every experiment
# Full tables: commonsense_reasoning/results/<exp>/summary.md ; figures: commonsense_reasoning/results/<exp>/plots/*.png
cd "$(dirname "$0")/.."
B=commonsense_reasoning
EXP="${1:?usage: monitor.sh <main|ablation|scarcity|small|all> [seconds]}"; INT="${2:-60}"; PY="${PY:-python}"
status_line() {
  local f="$B/logs/$1/_queue_status.json"
  [[ -f $f ]] && "$PY" -c "import json;s=json.load(open('$f'));print(f\"$1: {s['done']}/{s['total']} done | running {len(s['running'])} | pending {s['pending']} | failed {len(s['failed'])} | {s['time']}\")" \
              || echo "$1: not started"
}
while true; do
  clear 2>/dev/null || true
  echo "================ $EXP | $(date '+%F %T') ================"
  if [[ "$EXP" == all ]]; then
    for e in main ablation scarcity small; do
      status_line $e
      [[ -d "$B/results/$e" ]] && "$PY" $B/summarize.py "$B/results/$e" --brief 2>/dev/null
    done
  else
    status_line "$EXP"
    command -v nvidia-smi > /dev/null && nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
       --format=csv,noheader | sed 's/^/  GPU /'
    grep FAILED "$B/logs/$EXP/_queue.log" 2>/dev/null | tail -n 5
    "$PY" $B/summarize.py "$B/results/$EXP" --quiet --plots 2>/dev/null
    head -n "${LINES_MAX:-80}" "$B/results/$EXP/summary.md" 2>/dev/null
  fi
  [[ "${ONCE:-0}" == 1 ]] && break
  sleep "$INT"
done
