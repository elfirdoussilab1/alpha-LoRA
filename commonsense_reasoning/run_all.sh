#!/usr/bin/env bash
# Everything back-to-back on the single GPU, most important first (~10-13 h with defaults; identical runs reused).
# Re-running resumes (finished runs are skipped).
#   bash commonsense_reasoning/run_all.sh 2>&1 | tee commonsense_reasoning/run_all.log

# For monitoring, use:
#   bash commonsense_reasoning/monitor.sh all
cd "$(dirname "$0")/.."
export WAIT=1
set -x
bash commonsense_reasoning/run_main.sh       # ~5-7 h   main comparison
bash commonsense_reasoning/run_scarcity.sh   # ~2-3 h   theory prediction (iii)
bash commonsense_reasoning/run_ablation.sh   # ~2.5-3 h design choices
echo "ALL DONE -> commonsense_reasoning/results/*/summary.md"
