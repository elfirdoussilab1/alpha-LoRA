#!/usr/bin/env bash
# EVERYTHING FOR THE REBUTTAL in one command (start it inside tmux):   bash run_rebuttal.sh
# Two chains share the single GPU (the roberta-base jobs are small enough to run next to the LLM jobs):
#   LLM chain : commonsense reasoning, Qwen2.5-7B -- main table, data scarcity, ablation (finished runs are skipped,
#               identical configurations are reused from the run cache) -- then Qwen2.5-0.5B / 1.5B on the full test sets
#   GLUE chain: roberta-base, all 10 methods x RTE/MRPC/SST-2/QNLI x 3 seeds x 2 LRs (LR picked on dev)
# alpha-LoRA-theory uses ONE global alpha everywhere (see commonsense_reasoning/common.sh for why).
# Expected wall time: LLM chain ~5-6 h, GLUE chain ~12-20 h (both slower while they share the GPU).
# Re-running resumes. Ctrl-C stops both chains. All tables end up in rebuttal_results.md.
#   progress: bash commonsense_reasoning/monitor.sh all      |   RES_DIR=glue_matched/results_all bash glue_matched/monitor.sh
set -uo pipefail
cd "$(dirname "$0")"
VENV="${VENV:-/scratches/hypatia/ae606/venvs/biruni-3.12.3}"
[[ -n "${VIRTUAL_ENV:-}" ]] || source "$VENV/bin/activate"
PY="${PY:-python}"
"$PY" -c "import torch, peft, transformers, datasets" || { echo "missing packages in $("$PY" -c 'import sys; print(sys.prefix)')"; exit 1; }
"$PY" -c "import tabulate" 2>/dev/null || "$PY" -m pip install -q tabulate     # needed by both summarize.py
trap 'echo "stopping..."; kill 0' INT TERM

# the scarcity runs made with the old per-layer alpha-LoRA-theory are moved aside (kept, not deleted) so that the
# new global-alpha runs are not skipped by the queue nor mixed with them under the same label
"$PY" - <<'PYEOF'
import glob, json, os
src, dst = "commonsense_reasoning/results/scarcity", "commonsense_reasoning/results/scarcity_theory_layer"
for f in glob.glob(os.path.join(src, "*_aLoRA-theory_s*.json")):
    if f.endswith(".progress.json") or json.load(open(f))["args"].get("theory_groups") != "layer":
        continue
    os.makedirs(dst, exist_ok=True); os.replace(f, os.path.join(dst, os.path.basename(f)))
    print("moved per-layer theory run aside:", os.path.basename(f))
PYEOF

mkdir -p commonsense_reasoning/logs glue_matched/logs_all
( bash commonsense_reasoning/run_all.sh && bash commonsense_reasoning/run_small.sh ) \
  > commonsense_reasoning/logs/rebuttal_llm.log 2>&1 &
LLM=$!
WAIT=1 bash glue_matched/run_glue_all.sh > glue_matched/logs_all/rebuttal_glue.log 2>&1 &
GLUE=$!
echo "[$(date '+%F %T')] LLM chain  (PID $LLM)  log: commonsense_reasoning/logs/rebuttal_llm.log"
echo "[$(date '+%F %T')] GLUE chain (PID $GLUE) log: glue_matched/logs_all/rebuttal_glue.log"
wait $LLM;  echo "[$(date '+%F %T')] LLM chain finished (rc=$?)"
wait $GLUE; echo "[$(date '+%F %T')] GLUE chain finished (rc=$?)"

# collect every table
OUT=rebuttal_results.md
{
  echo "# Rebuttal results ($(date '+%F %T'))"
  for e in main scarcity ablation small; do
    d=commonsense_reasoning/results/$e
    [[ -d $d ]] || continue
    "$PY" commonsense_reasoning/summarize.py $d --quiet --plots
    echo; echo "## Commonsense reasoning: $e"; echo; cat $d/summary.md
  done
  echo; echo "## GLUE (roberta-base), all methods"; echo
  "$PY" glue_matched/summarize.py glue_matched/results_all
} > $OUT 2>&1
echo "ALL DONE -> $OUT   (plots: commonsense_reasoning/results/*/plots/)"
