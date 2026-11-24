#!/usr/bin/env bash
# Shared helpers (sourced by the run_*.sh launchers).
#   env overrides: GPUS=0  PY=python  SEEDS="1 2 3"  MODELS="Qwen/Qwen2.5-7B mistralai/Mistral-7B-v0.3"
#                  GRAD_CKPT=0 (no gradient checkpointing: ~25% faster, ~60-80 GB peak; default 1 is safe on 96 GB)
#                  DRY_RUN=1 (only write the job list)   WAIT=1 (run the queue in the foreground)
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$DIR/.." && pwd)"
cd "$ROOT"
B=commonsense_reasoning
GPUS="${GPUS:-0}"                 # single RTX Pro 6000 (96 GB)
PY="${PY:-python}"
SEEDS="${SEEDS:-1 2 3}"
MODELS="${MODELS:-Qwen/Qwen2.5-7B}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/scratches/hypatia/ae606/data/huggingface}"   # models + datasets on scratch (home quota is small)

# one 7B run at a time on the single GPU (runs are ordered seed by seed: a 1-seed table is available early)
COST="${COST:-1.0}"
COMMON_ARGS="--batch_size 16 --num_evals 8 --max_len 384"
[[ "${GRAD_CKPT:-1}" == 1 ]] && COMMON_ARGS="$COMMON_ARGS --grad_ckpt"

# one learning rate per method (standard values; no LR grid on a single GPU)
LR_LORA="${LR_LORA:-2e-4}"     # LoRA, DoRA and all alpha-LoRA variants (matched)
LR_PLUS="${LR_PLUS:-5e-5}"     # LoRA+ base lr (lr_B = 16 x) and its alpha combinations
LR_IA3="${LR_IA3:-3e-3}"

# the algorithms (shared by all launchers)
GRAD="--method lora --alpha grad --T 10 --lr_alpha 5e-3"                                    # alpha-LoRA-grad (paper)
GRADV2="--method lora --alpha grad --T 10 --lr_alpha 5e-3 --alpha_data heldout --alpha_start 0.2 --alpha_param log --alpha_accum 2"  # proposed
# alpha-LoRA-theory: ONE global alpha (the scalar alpha of the theory). Per-layer / per-module alpha hit the trust region
# on almost every group and broke the model after the update (ablation: dev 90.5 -> 66, final test 85.7 / 78.2), while
# the global alpha moves by < 0.07, keeps improving dev and has the best dev accuracy of all theory variants.
THEORY="--method lora --alpha theory --theory_at 0.3,0.6 --theory_groups global"

init_exp() {
  EXP="$1"; RES="$B/results/$EXP"; LOGS="$B/logs/$EXP"; JOBS="$LOGS/jobs.tsv"
  mkdir -p "$RES" "$LOGS"; : > "$JOBS"
}

# add_job RUN_ID [train.py args...]
add_job() {
  local rid="$1"; shift
  rid="${rid//+/plus}"
  printf '%s\t%s\t%s\t%s\n' "$rid" "$RES" "$COST" \
    "$PY $B/train.py --out_dir $RES --run_id $rid $COMMON_ARGS $*" >> "$JOBS"
}

mshort() { basename "$1"; }

prefetch() {   # prefetch MODEL...
  [[ "${DRY_RUN:-0}" == 1 ]] && return 0
  "$PY" - "$@" <<'PYEOF'
import sys
sys.path.insert(0, "commonsense_reasoning")
from huggingface_hub import snapshot_download
from data import prefetch
prefetch()
for m in sys.argv[1:]:
    snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken", "tokenizer*"])
print("prefetch ok")
PYEOF
}

launch() {
  local n; n=$(wc -l < "$JOBS" | tr -d ' ')
  echo "[$EXP] $n jobs -> $JOBS"
  if [[ "${DRY_RUN:-0}" == 1 ]]; then echo "[$EXP] DRY_RUN, first jobs:"; head -n 3 "$JOBS" | cut -f4; return 0; fi
  if pgrep -f "run_queue.py --jobs $JOBS" > /dev/null; then echo "[$EXP] queue already running"; return 0; fi
  if [[ "${WAIT:-0}" == 1 ]]; then
    "$PY" $B/run_queue.py --jobs "$JOBS" --gpus "$GPUS" --log_dir "$LOGS" 2>&1 | tee -a "$LOGS/_queue.log"
    "$PY" $B/summarize.py "$RES" --quiet || true
    return 0
  fi
  nohup "$PY" $B/run_queue.py --jobs "$JOBS" --gpus "$GPUS" --log_dir "$LOGS" >> "$LOGS/_queue.log" 2>&1 &
  echo "[$EXP] queue started (PID $!).  Monitor: bash $B/monitor.sh $EXP"
  echo "[$EXP] stop: pkill -f 'run_queue.py --jobs $JOBS'; pkill -f 'out_dir $RES'"
}
