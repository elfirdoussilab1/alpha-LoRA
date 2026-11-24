#!/usr/bin/env bash
# ALL METHODS on GLUE (roberta-base), same protocol as run_glue_matched.sh (fixed dev split carved out of train for
# checkpoint / LR selection, official validation set = test, same seeds and data for every method):
#   LoRA | DoRA | IA3 | LoRA+ | alpha-LoRA-grad (paper) | alpha-LoRA-grad-v2 | alpha-LoRA-theory
#   | alpha-LoRA-grad + LoRA+ | alpha-LoRA-grad-v2 + LoRA+ | alpha-LoRA-theory + LoRA+
# Shared: r=8, lora_alpha=8 on query/value (+ classifier head), AdamW wd 0.01, linear schedule 6% warmup, batch 64.
# Every method gets the same small LR grid around its usual value; the summary picks each method's LR per task on DEV.
# 10 methods x 4 tasks x 3 seeds x 2 LRs = 240 runs, ~3 at a time on the GPU: roughly 10-20 h on 1 GPU.
# Runs in the foreground (start it inside tmux); re-running resumes (finished runs are skipped).
#
#   bash glue_matched/run_glue_all.sh
#   INCLUDE_LARGE=1 bash glue_matched/run_glue_all.sh            # + MNLI, QQP (3 epochs each)
#   METHODS="LoRA alpha-LoRA-theory" SEEDS="1" TASKS="rte mrpc" bash glue_matched/run_glue_all.sh
#   RES_DIR=glue_matched/results_all bash glue_matched/monitor.sh  # progress + live tables
#   WAIT=0 bash ...                                               # start the queue in the background instead
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python}"; GPUS="${GPUS:-0}"; SEEDS="${SEEDS:-1 2 3}"
TASKS="${TASKS:-rte mrpc sst2 qnli}"; [[ "${INCLUDE_LARGE:-0}" == 1 ]] && TASKS="$TASKS mnli qqp"
LRS="${LRS:-2e-4 5e-4}"             # LoRA, DoRA and every alpha-LoRA variant (matched)
LRS_PLUS="${LRS_PLUS:-2e-5 5e-5}"   # LoRA+ base lr (lr_B = 16 x) and its alpha combinations
LRS_IA3="${LRS_IA3:-3e-3 1e-2}"
ALL="LoRA DoRA IA3 LoRA+ alpha-LoRA-grad alpha-LoRA-grad-v2 alpha-LoRA-theory alpha-LoRA-grad+LoRA+ alpha-LoRA-grad-v2+LoRA+ alpha-LoRA-theory+LoRA+"
METHODS="${METHODS:-$ALL}"
COST="${COST:-0.34}"                # roberta-base is small: 3 runs share the GPU
RES=glue_matched/results_all; LOGS=glue_matched/logs_all; JOBS=$LOGS/jobs.tsv
mkdir -p $RES $LOGS; : > $JOBS
export HF_HOME="${HF_HOME:-/scratches/hypatia/ae606/data/huggingface}" TOKENIZERS_PARALLELISM=false
epochs() { case "$1" in rte|mrpc) echo 30 ;; sst2|qnli) echo 5 ;; *) echo 3 ;; esac; }

# the algorithms (same settings as commonsense_reasoning/common.sh)
GRAD="--alpha grad --T 10 --lr_alpha 5e-3"
GRADV2="--alpha grad --T 10 --lr_alpha 5e-3 --alpha_data heldout --alpha_start 0.2 --alpha_param log --alpha_accum 2"
THEORY="--alpha theory --theory_at 0.3,0.6 --theory_groups global"   # one global alpha (see commonsense_reasoning/common.sh)
PLUS="--loraplus_ratio 16"

# job LABEL "LR GRID" [train_glue.py args...]
job() {
  local label="$1" lrs="$2"; shift 2
  [[ " $METHODS " == *" $label "* ]] || return 0
  for lr in $lrs; do
    local rid="${task}_${label//+/plus}_lr${lr}_s${s}"
    printf '%s\t%s\t%s\t%s\n' "$rid" "$RES" "$COST" \
      "$PY glue_matched/train_glue.py --task $task --lr $lr --epochs $(epochs $task) --seed $s --label $label --out_dir $RES --run_id $rid $*" >> $JOBS
  done
}

[[ "${DRY_RUN:-0}" == 1 ]] || "$PY" -c "
from datasets import load_dataset; from huggingface_hub import snapshot_download
[load_dataset('nyu-mll/glue', t) for t in '$TASKS'.split()]
snapshot_download('FacebookAI/roberta-base', allow_patterns=['*.json','*.safetensors','*.txt'])
print('prefetch ok')"

for s in $SEEDS; do for task in $TASKS; do       # seed by seed: a full 1-seed table is available early
  job LoRA                     "$LRS"      --method lora
  job DoRA                     "$LRS"      --method dora
  job IA3                      "$LRS_IA3"  --method ia3
  job LoRA+                    "$LRS_PLUS" --method lora $PLUS
  job alpha-LoRA-grad          "$LRS"      --method lora $GRAD
  job alpha-LoRA-grad-v2       "$LRS"      --method lora $GRADV2
  job alpha-LoRA-theory        "$LRS"      --method lora $THEORY
  job alpha-LoRA-grad+LoRA+    "$LRS_PLUS" --method lora $GRAD $PLUS
  job alpha-LoRA-grad-v2+LoRA+ "$LRS_PLUS" --method lora $GRADV2 $PLUS
  job alpha-LoRA-theory+LoRA+  "$LRS_PLUS" --method lora $THEORY $PLUS
done; done
echo "$(wc -l < $JOBS | tr -d ' ') jobs -> $JOBS"
[[ "${DRY_RUN:-0}" == 1 ]] && { head -n 3 $JOBS | cut -f4; exit 0; }
if pgrep -f "run_queue.py --jobs $JOBS" > /dev/null; then echo "queue already running"; exit 0; fi
if [[ "${WAIT:-1}" == 1 ]]; then
  "$PY" glue_matched/run_queue.py --jobs $JOBS --gpus $GPUS --log_dir $LOGS 2>&1 | tee -a $LOGS/_queue.log
  "$PY" glue_matched/summarize.py $RES || true
  exit 0
fi
nohup "$PY" glue_matched/run_queue.py --jobs $JOBS --gpus $GPUS --log_dir $LOGS >> $LOGS/_queue.log 2>&1 &
echo "queue started (PID $!). Monitor: RES_DIR=$RES bash glue_matched/monitor.sh   Stop: pkill -f 'run_queue.py --jobs $JOBS'"
