#!/usr/bin/env bash
# LoRA vs alpha-LoRA on GLUE (roberta-base) with IDENTICAL hyper-parameters for both methods, all tasks and all seeds.
# Shared: LoRA r=8, lora_alpha=8 on query/value + full classifier head, AdamW (wd 0.01), linear schedule with 6% warmup,
#         batch 64, same epochs per task, same seeds, same dev split, checkpoint picked on dev (never on test).
# alpha-LoRA only: per-neuron alpha, Adam lr 5e-3, one step every T=10 on a fresh batch (Algorithm 1) -- same everywhere.
# alpha-LoRA-theory only: closed-form per-layer alpha at 30% and 60% of training, from a probe set held out of train.
# Each LR in LRS is run for BOTH methods ("matched LR"); with 2 LRs the summary also shows each method at its own
# best LR chosen on dev ("independently tuned"), which covers both requests of Reviewer LUNR.
#
#   bash glue_matched/run_glue_matched.sh                       # RTE, MRPC, SST-2, QNLI: 48 runs, ~2-4 h on 1 GPU
#   INCLUDE_LARGE=1 bash glue_matched/run_glue_matched.sh       # + MNLI, QQP (3 epochs each): +24 runs, ~+4-6 h
#   LRS="2e-4" SEEDS="1 2 3" TASKS="rte mrpc" bash ...          # overrides
#   METHODS="lora alpha-lora-theory" bash ...                   # subset of methods
#   WAIT=1 bash glue_matched/run_glue_matched.sh                # run the queue in the foreground (e.g. inside tmux)
#   bash glue_matched/monitor.sh                                # progress + live tables
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python}"; GPUS="${GPUS:-0}"; SEEDS="${SEEDS:-1 2 3}"; LRS="${LRS:-1e-4 2e-4}"
METHODS="${METHODS:-lora alpha-lora alpha-lora-theory}"
TASKS="${TASKS:-rte mrpc sst2 qnli}"; [[ "${INCLUDE_LARGE:-0}" == 1 ]] && TASKS="$TASKS mnli qqp"
COST="${COST:-0.34}"                      # roberta-base is small: 3 runs share the GPU
RES=glue_matched/results; LOGS=glue_matched/logs; JOBS=$LOGS/jobs.tsv
mkdir -p $RES $LOGS; : > $JOBS
epochs() { case "$1" in rte|mrpc) echo 30 ;; sst2|qnli) echo 5 ;; *) echo 3 ;; esac; }

[[ "${DRY_RUN:-0}" == 1 ]] || "$PY" -c "
from datasets import load_dataset; from huggingface_hub import snapshot_download
[load_dataset('nyu-mll/glue', t) for t in '$TASKS'.split()]
snapshot_download('FacebookAI/roberta-base', allow_patterns=['*.json','*.safetensors','*.txt'])
print('prefetch ok')"

for s in $SEEDS; do for task in $TASKS; do for lr in $LRS; do for method in $METHODS; do
  rid="${task}_${method}_lr${lr}_s${s}"
  printf '%s\t%s\t%s\t%s\n' "$rid" "$RES" "$COST" \
    "$PY glue_matched/train_glue.py --task $task --method $method --lr $lr --epochs $(epochs $task) --seed $s --out_dir $RES --run_id $rid" >> $JOBS
done; done; done; done
echo "$(wc -l < $JOBS | tr -d ' ') jobs -> $JOBS"
[[ "${DRY_RUN:-0}" == 1 ]] && { head -n 2 $JOBS | cut -f4; exit 0; }
if [[ "${WAIT:-0}" == 1 ]]; then
  "$PY" glue_matched/run_queue.py --jobs $JOBS --gpus $GPUS --log_dir $LOGS 2>&1 | tee -a $LOGS/_queue.log
  "$PY" glue_matched/summarize.py || true
  exit 0
fi
nohup "$PY" glue_matched/run_queue.py --jobs $JOBS --gpus $GPUS --log_dir $LOGS >> $LOGS/_queue.log 2>&1 &
echo "queue started (PID $!). Monitor: bash glue_matched/monitor.sh   Stop: pkill -f glue_matched/run_queue.py; pkill -f glue_matched/train_glue.py"
