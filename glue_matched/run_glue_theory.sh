#!/usr/bin/env bash
# alpha-LoRA (Theory) on GLUE (roberta-base): why does it trail LoRA on the small tasks, and can the probe design fix it?
# In run_glue_all.sh the theory variant holds 10% of the training set out as its probe, so before any alpha update it
# is LoRA trained on 90% of the data -- already 1.3-2.4 points behind LoRA on RTE / MRPC. Rows:
#   LoRA                          reference (100% of the training data)
#   alpha-LoRA-theory             as in run_glue_all.sh (probe = 10% of train, held out for the whole run)
#   alpha-LoRA-theory_no-update   control: same 90% data and probe passes, alpha never moves (trust radius 0)
#                                 -> theory vs no-update = effect of alpha alone; no-update vs LoRA = cost of the probe
#   alpha-LoRA-theory_dev-probe   fix A: the probe is the dev split already held out for checkpoint selection
#   alpha-LoRA-theory_reinsert    fix B: probe held out until the last alpha update (60%), then it rejoins training
# Same protocol as run_glue_all.sh (dev split for LR / checkpoint selection, GLUE validation set = test).
# Default: RTE + MRPC x 2 LRs x 5 seeds; LoRA / theory seeds 1-3 are copied from glue_matched/results_all when present.
# ~76 short runs, 3 at a time: about 1 h on one GPU. Runs in the foreground (start it inside tmux); re-running resumes.
#
#   bash glue_matched/run_glue_theory.sh
#   TASKS="rte mrpc sst2 qnli" bash glue_matched/run_glue_theory.sh     # + the large tasks (several more hours)
#   RES_DIR=glue_matched/results_theory bash glue_matched/monitor.sh    # progress + live tables
set -euo pipefail
cd "$(dirname "$0")/.."
VENV="${VENV:-/scratches/hypatia/ae606/venvs/biruni-3.12.3}"
[[ -n "${VIRTUAL_ENV:-}" ]] || source "$VENV/bin/activate"
PY="${PY:-python}"; GPUS="${GPUS:-0}"; SEEDS="${SEEDS:-1 2 3 4 5}"
TASKS="${TASKS:-rte mrpc}"; LRS="${LRS:-2e-4 5e-4}"
COST="${COST:-0.34}"                # roberta-base is small: 3 runs share the GPU
RES=glue_matched/results_theory; LOGS=glue_matched/logs_theory; JOBS=$LOGS/jobs.tsv
mkdir -p $RES $LOGS; : > $JOBS
export HF_HOME="${HF_HOME:-/scratches/hypatia/ae606/data/huggingface}" TOKENIZERS_PARALLELISM=false
epochs() { case "$1" in rte|mrpc) echo 30 ;; sst2|qnli) echo 5 ;; *) echo 3 ;; esac; }
THEORY="--alpha theory --theory_at 0.3,0.6 --theory_groups global"     # same as run_glue_all.sh

# job LABEL [train_glue.py args...]
job() {
  local label="$1"; shift
  for lr in $LRS; do
    local rid="${task}_${label}_lr${lr}_s${s}"
    # identical configuration already trained by run_glue_all.sh -> reuse it
    [[ -f "glue_matched/results_all/$rid.json" && ! -f "$RES/$rid.json" ]] && cp "glue_matched/results_all/$rid.json" "$RES/"
    printf '%s\t%s\t%s\t%s\n' "$rid" "$RES" "$COST" \
      "$PY glue_matched/train_glue.py --task $task --lr $lr --epochs $(epochs $task) --seed $s --label $label --out_dir $RES --run_id $rid $*" >> $JOBS
  done
}

for s in $SEEDS; do for task in $TASKS; do
  job LoRA                         --method lora
  job alpha-LoRA-theory            --method lora $THEORY
  job alpha-LoRA-theory_no-update  --method lora $THEORY --theory_trust 0
  job alpha-LoRA-theory_dev-probe  --method lora $THEORY --probe_source dev
  job alpha-LoRA-theory_reinsert   --method lora $THEORY --probe_reinsert
done; done
echo "$(wc -l < $JOBS | tr -d ' ') jobs -> $JOBS ($(ls $RES/*.json 2>/dev/null | grep -vc progress) already done)"
[[ "${DRY_RUN:-0}" == 1 ]] && { head -n 3 $JOBS | cut -f4; exit 0; }
if pgrep -f "run_queue.py --jobs $JOBS" > /dev/null; then echo "queue already running"; exit 0; fi
"$PY" glue_matched/run_queue.py --jobs $JOBS --gpus $GPUS --log_dir $LOGS 2>&1 | tee -a $LOGS/_queue.log
"$PY" glue_matched/summarize.py $RES
