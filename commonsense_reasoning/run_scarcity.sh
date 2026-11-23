#!/usr/bin/env bash
# DATA-SCARCITY SWEEP -- theory insight (iii): the benefit of rescaling grows when target data is scarce.
# n per task in {100, 250, 500} (x8 tasks); rows LoRA | DoRA | alpha-LoRA-grad-v2 | alpha-LoRA-theory; 2 seeds.
# The n=500 runs are identical to run_main.sh and are reused automatically (not retrained).
# Single GPU: 16 new runs (small n -> short runs), ~2-3 h.  More points: NS="100 250 500 1000" SEEDS="1 2 3"
source "$(dirname "$0")/common.sh"
init_exp scarcity
SEEDS="${SEEDS_SCARCITY:-1 2}"
prefetch $MODELS
ep_of() { case "$1" in 100) echo 5 ;; 250) echo 4 ;; *) echo 2 ;; esac; }
for M in $MODELS; do
  m=$(mshort "$M")
  for s in $SEEDS; do
    for n in ${NS:-100 250 500}; do
      b="--model $M --seed $s --lr $LR_LORA --group scarcity --n_per_task $n --epochs $(ep_of $n)"
      add_job "${m}_n${n}_LoRA_s${s}"          $b --method lora --label LoRA
      add_job "${m}_n${n}_aLoRA-theory_s${s}"  $b $THEORY --label alpha-LoRA-theory
      add_job "${m}_n${n}_aLoRA-grad-v2_s${s}" $b $GRADV2 --label alpha-LoRA-grad-v2
      add_job "${m}_n${n}_DoRA_s${s}"          $b --method dora --label DoRA
    done
  done
done
launch
