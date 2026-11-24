#!/usr/bin/env bash
# SMALLER MODELS -- the run_main.sh comparison on Qwen2.5-0.5B and Qwen2.5-1.5B, tested on the FULL test sets
# (22.4k examples instead of 8 x 500: ~2-3x smaller test noise). Weaker base models leave more headroom and match
# the theory's regime better (pre-trained features less aligned with the target tasks -> optimal alpha further from 1).
#   rows: zero-shot | LoRA | DoRA | IA3 | LoRA+ | alpha-LoRA-grad (paper) | alpha-LoRA-grad-v2 | alpha-LoRA-theory
#         | alpha-LoRA-grad-v2 + LoRA+ | alpha-LoRA-theory + LoRA+       (same hyper-parameters as run_main.sh)
# 2 models x 28 runs, 2 runs share the GPU. Runs in the foreground (start it inside tmux); re-running resumes.
#   bash commonsense_reasoning/run_small.sh
#   MODELS="Qwen/Qwen2.5-0.5B" bash commonsense_reasoning/run_small.sh
#   bash commonsense_reasoning/monitor.sh small
MODELS="${MODELS:-Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B}"
COST="${COST:-0.5}"
export WAIT="${WAIT:-1}"
source "$(dirname "$0")/common.sh"
init_exp small
prefetch $MODELS
FULL="--max_test 0"
for M in $MODELS; do
  m=$(mshort "$M")
  add_job "${m}_zero-shot" --model $M --method none --label zero-shot --group small --seed 0 $FULL
  for s in $SEEDS; do
    b="--model $M --seed $s --group small --lr $LR_LORA $FULL"
    add_job "${m}_LoRA_s${s}"          $b --method lora --label LoRA
    add_job "${m}_aLoRA-theory_s${s}"  $b $THEORY --label alpha-LoRA-theory
    add_job "${m}_aLoRA-grad-v2_s${s}" $b $GRADV2 --label alpha-LoRA-grad-v2
    add_job "${m}_aLoRA-grad_s${s}"    $b $GRAD   --label alpha-LoRA-grad
    add_job "${m}_DoRA_s${s}"          $b --method dora --label DoRA
    add_job "${m}_IA3_s${s}" --model $M --seed $s --group small --lr $LR_IA3 --method ia3 --label IA3 $FULL
    b="--model $M --seed $s --group small --lr $LR_PLUS --loraplus_ratio 16 $FULL"
    add_job "${m}_LoRA+_s${s}"               $b --method lora --label LoRA+
    add_job "${m}_aLoRA-theory+LoRA+_s${s}"  $b $THEORY --label alpha-LoRA-theory+LoRA+
    add_job "${m}_aLoRA-grad-v2+LoRA+_s${s}" $b $GRADV2 --label alpha-LoRA-grad-v2+LoRA+
  done
done
launch
