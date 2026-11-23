#!/usr/bin/env bash
# MAIN TABLE -- commonsense reasoning (BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-e, ARC-c, OBQA), multi-task
# fine-tuning of Qwen2.5-7B on 500 examples/task (low-data regime, where the theory predicts the largest gains),
# test on 500 examples/task (one forward pass per example).
#   rows: zero-shot | LoRA | DoRA | IA3 | LoRA+ | alpha-LoRA-grad (paper) | alpha-LoRA-grad-v2 (proposed)
#         | alpha-LoRA-theory (proposed) | alpha-LoRA-grad-v2 + LoRA+ | alpha-LoRA-theory + LoRA+
# Single GPU: 28 runs, executed seed by seed (after the first ~9 runs you already have a full 1-seed table).
# Expect ~10-15 min per run (DoRA ~1.5x) -> ~5-7 h. Check the real per-run time in the monitor after the first run.
source "$(dirname "$0")/common.sh"
init_exp main
prefetch $MODELS
for M in $MODELS; do
  m=$(mshort "$M")
  add_job "${m}_zero-shot" --model $M --method none --label zero-shot --group main --seed 0
  for s in $SEEDS; do
    b="--model $M --seed $s --group main --lr $LR_LORA"
    add_job "${m}_LoRA_s${s}"          $b --method lora --label LoRA
    add_job "${m}_aLoRA-theory_s${s}"  $b $THEORY --label alpha-LoRA-theory
    add_job "${m}_aLoRA-grad-v2_s${s}" $b $GRADV2 --label alpha-LoRA-grad-v2
    add_job "${m}_aLoRA-grad_s${s}"    $b $GRAD   --label alpha-LoRA-grad
    add_job "${m}_DoRA_s${s}"          $b --method dora --label DoRA
    add_job "${m}_IA3_s${s}" --model $M --seed $s --group main --lr $LR_IA3 --method ia3 --label IA3
    b="--model $M --seed $s --group main --lr $LR_PLUS --loraplus_ratio 16"
    add_job "${m}_LoRA+_s${s}"               $b --method lora --label LoRA+
    add_job "${m}_aLoRA-theory+LoRA+_s${s}"  $b $THEORY --label alpha-LoRA-theory+LoRA+
    add_job "${m}_aLoRA-grad-v2+LoRA+_s${s}" $b $GRADV2 --label alpha-LoRA-grad-v2+LoRA+
  done
done
launch
