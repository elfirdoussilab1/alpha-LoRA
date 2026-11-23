#!/usr/bin/env bash
# ABLATIONS (Qwen2.5-7B, 500 ex/task, lr 2e-4, 2 seeds)
#  group "grad":   effect of the proposed changes to alpha-LoRA-grad (each removed from v2) + the Prop.-4.3 objective
#  group "theory": alpha-LoRA-theory design (grouping of alpha, number of re-estimations)
# The reference rows (paper grad, v2, theory) are identical to run_main.sh and are reused automatically.
# Single GPU: 12 new runs, ~2.5-3 h.
source "$(dirname "$0")/common.sh"
init_exp ablation
SEEDS="${SEEDS_ABLATION:-1 2}"
prefetch $MODELS
for M in $MODELS; do
  m=$(mshort "$M")
  for s in $SEEDS; do
    b="--model $M --seed $s --lr $LR_LORA --group grad"
    add_job "${m}_grad-paper_s${s}"        $b $GRAD                            --label alpha-LoRA-grad_paper
    add_job "${m}_grad-v2_s${s}"           $b $GRADV2                          --label alpha-LoRA-grad-v2
    add_job "${m}_grad-v2-noheldout_s${s}" $b $GRADV2 --alpha_data fresh       --label v2_without_heldout
    add_job "${m}_grad-v2-nodelay_s${s}"   $b $GRADV2 --alpha_start 0          --label v2_without_delayed_start
    add_job "${m}_grad-v2-sharpe_s${s}"    $b $GRADV2 --alpha_objective sharpe --label v2_with_sharpe_objective
    b="--model $M --seed $s --lr $LR_LORA --group theory"
    add_job "${m}_theory_s${s}"            $b $THEORY                          --label alpha-LoRA-theory_layer_2x
    add_job "${m}_theory-global_s${s}"     $b $THEORY --theory_groups global   --label groups_global
    add_job "${m}_theory-module_s${s}"     $b $THEORY --theory_groups module   --label groups_module
    add_job "${m}_theory-1x_s${s}"         $b $THEORY --theory_at 0.5          --label one_update_at_0.5
  done
done
launch
