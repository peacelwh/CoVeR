#!/usr/bin/env bash
# Batch script for the experiments of the paper. Run from the repository root:
#
#   bash scripts/run_paper.sh main        # LLM-Fusion and CoVeR, selected recipe, 3 seeds (comparison with the state of the art)
#   bash scripts/run_paper.sh analysis    # default recipe, 5 seeds (test-time override, corrections, ablation)
#   bash scripts/run_paper.sh permanent   # permanent missingness, 3 seeds
#   bash scripts/run_paper.sh tables      # collect results/runs into results/tables
#
# Environment: GPU=<id> (default 0), DATASETS="<datasets>" (default: the four MMSA corpora; mosi sims simsv2 for the
# analyses), MISSING=permanent|permanent_incomplete (stage permanent), LLM=<path or hub id of Qwen3-0.6B>.
set -euo pipefail
STAGE=${1:-main}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
if [ "$STAGE" = analysis ]; then DATASETS=${DATASETS:-"mosi sims simsv2"}; else DATASETS=${DATASETS:-"mosi mosei sims simsv2"}; fi
MISSING=${MISSING:-permanent}
LLM_ARG=${LLM:+--llm_path $LLM}

case "$STAGE" in
  main)
    for ds in $DATASETS; do for seed in 42 43 44; do
      python scripts/train.py --dataset "$ds" --method llm_fusion --seed "$seed" $LLM_ARG
      python scripts/train.py --dataset "$ds" --method cover --seed "$seed" $LLM_ARG
    done; done ;;
  analysis)
    for ds in $DATASETS; do for seed in 42 43 44 45 46; do
      python scripts/train.py --dataset "$ds" --method llm_fusion --recipe default --seed "$seed" $LLM_ARG
      python scripts/train.py --dataset "$ds" --method cover --recipe default --seed "$seed" $LLM_ARG
      python scripts/train.py --dataset "$ds" --method cover --recipe default --no_gate --seed "$seed" $LLM_ARG
      python scripts/train.py --dataset "$ds" --method learned_gate --seed "$seed" $LLM_ARG
      python scripts/train.py --dataset "$ds" --method llm_fusion --recipe default --conflict_weight 5 \
        --name llm_fusion_default_reweight --seed "$seed" $LLM_ARG
      run=results/runs/$ds/llm_fusion_default_s$seed
      python scripts/evaluate.py --run "$run" --tau_test 0.3 --out "$run/results_override.json" $LLM_ARG
      python scripts/evaluate.py --run "$run" --steering $LLM_ARG
    done; done ;;
  permanent)
    for ds in $DATASETS; do
      judges=""
      if [ "$ds" != sims ] && [ "$ds" != simsv2 ]; then   # expert judgments from the visible samples only
        python scripts/build_mrbench_a.py --dataset "$ds" --missing "$MISSING" --out_dir "data/mrbench_a_permanent/$MISSING"
        judges="--judge_labels_dir data/mrbench_a_permanent/$MISSING"
      fi
      for seed in 42 43 44; do
        python scripts/train.py --dataset "$ds" --method llm_fusion --missing "$MISSING" --name "llm_fusion_$MISSING" \
          --no_rates --seed "$seed" $LLM_ARG
        python scripts/train.py --dataset "$ds" --method cover --missing "$MISSING" --name "cover_$MISSING" \
          --no_rates --seed "$seed" $judges $LLM_ARG
      done
    done ;;
  tables)
    python scripts/make_tables.py --runs results/runs --out results/tables ;;
  *)
    echo "usage: bash scripts/run_paper.sh {main|analysis|permanent|tables}"; exit 1 ;;
esac
