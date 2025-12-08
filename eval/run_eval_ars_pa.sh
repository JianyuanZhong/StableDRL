#!/usr/bin/env bash

set -euo pipefail

# GPUs to use (override by passing ids as args: e.g., bash run_eval_ars_pa.sh 0 1 2 3)
GPU_IDS=(0 1 2 3 4 5 6 7)
if [ $# -gt 0 ]; then
  GPU_IDS=()
  for arg in "$@"; do GPU_IDS+=("$arg"); done
fi
GPU_LIST=$(IFS=,; echo "${GPU_IDS[*]}")
NUM_GPUS=${#GPU_IDS[@]}

MASTER_PORT=${MASTER_PORT:-29451}

# Base model and selected best adapter checkpoint
MODEL_PATH=${MODEL_PATH:-"/home/ywxzml3j/ywxzml3juser46/diffusion_LM/d1/eval/LLaDA-8B-Instruct"}
ADAPTER_ROOT=${ADAPTER_ROOT:-"/home/ywxzml3j/ywxzml3juser46/diffusion_LM/d1/SFT/llada-kodcode-numina/llada-kodcode-numina-sbatch"}
BEST_CKPT=${BEST_CKPT:-"checkpoint-51200"}
CHECKPOINT_PATH="$ADAPTER_ROOT/$BEST_CKPT"

# Tasks
TASKS=("mbpp" "humaneval")
GEN_LENGTH=256
BLOCK_LENGTH=8

# ARS settings (EB-PA progression + adaptive correction)
CFG_SCALE=${CFG_SCALE:-0.0}

# Sweeps
GAMMAS=${GAMMAS:-"0.05 0.1 0.2"}
REMASK_RS=${REMASK_RS:-"1.5"}
PA_LAMBDAS=${PA_LAMBDAS:-"0.01 0.025 0.075 0.1"}
TEMPERATURES=${TEMPERATURES:-"0.3 0.75 1.0 1.5"}
STEPS_LIST=${STEPS_LIST:-"256"} # max_steps for ARS

echo "Using GPUs: $GPU_LIST (nproc_per_node=$NUM_GPUS)"
echo "Base model: $MODEL_PATH"
echo "Checkpoint: $CHECKPOINT_PATH"

mkdir -p eval_results
mkdir -p eval_results_ars_pa_v2

for task in "${TASKS[@]}"; do
  batch_size=8
  for steps in $STEPS_LIST; do
    for gamma in $GAMMAS; do
      for temp in $TEMPERATURES; do
        for pa_lambda in $PA_LAMBDAS; do
          for R in $REMASK_RS; do
            out_dir="eval_results_ars_pa_v2/"
            csv_path="${out_dir}/summary_ars_pa.csv"
            mkdir -p "$out_dir"

            echo "Running ARS-PA $task | steps=$steps | gamma=$gamma | temp=$temp | pa_lambda=$pa_lambda | R=$R"

            CUDA_VISIBLE_DEVICES=$GPU_LIST torchrun \
              --nproc_per_node $NUM_GPUS \
              --master_port $MASTER_PORT \
              evaluation_p2.py \
              --sampler ars \
              --dataset $task \
              --batch_size $batch_size \
              --gen_length $GEN_LENGTH \
              --block_length $BLOCK_LENGTH \
              --diffusion_steps $steps \
              --output_dir "$out_dir" \
              --model_path "$MODEL_PATH" \
              --checkpoint_path "$CHECKPOINT_PATH" \
              --merge_lora \
              --csv_path "$csv_path" \
              --temperature $temp \
              --cfg_scale $CFG_SCALE \
              --score_type pa \
              --pa_lambda $pa_lambda \
              --gamma $gamma \
              --remask_R $R
          done
        done
      done
    done
  done
done

echo "ARS-PA evaluations completed!"


