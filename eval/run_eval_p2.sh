#!/bin/bash

# GPUs to use (override by passing ids as args: e.g., bash run_eval_p2.sh 0 1 2 3)
GPU_IDS=(0 1 2 3 4 5 6 7)
if [ $# -gt 0 ]; then
  GPU_IDS=()
  for arg in "$@"; do GPU_IDS+=("$arg"); done
fi
GPU_LIST=$(IFS=,; echo "${GPU_IDS[*]}")
NUM_GPUS=${#GPU_IDS[@]}

MASTER_PORT=${MASTER_PORT:-29431}

# Base model and selected best adapter checkpoint
MODEL_PATH=${MODEL_PATH:-"/home/ywxzml3j/ywxzml3juser46/diffusion_LM/d1/eval/LLaDA-8B-Instruct"}
ADAPTER_ROOT=${ADAPTER_ROOT:-"/home/ywxzml3j/ywxzml3juser46/diffusion_LM/d1/SFT/llada-kodcode-numina/llada-kodcode-numina-sbatch"}
BEST_CKPT=${BEST_CKPT:-"checkpoint-51200"}
CHECKPOINT_PATH="$ADAPTER_ROOT/$BEST_CKPT"

# Tasks
TASKS=("gsm8k" "math")
GEN_LENGTH=256
BLOCK_LENGTH=8

# P2 settings
PLANNER_MODE=${PLANNER_MODE:-"self"}
KAPPA_SCHEDULE=${KAPPA_SCHEDULE:-"linear"}
CFG_SCALE=${CFG_SCALE:-0.0}

# Sweeps
ETAS=${ETAS:-"0.25 0.5 1.0 1.2 1.6"}
TEMPERATURES=${TEMPERATURES:-"0.0 0.1 0.75 1.0"}
STEPS_LIST=${STEPS_LIST:-"64 128 256"}

echo "Using GPUs: $GPU_LIST (nproc_per_node=$NUM_GPUS)"
echo "Base model: $MODEL_PATH"
echo "Checkpoint: $CHECKPOINT_PATH"

mkdir -p eval_results

mkdir -p eval_results_p2

for task in "${TASKS[@]}"; do
  batch_size=8
  for steps in $STEPS_LIST; do
    for eta in $ETAS; do
      for temp in $TEMPERATURES; do
        out_dir="eval_results_p2/"
        csv_path="${out_dir}/summary_p2.csv"
        mkdir -p "$out_dir"

        echo "Running P2 $task | steps=$steps | eta=$eta | temp=$temp | planner=$PLANNER_MODE | kappa=$KAPPA_SCHEDULE"

        CUDA_VISIBLE_DEVICES=$GPU_LIST torchrun \
          --nproc_per_node $NUM_GPUS \
          --master_port $MASTER_PORT \
          evaluation_p2.py \
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
          --planner_mode "$PLANNER_MODE" \
          --eta $eta \
          --kappa_schedule "$KAPPA_SCHEDULE" \
          --temperature $temp \
          --cfg_scale $CFG_SCALE
      done
    done
  done
done

echo "P2 evaluations completed!"


