#!/bin/bash

#!/bin/bash

# Configuration variables
GPU_IDS=(0 1 2 3 4 5 6 7)

MASTER_PORT=29411

# Base model and adapter (LoRA) directory
# You can override these by exporting env vars before calling the script.
MODEL_PATH=${MODEL_PATH:-"/home/ywxzml3j/ywxzml3juser46/diffusion_LM/d1/eval/LLaDA-8B-Instruct"}
# Default adapter root to the provided SFT run directory
ADAPTER_ROOT=${ADAPTER_ROOT:-"/home/ywxzml3j/ywxzml3juser46/diffusion_LM/d1/SFT/llada-kodcode-numina/llada-kodcode-numina-sbatch"}

# Tasks and generation lengths to evaluate
# TASKS=("gsm8k" "math")
TASKS=("mbpp")
GEN_LENGTHS=(256)

# CLI overrides
# Usage examples:
#   bash run_eval.sh 0 1 2 3
#   MODEL_PATH=/path/to/base ADAPTER_ROOT=/path/to/adapters bash run_eval.sh 0 1 2 3
if [ $# -gt 0 ]; then
  GPU_IDS=()
  for arg in "$@"; do GPU_IDS+=("$arg"); done
fi

GPU_LIST=$(IFS=,; echo "${GPU_IDS[*]}")
NUM_GPUS=${#GPU_IDS[@]}
echo "Using GPUs: $GPU_LIST (nproc_per_node=$NUM_GPUS)"

echo "Base model: $MODEL_PATH"

mkdir -p eval_results_original

# 1) Baseline evaluations (no LoRA)
for task in "${TASKS[@]}"; do
  for gen_length in "${GEN_LENGTHS[@]}"; do
    # Batch size heuristic based on sequence length
    if [ "$gen_length" -ge 512 ]; then
      batch_size=8
    else
      batch_size=16
    fi

    echo "Running baseline $task | gen_length=$gen_length | batch_size=$batch_size"

    CUDA_VISIBLE_DEVICES=$GPU_LIST torchrun \
      --nproc_per_node $NUM_GPUS \
      --master_port $MASTER_PORT \
      eval.py \
      --dataset $task \
      --batch_size $batch_size \
      --gen_length $gen_length \
      --output_dir "eval_results_original" \
      --model_path "$MODEL_PATH" \
      --csv_path "eval_results/summary.csv"
  done
done

# 2) LoRA adapter evaluations (if adapters present)
# echo "Adapter root: $ADAPTER_ROOT"
# mapfile -t CHECKPOINTS < <(find "$ADAPTER_ROOT" -maxdepth 1 -type d -name "checkpoint-*" | sort -V)
# if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
#   echo "No checkpoints found under $ADAPTER_ROOT. Skipping LoRA evaluations."
# else
#   echo "Found ${#CHECKPOINTS[@]} checkpoints"

#   for task in "${TASKS[@]}"; do
#     for gen_length in "${GEN_LENGTHS[@]}"; do
#       # Batch size heuristic based on sequence length
#       if [ "$gen_length" -ge 512 ]; then
#         batch_size=4
#       else
#         batch_size=8
#       fi

#       for ckpt in "${CHECKPOINTS[@]}"; do
#         if [ ! -f "$ckpt/adapter_model.safetensors" ]; then
#           echo "Skipping $ckpt (no adapter_model.safetensors)"
#           continue
#         fi

#         echo "Running $task | gen_length=$gen_length | batch_size=$batch_size | ckpt=$ckpt"

#         CUDA_VISIBLE_DEVICES=$GPU_LIST torchrun \
#           --nproc_per_node $NUM_GPUS \
#           --master_port $MASTER_PORT \
#           eval.py \
#           --dataset $task \
#           --batch_size $batch_size \
#           --gen_length $gen_length \
#           --output_dir "eval_results" \
#           --model_path "$MODEL_PATH" \
#           --checkpoint_path "$ckpt" \
#           --merge_lora \
#           --csv_path "eval_results/summary.csv"
#       done
#     done
#   done
# fi


echo "All evaluations completed!"
