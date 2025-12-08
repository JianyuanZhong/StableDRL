export PROC_PER_NODES=8
# REPLACE WITH YOUR CHECKPOINT PATH
CKPT_PATH="./runs/your_run/ckpt-XXXXXX" 

torchrun \
    --standalone \
    --nproc-per-node=$PROC_PER_NODES \
    math_metrics.py \
        --ckpt_path "$CKPT_PATH" \
        --local_data_path openai/gsm8k \
        --num_workers 4 \
        --seed 112
