export PROC_PER_NODES=8

# REPLACE WITH YOUR CHECKPOINT PATH
CKPT_PATH="./runs/your_run/ckpt-XXXXXX"

torchrun \
    --standalone \
    --nproc-per-node=$PROC_PER_NODES \
    code_metrics.py \
        --ckpt_path "$CKPT_PATH" \
        --task MBPP \
        --seed 112
