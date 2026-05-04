#!/bin/bash

# Exit on error
set -e

# =========================
# User-configurable settings
# =========================

ROOT="/monolith/global_data/astro_compression/CAMELS"
OUTPUT_DIR="./outputs_instance_norm"

TRAIN_SIZE=13000
VAL_SIZE=1000

BATCH_SIZE=16
GPUS=4

# Random seeds (10 runs per case)
SEEDS=(137)

# Optional flags
USE_WANDB="--use_wandb"

for SEED in "${SEEDS[@]}"; do

    # -------------------------------------------------------
    # 1. 8-channel InstanceNorm WITH temporal masking
    # -------------------------------------------------------
    RUN_NAME="ch8_inorm_seed${SEED}"
    SAVE_DIR="${OUTPUT_DIR}/${RUN_NAME}"

    echo "========================================"
    echo "Starting run: $RUN_NAME  (seed=$SEED)"
    echo "Saving to: $SAVE_DIR"
    echo "========================================"

    mkdir -p "$SAVE_DIR"

    uv run train.py \
        --root "$ROOT" \
        --output_dir "$OUTPUT_DIR" \
        --run_name "$RUN_NAME" \
        --train_size "$TRAIN_SIZE" \
        --val_size "$VAL_SIZE" \
        --batch_size "$BATCH_SIZE" \
        --gpus "$GPUS" \
        --latent_img_channels 8 \
        --seed "$SEED" \
        $USE_WANDB

    # -------------------------------------------------------
    # 2. 8-channel GroupNorm WITHOUT temporal masking
    # -------------------------------------------------------
    RUN_NAME="ch8_inorm_unmasked_seed${SEED}"
    SAVE_DIR="${OUTPUT_DIR}/${RUN_NAME}"

    echo "========================================"
    echo "Starting run: $RUN_NAME  (seed=$SEED)"
    echo "Saving to: $SAVE_DIR"
    echo "========================================"

    mkdir -p "$SAVE_DIR"

    uv run train.py \
        --root "$ROOT" \
        --output_dir "$OUTPUT_DIR" \
        --run_name "$RUN_NAME" \
        --train_size "$TRAIN_SIZE" \
        --val_size "$VAL_SIZE" \
        --batch_size "$BATCH_SIZE" \
        --gpus "$GPUS" \
        --latent_img_channels 8 \
        --no-temporal-masking \
        --seed "$SEED" \
        $USE_WANDB

done

echo "✅ All runs completed."
