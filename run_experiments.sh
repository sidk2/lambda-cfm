#!/bin/bash

# Ensure the script exits if any command fails
set -e

# Configuration
# ==============================================================================
# IMPORTANT: Update ROOT_DIR to point to your actual CAMELS dataset directory!
# ==============================================================================
ROOT_DIR="/monolith/global_data/astro_compression/CAMELS/"
OUTPUT_DIR="./checkpoints/8x8_models_upsampler/"

SEEDS=(861 670 2104 90 1572) # Add or change random seeds as needed
TRAIN_SIZE=13000
VAL_SIZE=1000
GPUS=6

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

for SEED in "${SEEDS[@]}"; do
    echo "====================================================================="
    echo "🚀 Starting experimental runs for RANDOM SEED: $SEED"
    echo "====================================================================="

    # 1. Train WITH temporal masking (default behavior)
    RUN_NAME="cosmoflow_seed_${SEED}_with_masking"
    echo "➡️ Training WITH temporal masking ($RUN_NAME)..."
    
    uv run src/cosmo_compression/train.py \
        --root "$ROOT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --run_name "$RUN_NAME" \
        --train_size $TRAIN_SIZE \
        --val_size $VAL_SIZE \
        --gpus $GPUS \
        --use_wandb \
        --random-seed $SEED

    echo "✅ Finished $RUN_NAME"
    echo "---------------------------------------------------------------------"

    # 2. Train WITHOUT temporal masking (passing the new flag)
    RUN_NAME="cosmoflow_seed_${SEED}_without_masking"
    echo "➡️ Training WITHOUT temporal masking ($RUN_NAME)..."
    
    uv run src/cosmo_compression/train.py \
        --root "$ROOT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --run_name "$RUN_NAME" \
        --train_size $TRAIN_SIZE \
        --val_size $VAL_SIZE \
        --gpus $GPUS \
        --use_wandb \
        --random-seed $SEED \
        --disable_temporal_masking

    echo "✅ Finished $RUN_NAME"
    echo "====================================================================="
done

echo "🎉 All experiments completed successfully!"
