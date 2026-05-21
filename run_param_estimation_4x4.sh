#!/bin/bash
set -e

# Define directories
ROOT_DIR="/monolith/global_data/astro_compression/CAMELS/"
MODELS_DIR="/home/sid/cosmo_compression/checkpoints/4x4_models"
OUTPUT_DIR="./results_param_est_4x4"

cd /home/sid/cosmo_compression/examples/review_experiments

# Run the parameter estimation script
# The python script itself is now programmed to iterate through all .ckpt files
# inside MODELS_DIR and automatically exclude those containing "last" in their filenames.
/home/sid/cosmo_compression/.venv/bin/python param_estimation.py \
    --models-dir "$MODELS_DIR" \
    --root "$ROOT_DIR" \
    --output-dir "$OUTPUT_DIR" 2>&1 | tee param_estimation.log

echo "Parameter estimation completed successfully!"
