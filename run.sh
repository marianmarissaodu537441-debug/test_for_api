#!/usr/bin/env bash
# Run AMI + ADF-IF function/method-level coarse compression on API prompts.
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Coder-7B-Instruct}"
DATASET_PATH="${DATASET_PATH:-new_first100.json}"
DEVICE_MAP="${DEVICE_MAP:-cuda}"
RESULT_DIR="${RESULT_DIR:-results/api_coarse}"
LOG_DIR="${LOG_DIR:-logs/api_coarse}"

# Override from the environment for a one-budget run, e.g. TOKEN_BUDGETS="2048".
read -r -a TOKEN_BUDGETS <<< "${TOKEN_BUDGETS:-2048 4096}"

mkdir -p "$RESULT_DIR" "$LOG_DIR"

for token_budget in "${TOKEN_BUDGETS[@]}"; do
    output_json="$RESULT_DIR/new_first100_ami_adf_if_t${token_budget}.json"
    log_file="$LOG_DIR/new_first100_ami_adf_if_t${token_budget}.log"

    echo "Running API coarse compression: dataset=$DATASET_PATH, budget=$token_budget, model=$MODEL_NAME"
    python api_coarse_compress.py \
        --dataset "$DATASET_PATH" \
        --token-budget "$token_budget" \
        --model-name "$MODEL_NAME" \
        --device-map "$DEVICE_MAP" \
        --output-json "$output_json" 2>&1 | tee "$log_file"
done

echo "API coarse-compression experiments completed. Results: $RESULT_DIR"
