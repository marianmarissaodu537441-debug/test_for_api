#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

MODEL_NAME="Qwen/Qwen2.5-Coder-7B-Instruct"
MODEL_PATH_NAME="qwencoder-7b-instruct"
BASE_RESULT_DIR="results/${MODEL_PATH_NAME}"
BASE_LOG_DIR="logs/${MODEL_PATH_NAME}"

mkdir -p ${BASE_LOG_DIR}
mkdir -p ${BASE_RESULT_DIR}

echo "Starting experiments for ${MODEL_NAME} on GPU ${CUDA_VISIBLE_DEVICES}"

# --- CodeCompressor Method Configuration ---
TARGET_TOKENS=(2048 4096)
FINE_RATIOS=(0.5 0.8)
BETAS=(0.0 0.5)

echo "--- Running CodeCompressor with various configurations ---"
for tokens in "${TARGET_TOKENS[@]}"; do
    for ratio in "${FINE_RATIOS[@]}"; do
        for beta in "${BETAS[@]}"; do
            echo "Running CodeCompressor: target_tokens=${tokens}, fine_ratio=${ratio}, beta=${beta}"
            python main.py \
                --model_name ${MODEL_NAME} \
                --compression_model_name ${MODEL_NAME} \
                --method code_compressor \
                --filter_background_tokens_min 5000 \
                --result_dir "${BASE_RESULT_DIR}" \
                --num_examples 500 \
                --code_compressor_target_token ${tokens} \
                --code_compressor_fine_ratio ${ratio} \
                --importance_beta ${beta} > "${BASE_LOG_DIR}/code_compressor_t${tokens}_fr${ratio}_b${beta}.log" 2>&1
            echo "Finished CodeCompressor: target_tokens=${tokens}, fine_ratio=${ratio}, beta=${beta}"
        done
    done
done

echo "--- Finished CodeCompressor ---"
