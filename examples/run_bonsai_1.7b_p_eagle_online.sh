#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$ROOT_DIR/cache/compiled_kernels}

NUM_GPUS=${1:-1}
TP_SIZE=${2:-1}
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-8}

# Public unpacked Bonsai is the correct HF/Transformers training target.
BONSAI_TARGET_MODEL_PATH=${BONSAI_TARGET_MODEL_PATH:-prism-ml/Bonsai-1.7B-unpacked}
BONSAI_EAGLE3_CKPT=${BONSAI_EAGLE3_CKPT:-/home/local/Projects/models/registry/local/Bonsai-1.7B-EAGLE3-local/weights}
BONSAI_P_EAGLE_OUTPUT_DIR=${BONSAI_P_EAGLE_OUTPUT_DIR:-/home/local/Projects/THOTH/artifacts/models/local/Bonsai-1.7B-P-EAGLE-local-smoke}
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-$ROOT_DIR/cache/dataset/sharegpt_train.sample-1024.jsonl}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-sdpa}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path $BONSAI_TARGET_MODEL_PATH \
    --ckpt-dir $BONSAI_EAGLE3_CKPT \
    --speculative-algorithm P_EAGLE \
    --parallel-drafting \
    --train-mask-hidden-only \
    --k-train 5 \
    --cod-retention 0.8 \
    --ttt-length 5 \
    --train-data-path $TRAIN_DATA_PATH \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $BONSAI_P_EAGLE_OUTPUT_DIR \
    --num-epochs 1 \
    --batch-size 1 \
    --draft-accumulation-steps 8 \
    --learning-rate 1e-4 \
    --warmup-ratio 0.0025 \
    --max-length 256 \
    --attention-backend $ATTENTION_BACKEND \
    --chat-template qwen \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.embed_tokens.weight \
    --tp-size $TP_SIZE \
    --target-model-backend hf \
    --log-interval 10 \
    --max-num-steps 500
