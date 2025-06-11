#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Note that usually LoRA needs to use larger learning rate
# Use default values if not provided
OUTPUT="${1:-./output}"
ZERO_STAGE="${2:-0}"

# Make output directory
mkdir -p "$OUTPUT"

# Start training
echo "Output directory: $OUTPUT"
echo "ZeRO Stage: $ZERO_STAGE"

deepspeed --num_gpus 1 main.py \
  --model_name_or_path facebook/opt-350m \
  --data_path Dahoas/rm-static \
   --data_split 2,4,4 \
  --gradient_accumulation_steps 8 \
  --lora_dim 128 \
  --zero_stage "$ZERO_STAGE" \
  --enable_tensorboard \
  --tensorboard_path "$OUTPUT" \
  --deepspeed \
  --output_dir "$OUTPUT" \
  &> $OUTPUT/training.log
