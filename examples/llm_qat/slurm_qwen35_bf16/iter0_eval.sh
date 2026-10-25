#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standalone iteration-0 eval of the UNMODIFIED BF16 Qwen3.5-9B, on the exact
# dataset_config / eval_samples / model_max_length the training run uses.
#
# This is the second, independent iter-0 measurement required by
# feedback_training_metric_baseline and CLAUDE.md's "no baseline, no claim":
# bf16_train.sh separately sets --eval_on_start True, and BOTH are done, always.
# do_train=False means HF Trainer writes no trainer_state.json, so the printed
# "Evaluation results" line is captured into iter0_eval.json by hand.
#
# Side benefit: this job builds the tokenized blend cache that bf16_train.sh
# then reuses (same blend config, same max_length, same world_size).
#SBATCH --job-name=coreai_dlalgo_modelopt-qwen35_bf16.iter0_eval
#SBATCH --account=coreai_dlalgo_modelopt
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=3:00:00
#SBATCH --output=/lustre/fsw/portfolios/coreai/users/rohjoshi/qwen35_bf16/logs/iter0_eval_%j.log

R=/lustre/fsw/portfolios/coreai/users/rohjoshi
W=$R/qwen35_bf16
CODE_DIR=$W/code/Model-Optimizer
mkdir -p $W/logs

srun \
    --container-image="nvcr.io#nvidia/tensorrt-llm/release:1.2.0" \
    --container-mounts="/lustre/fsw/portfolios/coreai:/lustre/fsw/portfolios/coreai" \
    --container-workdir="${CODE_DIR}/examples/llm_qat" \
    --no-container-mount-home \
    bash -c "
        set -ex
        export HF_HOME=${W}/hf_home
        export HF_HUB_OFFLINE=1

        # Container ships transformers 4.57.3 (no qwen3_5) and no liger-kernel;
        # flash-attn 2.7.4 and accelerate 1.13.0 are already present and in range.
        python -m pip install --no-cache-dir -q -U 'transformers>=5.0,<5.15'
        python -m pip install --no-cache-dir -q -e '${CODE_DIR}[hf]' --no-deps
        python -m pip install --no-cache-dir -q -r requirements.txt

        accelerate launch --config-file configs/accelerate/fsdp2.yaml \
            --num_processes 4 \
            --fsdp_transformer_layer_cls_to_wrap Qwen3_5DecoderLayer \
            train.py \
            --model_name_or_path ${W}/models/Qwen3.5-9B \
            --dataset_config configs/dataset/nano35_blend_jhb.yaml \
            --model_max_length 32768 \
            --attn_implementation flash_attention_2 \
            --train_samples 1600 \
            --eval_samples 50 \
            --do_train False \
            --do_eval True \
            --per_device_eval_batch_size 1 \
            --report_to none \
            --output_dir ${W}/iter0_eval
    "
