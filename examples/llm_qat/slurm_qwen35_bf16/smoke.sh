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

# Smoke test for the Qwen3.5-9B BF16 training run on OCI-JHB: tiny blend, 2 steps.
# Validates the aarch64 container + runtime installs + FSDP2 across 4 GB300 +
# the transferred data subset, before committing to the real run.
#
# Checks to make on the log (per reference_qwen35_qad_pipeline_runbook step 3):
#   - real {'loss': ..., 'grad_norm': ...} lines, not just exit 0
#   - no ValueError about gradient_checkpointing (fsdp2.yaml already sets it)
# Delete .dataset_cache/tokenized/<smoke hash> afterwards so a poisoned cache
# cannot mask a later fix.
#SBATCH --job-name=coreai_dlalgo_modelopt-qwen35_bf16.smoke
#SBATCH --account=coreai_dlalgo_modelopt
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=2:00:00
#SBATCH --output=/lustre/fsw/portfolios/coreai/users/rohjoshi/qwen35_bf16/logs/smoke_%j.log

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
            --dataset_config configs/dataset/nano35_blend_jhb_smoke.yaml \
            --shuffle_buffer 64 \
            --model_max_length 32768 \
            --attn_implementation flash_attention_2 \
            --train_samples 8 \
            --eval_samples 4 \
            --do_train True \
            --do_eval True \
            --eval_on_start True \
            --eval_strategy steps \
            --eval_steps 1 \
            --max_steps 2 \
            --per_device_train_batch_size 1 \
            --per_device_eval_batch_size 1 \
            --gradient_accumulation_steps 1 \
            --learning_rate 1e-5 \
            --lr_scheduler_type cosine \
            --logging_steps 1 \
            --save_strategy no \
            --report_to none \
            --output_dir ${W}/smoke
    "
