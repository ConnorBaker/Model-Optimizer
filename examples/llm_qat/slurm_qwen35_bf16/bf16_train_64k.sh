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

# Qwen3.5-9B BF16 training run (NO quantization) at seq 65536 / 400 steps, on OCI-JHB.
#
# Follow-up to the 50-step / seq-32768 run (job 295220). Same objective -- plain cross-entropy
# on assistant tokens, no teacher, no quantization -- and the same nano35 Nemotron blend; only
# the sequence length and step count change.
#
# Purpose is the data-compatibility question, not quantization: does training a Qwen model on
# Nemotron data degrade it? The previous run's 52.4M tokens was 0.30% of the ModelOpt qad
# reference budget, too small a dose to answer that. This run is 400 x 32 x 65536 = 838.9M
# tokens, ~16x more and ~4.9% of that reference budget.
#
# blend_size raised to 21000 so 12800 train samples are available for 400 steps at GBS 32 in a
# SINGLE epoch -- repeating a smaller pool would confound "data damages the model" with "the
# model saw the same samples five times".
#
# --eval_on_start True is mandatory (feedback_training_metric_baseline). iter0_eval_64k.sh
# independently measures the same anchor; both are run, always.
#SBATCH --job-name=coreai_dlalgo_modelopt-qwen35_bf16.train64k
#SBATCH --account=coreai_dlalgo_modelopt
#SBATCH --partition=batch_long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=6-00:00:00
#SBATCH --output=/lustre/fsw/portfolios/coreai/users/rohjoshi/qwen35_bf16/logs/bf16_train64k_%j.log

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

        python -m pip install --no-cache-dir -q -U 'transformers>=5.0,<5.15'
        python -m pip install --no-cache-dir -q -e '${CODE_DIR}[hf]' --no-deps
        python -m pip install --no-cache-dir -q -r requirements.txt

        accelerate launch --config-file configs/accelerate/fsdp2.yaml \
            --num_processes 4 \
            --fsdp_transformer_layer_cls_to_wrap Qwen3_5DecoderLayer \
            train.py \
            --model_name_or_path ${W}/models/Qwen3.5-9B \
            --dataset_config configs/dataset/nano35_blend_jhb_64k.yaml \
            --model_max_length 65536 \
            --attn_implementation flash_attention_2 \
            --train_samples 12800 \
            --eval_samples 50 \
            --do_train True \
            --do_eval True \
            --eval_on_start True \
            --eval_strategy steps \
            --eval_steps 25 \
            --num_train_epochs 1 \
            --per_device_train_batch_size 1 \
            --per_device_eval_batch_size 1 \
            --gradient_accumulation_steps 8 \
            --learning_rate 1e-5 \
            --lr_scheduler_type cosine \
            --warmup_ratio 0.05 \
            --logging_steps 5 \
            --save_strategy steps \
            --save_steps 50 \
            --save_total_limit 2 \
            --report_to tensorboard \
            --output_dir ${W}/bf16_ce_64k
    "
