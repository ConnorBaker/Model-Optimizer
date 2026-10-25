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

# Qwen3.5-9B BF16 training run (NO quantization) on OCI-JHB.
#
# Objective is plain cross-entropy on assistant tokens: --distill is left off, so
# train.py builds a QATTrainer, and with no --recipe/--quant_cfg that is an
# ordinary HF Trainer (transformers_trainer.py gates all quantization on
# `quant_cfg is not None`). No teacher: student and teacher would be the same
# BF16 weights, making any KD divergence identically zero with no gradient.
#
# Settings chosen to match the earlier NVFP4 QAD runs on this model wherever the
# BF16/CE change allows: same nano35 60/40 blend at blend_size=4000, seq 32768,
# GBS 32, lr 1e-5 cosine + 5% warmup, 50 optimizer steps.
#
# Geometry differs from those runs: 4 GB300 here vs 8 B200 on gcp-nrt. Since
# _merge_distributed_shards concatenates rank-major, the pre-shuffle ordering --
# and therefore the train/eval split -- is world-size dependent even though the
# sample SET is identical. This run's eval loss is anchored against its own
# iter-0 only, and is NOT directly comparable to the stored PTQ/QAD eval numbers.
#
# --eval_on_start True is mandatory here (feedback_training_metric_baseline,
# reference_qwen35_qad_pipeline_runbook step 4). iter0_eval.sh independently
# measures the same anchor; both are run, always.
#SBATCH --job-name=coreai_dlalgo_modelopt-qwen35_bf16.train
#SBATCH --account=coreai_dlalgo_modelopt
#SBATCH --partition=batch_long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=8:00:00
#SBATCH --output=/lustre/fsw/portfolios/coreai/users/rohjoshi/qwen35_bf16/logs/bf16_train_%j.log

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
            --do_train True \
            --do_eval True \
            --eval_on_start True \
            --eval_strategy steps \
            --eval_steps 10 \
            --num_train_epochs 1 \
            --per_device_train_batch_size 1 \
            --per_device_eval_batch_size 1 \
            --gradient_accumulation_steps 8 \
            --learning_rate 1e-5 \
            --lr_scheduler_type cosine \
            --warmup_ratio 0.05 \
            --logging_steps 5 \
            --save_strategy steps \
            --save_steps 25 \
            --save_total_limit 2 \
            --report_to tensorboard \
            --output_dir ${W}/bf16_ce
    "
