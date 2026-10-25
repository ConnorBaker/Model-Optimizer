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

# Full QAD (distillation, NVFP4 student) of Qwen/Qwen3.5-9B via examples/llm_qat/train.py
# on gcp-nrt. Student: ptq_full/ (real PTQ NVFP4 checkpoint). Teacher: unquantized
# Qwen/Qwen3.5-9B. Data: the real Nano 3.5 blend (60/40 mix, chat_pretemplated path).
# 8xB200, FSDP2, no context parallelism -- validated feasible by qad_smoke.sh (job 562749:
# 2 steps, grad_norm 1.4-1.6, no OOM).
#
# Hyperparameters below are a reasoned first-run default (not user-specified), chosen to
# produce a real, resumable QAD run within one batch_long allocation:
#   effective global batch = per_device(1) * grad_accum(4) * 8 GPUs = 32
#   train_samples=1600 -> 50 optimizer steps/epoch; eval every 10 steps; checkpoint every 10
#   (resumable -- resubmitting with the same output_dir auto-detects the last checkpoint)
#SBATCH --job-name=coreai_numerics_oss-qwen35_qad.qad_full
#SBATCH --account=coreai_numerics_oss
#SBATCH --partition=batch_long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=10:00:00
#SBATCH --output=/lustre/fsw/portfolios/coreai/users/rohjoshi/nano35_qwen35_qad/logs/qad_full_%j.log

CODE_DIR=/lustre/fsw/portfolios/coreai/users/rohjoshi/code/Model-Optimizer
WORK_DIR=/lustre/fsw/portfolios/coreai/users/rohjoshi/nano35_qwen35_qad
STUDENT_DIR="${STUDENT_DIR:-${WORK_DIR}/ptq_full}"
HF_TOKEN_FILE=/home/rohjoshi/.cache/huggingface/token
mkdir -p "${WORK_DIR}/logs"

srun \
    --container-image="nvcr.io/nvidia/tensorrt-llm/release:1.2.0" \
    --container-mounts="/lustre/fsw/portfolios/coreai:/lustre/fsw/portfolios/coreai,${HF_TOKEN_FILE}:${HF_TOKEN_FILE}" \
    --container-workdir="${CODE_DIR}/examples/llm_qat" \
    --no-container-mount-home \
    bash -c "
        set -ex
        export HF_TOKEN=\$(cat ${HF_TOKEN_FILE} 2>/dev/null || true)

        python -m pip install --no-cache-dir -q -U 'transformers>=5.0,<5.15'
        python -m pip install --no-cache-dir -q -e '${CODE_DIR}[hf]' --no-deps
        python -m pip install --no-cache-dir -q -r requirements.txt

        accelerate launch --config-file configs/accelerate/fsdp2.yaml \
            --num_processes 8 \
            --fsdp_transformer_layer_cls_to_wrap Qwen3_5DecoderLayer \
            train.py \
            --model_name_or_path ${STUDENT_DIR} \
            --teacher_model Qwen/Qwen3.5-9B \
            --distill true \
            --dataset_config configs/dataset/nano35_blend.yaml \
            --model_max_length 32768 \
            --attn_implementation flash_attention_2 \
            --train_samples 1600 \
            --eval_samples 50 \
            --do_train True \
            --do_eval True \
            --eval_strategy steps \
            --eval_steps 10 \
            --num_train_epochs 1 \
            --per_device_train_batch_size 1 \
            --per_device_eval_batch_size 1 \
            --gradient_accumulation_steps 4 \
            --learning_rate 1e-5 \
            --lr_scheduler_type cosine \
            --warmup_ratio 0.05 \
            --logging_steps 2 \
            --save_strategy steps \
            --save_steps 10 \
            --save_total_limit 3 \
            --report_to tensorboard \
            --output_dir ${WORK_DIR}/qad_full
    "
