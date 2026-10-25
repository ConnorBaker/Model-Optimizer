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

# Iteration-0 baseline for the qad_5x_budget.sh run: eval-only pass on the PRE-QAD PTQ checkpoint
# (ptq_full/, unmodified), using the SAME dataset_config/eval_samples/model_max_length as
# qad_5x_budget.sh. A fresh anchor is required here (not a reuse of the original run's
# iter0_eval.json) because nano35_blend_8k.yaml's larger blend_size changes which 50 samples land
# in the eval split. Independent of qad_5x_budget.sh's own --eval_on_start True, per the mandatory
# "do both" rule (feedback_training_metric_baseline).
#SBATCH --job-name=coreai_numerics_oss-qwen35_qad.iter0_eval_8k
#SBATCH --account=coreai_numerics_oss
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fsw/portfolios/coreai/users/rohjoshi/nano35_qwen35_qad/logs/qad_iter0_eval_8k_%j.log

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
            --dataset_config configs/dataset/nano35_blend_8k.yaml \
            --model_max_length 32768 \
            --attn_implementation flash_attention_2 \
            --eval_samples 50 \
            --do_train False \
            --do_eval True \
            --per_device_eval_batch_size 1 \
            --report_to none \
            --output_dir ${WORK_DIR}/qad_iter0_eval_8k
    "
