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

# Smoke test: QAD (distillation, NVFP4 student) of Qwen/Qwen3.5-9B via examples/llm_qat/train.py
# on gcp-nrt. Full 9B teacher+student on one 8xB200 node, FSDP2, no CP -- this is the thing we
# actually need to validate (memory headroom without context parallelism). Real seq_length
# (32768) is kept. Uses a tiny subset of the REAL Nano 3.5 blend (chat_pretemplated path) and
# the REAL PTQ checkpoint (ptq_full), not throwaway stand-ins, so this also validates the
# actual data/tokenize path and the actual student checkpoint for the full run.
#SBATCH --job-name=coreai_dlalgo_modelopt-qwen35_qad.qad_smoke
#SBATCH --account=coreai_dlalgo_modelopt
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=00:45:00
#SBATCH --output=/lustre/fsw/portfolios/coreai/users/rohjoshi/nano35_qwen35_qad/logs/qad_smoke_%j.log

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
            --dataset_config configs/dataset/nano35_blend_smoke.yaml \
            --model_max_length 32768 \
            --attn_implementation flash_attention_2 \
            --train_samples 16 \
            --eval_samples 0 \
            --do_train True \
            --do_eval False \
            --eval_strategy no \
            --num_train_epochs 1 \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps 1 \
            --learning_rate 1e-5 \
            --logging_steps 1 \
            --save_strategy no \
            --report_to none \
            --output_dir ${WORK_DIR}/qad_smoke
    "
