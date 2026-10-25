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

# LONGER-BUDGET QAD experiment testing the token-budget hypothesis from the first run's
# investigation (see project_qwen35_9b_qad memory): the original qad_full.sh run (50 steps,
# GBS 32, ~52M tokens) used ~0.3% of the ModelOpt qad skill's own default QAD budget (17.3B
# tokens at seq 32768). This run uses ~5x the samples/steps (250 steps, GBS 32, ~262M tokens)
# with a cosine LR schedule sized to the NEW 250-step horizon (not decayed to zero at step 50
# like the original run -- that confound is exactly what made the original "plateau" ambiguous).
#
# Same student (ptq_full/), teacher, quantization recipe, LR peak (1e-5), and architecture
# settings as qad_full.sh -- ONLY the data budget and the schedule horizon it's sized to change,
# so this is a controlled comparison against the original run.
#
# --eval_on_start True is set per the now-mandatory rule (feedback_training_metric_baseline,
# reference_qwen35_qad_pipeline_runbook step 4) -- this run's own log carries its iter-0 anchor.
# A SEPARATE eval-only pass (qad_iter0_eval_8k.sh) against the same new dataset_config/eval
# split additionally confirms it, per the "do both" instruction.
#SBATCH --job-name=coreai_numerics_oss-qwen35_qad.qad_5x_budget
#SBATCH --account=coreai_numerics_oss
#SBATCH --partition=batch_long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=16:00:00
#SBATCH --output=/lustre/fsw/portfolios/coreai/users/rohjoshi/nano35_qwen35_qad/logs/qad_5x_budget_%j.log

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
            --train_samples 8000 \
            --eval_samples 50 \
            --do_train True \
            --do_eval True \
            --eval_on_start True \
            --eval_strategy steps \
            --eval_steps 25 \
            --num_train_epochs 1 \
            --per_device_train_batch_size 1 \
            --per_device_eval_batch_size 1 \
            --gradient_accumulation_steps 4 \
            --learning_rate 1e-5 \
            --lr_scheduler_type cosine \
            --warmup_ratio 0.05 \
            --logging_steps 5 \
            --save_strategy steps \
            --save_steps 25 \
            --save_total_limit 3 \
            --report_to tensorboard \
            --output_dir ${WORK_DIR}/qad_5x_budget
    "
