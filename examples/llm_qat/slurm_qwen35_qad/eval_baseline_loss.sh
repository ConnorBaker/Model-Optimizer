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

# Preliminary-eval-set baseline: eval-only pass (eval_loss + eval_ce_loss) on the UNQUANTIZED
# Qwen/Qwen3.5-9B, teacher=itself. eval_loss (KD vs teacher) is trivially ~0 by construction
# here since student==teacher; eval_ce_loss (plain CE against ground-truth labels) is the
# metric that's actually comparable across baseline/PTQ/QAD. Same dataset/eval_samples/
# model_max_length as qad_iter0_eval.sh (job 564180, the PTQ baseline) so directly comparable.
#SBATCH --job-name=coreai_numerics_oss-qwen35_qad.eval_baseline_loss
#SBATCH --account=coreai_numerics_oss
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fsw/portfolios/coreai/users/rohjoshi/nano35_qwen35_qad/logs/eval_baseline_loss_%j.log

CODE_DIR=/lustre/fsw/portfolios/coreai/users/rohjoshi/code/Model-Optimizer
WORK_DIR=/lustre/fsw/portfolios/coreai/users/rohjoshi/nano35_qwen35_qad
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
            --model_name_or_path Qwen/Qwen3.5-9B \
            --teacher_model Qwen/Qwen3.5-9B \
            --distill true \
            --dataset_config configs/dataset/nano35_blend.yaml \
            --model_max_length 32768 \
            --attn_implementation flash_attention_2 \
            --eval_samples 50 \
            --do_train False \
            --do_eval True \
            --per_device_eval_batch_size 1 \
            --report_to none \
            --output_dir ${WORK_DIR}/eval_baseline_loss
    "
