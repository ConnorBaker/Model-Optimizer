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

# Smoke test: PTQ (NVFP4) of Qwen/Qwen3.5-9B via examples/llm_qat/quantize.py on gcp-nrt.
# Tiny calib_size -- validates the pipeline (VLM->text load, quant, save), not recipe quality.
#SBATCH --job-name=coreai_dlalgo_modelopt-qwen35_qad.ptq_smoke
#SBATCH --account=coreai_dlalgo_modelopt
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:40:00
#SBATCH --output=/lustre/fsw/portfolios/coreai/users/rohjoshi/nano35_qwen35_qad/logs/ptq_smoke_%j.log

CODE_DIR=/lustre/fsw/portfolios/coreai/users/rohjoshi/code/Model-Optimizer
WORK_DIR=/lustre/fsw/portfolios/coreai/users/rohjoshi/nano35_qwen35_qad
HF_TOKEN_FILE=/home/rohjoshi/.cache/huggingface/token
mkdir -p "${WORK_DIR}/logs"

srun \
    --container-image="nvcr.io/nvidia/tensorrt-llm/release:1.2.0" \
    --container-mounts="${CODE_DIR}:${CODE_DIR},${WORK_DIR}:${WORK_DIR},${HF_TOKEN_FILE}:${HF_TOKEN_FILE}" \
    --container-workdir="${CODE_DIR}/examples/llm_qat" \
    --no-container-mount-home \
    bash -c "
        set -ex
        unset SLURM_PROCID SLURM_LOCALID SLURM_NTASKS WORLD_SIZE LOCAL_RANK RANK
        export HF_TOKEN=\$(cat ${HF_TOKEN_FILE} 2>/dev/null || true)

        python -m pip install --no-cache-dir -q -U 'transformers>=5.0,<5.15'
        python -m pip install --no-cache-dir -q -e '${CODE_DIR}[hf]' --no-deps
        python -m pip install --no-cache-dir -q -r requirements.txt

        python quantize.py \
            --model_name_or_path Qwen/Qwen3.5-9B \
            --dataset_config configs/dataset/blend_smoke.yaml \
            --recipe general/ptq/nvfp4_default-kv_fp8 \
            --calib_size 8 \
            --calib_batch_size 1 \
            --output_dir ${WORK_DIR}/ptq_smoke
    "
