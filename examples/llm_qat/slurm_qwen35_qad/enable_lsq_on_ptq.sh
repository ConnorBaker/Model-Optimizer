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

# Promote ptq_full/'s weight quantizers (dynamic-block NVFP4, scalar amax) to static-block LSQ
# (learnable per-block scales) via a DATA-FREE MSE weight recalibration -- no PTQ rerun, no
# calibration data, no forward passes (weight-only calibration operates on weight values alone).
# See enable_lsq_on_ptq.py docstring and project_qwen35_9b_qad memory for why this promotion
# step is needed (the PTQ checkpoint's weight quantizers have no persistent per-block amax to
# make learnable as-is).
#SBATCH --job-name=coreai_numerics_oss-qwen35_qad.enable_lsq
#SBATCH --account=coreai_numerics_oss
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/fsw/portfolios/coreai/users/rohjoshi/nano35_qwen35_qad/logs/enable_lsq_%j.log

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
        unset SLURM_PROCID SLURM_LOCALID SLURM_NTASKS WORLD_SIZE LOCAL_RANK RANK
        export HF_TOKEN=\$(cat ${HF_TOKEN_FILE} 2>/dev/null || true)

        python -m pip install --no-cache-dir -q -U 'transformers>=5.0,<5.15'
        python -m pip install --no-cache-dir -q -e '${CODE_DIR}[hf]' --no-deps
        python -m pip install --no-cache-dir -q -r requirements.txt

        python enable_lsq_on_ptq.py \
            --model_name_or_path ${WORK_DIR}/ptq_full \
            --output_dir ${WORK_DIR}/ptq_full_lsq
    "
