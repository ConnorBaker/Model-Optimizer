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

# Step 1 of the Qwen3.5-9B NVFP4 SCALE-ONLY (LSQ) QAD experiment, on OCI-JHB.
#
# Builds the LSQ student: weights -> STATIC per-block NVFP4 with a tied learnable amax,
# activations -> dynamic NVFP4, KV cache -> FP8. Recipe general/qad/nvfp4_lsq-mse_init-fp8_kv
# (tied_amax: true, learnable_amax: [pre, post] -> a single _amax_post Parameter per weight
# quantizer; MSE init with an FP8 scale sweep).
#
# WHY A NEW PTQ IS NEEDED rather than reusing the gcp-nrt ptq_full/ student: that checkpoint
# was built with general/ptq/nvfp4_default-kv_fp8, whose weight quantizers are DYNAMIC-block
# NVFP4 (block_sizes {-1: 16, type: dynamic}) carrying only a scalar _amax -- per-block scales
# are recomputed from block data every forward, so there is no persistent per-block tensor to
# learn. model_calib.lsq() skips every quantizer that is not a StaticBlockScaleQuantizer
# (model_calib.py:2335), which is exactly why the 2026-09-02 enable_lsq_on_ptq.py run (job
# 566269) produced a ptq_full_lsq/ with zero LSQ parameters: its AFTER summary still shows
# 'type': 'dynamic' on every weight quantizer and model.safetensors is byte-identical in size
# to ptq_full/.
#
# Calibration is kept IDENTICAL to the original ptq_full.sh so the student is as close as the
# recipe change allows: same blend_ptq_calib.yaml, --calib_size 512, --calib_batch_size 1, and
# the default model_max_length (8192) -- ptq_full.sh did not pass --model_max_length either.
# blend_ptq_calib.yaml streams 7 nvidia/Nemotron-* datasets from the HF hub, so this job needs
# outbound network and HF_TOKEN (compute nodes here reach PyPI during the pip steps below).
#SBATCH --job-name=coreai_dlalgo_modelopt-qwen35_scaleonly.ptq_lsq
#SBATCH --account=coreai_dlalgo_modelopt
#SBATCH --partition=batch_long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=8:00:00
#SBATCH --output=/lustre/fsw/portfolios/coreai/users/rohjoshi/qwen35_scaleonly/logs/ptq_lsq_%j.log

R=/lustre/fsw/portfolios/coreai/users/rohjoshi
W=$R/qwen35_scaleonly
BF16W=$R/qwen35_bf16
CODE_DIR=$BF16W/code/Model-Optimizer
mkdir -p $W/logs

# HF_TOKEN lives in the (un-mounted) home dir; read it here, srun propagates it into the container.
export HF_TOKEN=$(sed -n 's/^HF_TOKEN=//p' ~/.frontier_eval/.env)

srun \
    --container-image="nvcr.io#nvidia/tensorrt-llm/release:1.2.0" \
    --container-mounts="/lustre/fsw/portfolios/coreai:/lustre/fsw/portfolios/coreai" \
    --container-workdir="${CODE_DIR}/examples/llm_qat" \
    --no-container-mount-home \
    bash -c "
        set -ex
        # quantize.py is single-process; strip the Slurm vars that make torch think otherwise.
        unset SLURM_PROCID SLURM_LOCALID SLURM_NTASKS WORLD_SIZE LOCAL_RANK RANK
        export HF_HOME=${W}/hf_home

        python -m pip install --no-cache-dir -q -U 'transformers>=5.0,<5.15'
        python -m pip install --no-cache-dir -q -e '${CODE_DIR}[hf]' --no-deps
        python -m pip install --no-cache-dir -q -r requirements.txt

        python quantize.py \
            --model_name_or_path ${BF16W}/models/Qwen3.5-9B \
            --dataset_config configs/dataset/blend_ptq_calib.yaml \
            --recipe general/qad/nvfp4_lsq-mse_init-fp8_kv \
            --calib_size 512 \
            --calib_batch_size 1 \
            --output_dir ${W}/ptq_lsq
    "
