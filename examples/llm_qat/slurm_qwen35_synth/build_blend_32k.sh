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

# Build + cache the tokenized blend over the 2026-09-15 Qwen3.5-9B synthetic-data
# snapshot, so the eventual QAD run starts with a warm cache and any data-side
# failure surfaces here rather than inside a GPU job.
#
# Tokenization only -- no GPU needed, so this runs on the `cpu` partition (48 nodes,
# 7 d limit) following qwen35_scaleonly/count_tokens.sh, not the GPU `batch` pattern.
#
# Env recipe is copied from the proven slurm_qwen35_bf16/iter0_eval.sh:
# the container ships transformers 4.57.3 which has NO qwen3_5, so transformers is
# upgraded at job start; modelopt is pip-installed editable from the checkout.
#
# --dataset_cache_dir is set to an absolute path outside either Model-Optimizer
# checkout so the cache is reusable from whichever one the training run uses.
#SBATCH --job-name=coreai_dlalgo_modelopt-qwen35_qad_synth.build_blend_32k
#SBATCH --account=coreai_dlalgo_modelopt
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=3:00:00
#SBATCH --output=/lustre/fsw/portfolios/coreai/users/rohjoshi/qwen35_qad_synth/logs/build_blend_32k_%j.log

R=/lustre/fsw/portfolios/coreai/users/rohjoshi
W=$R/qwen35_qad_synth
CODE_DIR=$R/qwen35_bf16/code/Model-Optimizer
mkdir -p $W/logs $W/dataset_cache

srun \
    --container-image="nvcr.io#nvidia/tensorrt-llm/release:1.2.0" \
    --container-mounts="/lustre/fsw/portfolios/coreai:/lustre/fsw/portfolios/coreai" \
    --container-workdir="${CODE_DIR}/examples/llm_qat" \
    --no-container-mount-home \
    bash -c "
        set -ex
        export HF_HOME=${R}/qwen35_bf16/hf_home
        export HF_HUB_OFFLINE=1

        python -m pip install --no-cache-dir -q -U 'transformers>=5.0,<5.15'
        python -m pip install --no-cache-dir -q -e '${CODE_DIR}[hf]' --no-deps
        python -m pip install --no-cache-dir -q -r requirements.txt

        python -c 'import transformers; print(\"transformers\", transformers.__version__)'

        python dataset_utils.py \
            --dataset_config configs/dataset/qwen35_synth_blend.yaml \
            --model_name_or_path ${R}/qwen35_bf16/models/Qwen3.5-9B \
            --model_max_length 32768 \
            --num_proc 16 \
            --dataset_cache_dir ${W}/dataset_cache
    "
