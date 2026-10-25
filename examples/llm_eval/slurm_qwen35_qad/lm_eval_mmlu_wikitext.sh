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

# Preliminary-eval-set MMLU + Wikitext-2 perplexity for one checkpoint, via lm-evaluation-harness
# (examples/llm_eval/lm_eval_hf.py). Reused for baseline / PTQ / QAD by setting MODEL_PATH and
# JOB_TAG env vars at submission time. Already-quantized checkpoints (ptq_full/, qad_full/)
# auto-restore their NVFP4 state (mto.enable_huggingface_checkpointing() + is_quantized() check
# in lm_eval_hf.py) -- no --quant_cfg needed for those. 0-shot MMLU, following the exact
# `--tasks mmlu,wikitext` precedent already documented in examples/llm_qat/README.md.
#
# Required env vars: MODEL_PATH (HF hub id or local checkpoint dir), JOB_TAG (short label for
# job name / output dir, e.g. baseline|ptq|qad). Optional: LIMIT (set for a smoke test).
#SBATCH --job-name=coreai_numerics_oss-qwen35_qad.lm_eval
#SBATCH --account=coreai_numerics_oss
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fsw/portfolios/coreai/users/rohjoshi/nano35_qwen35_qad/logs/lm_eval_%x_%j.log

: "${MODEL_PATH:?set MODEL_PATH}"
: "${JOB_TAG:?set JOB_TAG}"

CODE_DIR=/lustre/fsw/portfolios/coreai/users/rohjoshi/code/Model-Optimizer
WORK_DIR=/lustre/fsw/portfolios/coreai/users/rohjoshi/nano35_qwen35_qad
HF_TOKEN_FILE=/home/rohjoshi/.cache/huggingface/token
LIMIT_ARG=""
if [ -n "${LIMIT}" ]; then
    LIMIT_ARG="--limit ${LIMIT}"
fi
mkdir -p "${WORK_DIR}/logs" "${WORK_DIR}/lm_eval_${JOB_TAG}"

srun \
    --container-image="nvcr.io/nvidia/tensorrt-llm/release:1.2.0" \
    --container-mounts="/lustre/fsw/portfolios/coreai:/lustre/fsw/portfolios/coreai,${HF_TOKEN_FILE}:${HF_TOKEN_FILE}" \
    --container-workdir="${CODE_DIR}/examples/llm_eval" \
    --no-container-mount-home \
    bash -c "
        set -ex
        export HF_TOKEN=\$(cat ${HF_TOKEN_FILE} 2>/dev/null || true)

        python -m pip install --no-cache-dir -q -U 'transformers>=5.0,<5.15'
        python -m pip install --no-cache-dir -q -e '${CODE_DIR}[hf]' --no-deps
        python -m pip install --no-cache-dir -q -r requirements.txt

        python lm_eval_hf.py --model hf \
            --model_args pretrained=${MODEL_PATH} \
            --tasks mmlu,wikitext \
            --batch_size 4 \
            --output_path ${WORK_DIR}/lm_eval_${JOB_TAG} \
            ${LIMIT_ARG}
    "
