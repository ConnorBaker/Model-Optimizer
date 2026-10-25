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

# Preliminary eval set, part 1: MMLU + Wikitext-2 PPL for the three checkpoints of the
# Qwen3.5-9B NVFP4 scale-only (tied-LSQ) QAD experiment:
#   base    -- unquantized Qwen3.5-9B (control; also re-measures the gcp-nrt baseline on JHB
#              hardware, since the earlier 0.7870 MMLU came off 8x B200 with a different stack)
#   ptq_lsq -- the untrained LSQ student (static per-block NVFP4 + tied LSQ, MSE init)
#   qad     -- qad_scaleonly_slr1e-4, the trained scale-only checkpoint
#
# MMLU and Wikitext-2 are blend-independent, so these are directly comparable to the earlier
# gcp-nrt runs' numbers (baseline 0.7870, ptq_full 0.7620, QAD-50step 0.7668, QAD-5x 0.7674).
#
# Quantized checkpoints auto-restore their NVFP4 + LSQ state inside lm_eval_hf.py
# (is_quantized() check + mto.enable_huggingface_checkpointing()) -- no --quant_cfg needed.
#
# JHB forces whole-node allocation, so all three run CONCURRENTLY on separate GPUs of the one
# node rather than burning three allocations. Each needs a distinct MASTER_PORT: concurrent
# processes inherit the same one from srun and the later ones die with EADDRINUSE (hit on
# job 373666 in the bf16 experiment).
#
# HF_HUB_OFFLINE is deliberately NOT set -- lm-eval fetches the MMLU and wikitext datasets.
# Set LIMIT=10 for a smoke test.
#SBATCH --job-name=coreai_dlalgo_modelopt-qwen35_scaleonly.lm_eval
#SBATCH --account=coreai_dlalgo_modelopt
#SBATCH --partition=batch_long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --output=/lustre/fsw/portfolios/coreai/users/rohjoshi/qwen35_scaleonly/logs/lm_eval_%j.log

R=/lustre/fsw/portfolios/coreai/users/rohjoshi
W=$R/qwen35_scaleonly
BF16W=$R/qwen35_bf16
CODE_DIR=$BF16W/code/Model-Optimizer
TAG="${TAG:-full}"

# token lives in the home dir, which is NOT mounted into the container; read it here and let
# srun propagate it
set -a; . ~/.frontier_eval/.env; set +a
export HF_TOKEN

LIMIT_ARG=""
if [ -n "${LIMIT}" ]; then LIMIT_ARG="--limit ${LIMIT}"; fi

mkdir -p $W/logs $W/lm_eval_${TAG}_base $W/lm_eval_${TAG}_ptqlsq $W/lm_eval_${TAG}_qad

srun \
    --container-image="nvcr.io#nvidia/tensorrt-llm/release:1.2.0" \
    --container-mounts="/lustre/fsw/portfolios/coreai:/lustre/fsw/portfolios/coreai" \
    --container-workdir="${CODE_DIR}/examples/llm_eval" \
    --no-container-mount-home \
    bash -c "
        set -ex
        export HF_HOME=${W}/hf_home_eval
        export HF_TOKEN=${HF_TOKEN}

        python -m pip install --no-cache-dir -q -U 'transformers>=5.0,<5.15'
        python -m pip install --no-cache-dir -q -e '${CODE_DIR}[hf]' --no-deps
        python -m pip install --no-cache-dir -q -r requirements.txt

        run_one () {
            local gpu=\$1 model=\$2 out=\$3 tagname=\$4
            CUDA_VISIBLE_DEVICES=\$gpu MASTER_PORT=\$((29500 + gpu * 13)) \
            python lm_eval_hf.py --model hf \
                --model_args pretrained=\$model \
                --tasks mmlu,wikitext \
                --batch_size 16 \
                --output_path \$out \
                ${LIMIT_ARG} > ${W}/logs/lm_eval_${TAG}_\${tagname}.out 2>&1
            echo \"EXIT \$tagname = \$?\"
        }

        run_one 0 ${BF16W}/models/Qwen3.5-9B      ${W}/lm_eval_${TAG}_base   base   &
        P1=\$!
        run_one 1 ${W}/ptq_lsq                    ${W}/lm_eval_${TAG}_ptqlsq ptqlsq &
        P2=\$!
        run_one 2 ${W}/qad_scaleonly_slr1e-4      ${W}/lm_eval_${TAG}_qad    qad    &
        P3=\$!
        wait \$P1; R1=\$?
        wait \$P2; R2=\$?
        wait \$P3; R3=\$?
        echo \"base rc=\$R1  ptqlsq rc=\$R2  qad rc=\$R3\"
        tail -n 25 ${W}/logs/lm_eval_${TAG}_base.out   || true
        tail -n 25 ${W}/logs/lm_eval_${TAG}_ptqlsq.out || true
        tail -n 25 ${W}/logs/lm_eval_${TAG}_qad.out    || true
        [ \$R1 -eq 0 ] && [ \$R2 -eq 0 ] && [ \$R3 -eq 0 ]
    "
