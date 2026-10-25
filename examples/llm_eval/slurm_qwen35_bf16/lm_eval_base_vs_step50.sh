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

# Preliminary eval set (MMLU + Wikitext-2 PPL) comparing the UNMODIFIED Qwen3.5-9B against the
# BF16 step-50 checkpoint from job 295220 (seq 32768, 50 steps, plain CE on the nano35 Nemotron
# blend). Purpose is the data-compatibility question: does training a Qwen model on Nemotron
# data degrade general capability? In-domain CE fell 8.28%, which measures fitting, not damage.
#
# Follows the `--tasks mmlu,wikitext` precedent in examples/llm_qat/README.md and the earlier
# QAD-era script slurm_qwen35_qad/lm_eval_mmlu_wikitext.sh. Both checkpoints are plain BF16 --
# no modelopt state, no --quant_cfg needed.
#
# JHB forces whole-node allocation (a 1-GPU job is rejected with QOSMinGRES), so both
# checkpoints run CONCURRENTLY on separate GPUs of the one node rather than wasting two
# allocations. Set LIMIT=10 for a smoke test.
#
# NOTE: HF_HUB_OFFLINE is deliberately NOT set here (unlike the training scripts) -- lm-eval
# needs to fetch the MMLU and wikitext datasets. Compute nodes have internet, verified.
#SBATCH --job-name=coreai_dlalgo_modelopt-qwen35_bf16.lm_eval
#SBATCH --account=coreai_dlalgo_modelopt
#SBATCH --partition=batch_long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --output=/lustre/fsw/portfolios/coreai/users/rohjoshi/qwen35_bf16/logs/lm_eval_%j.log

R=/lustre/fsw/portfolios/coreai/users/rohjoshi
W=$R/qwen35_bf16
CODE_DIR=$W/code/Model-Optimizer
TAG="${TAG:-full}"

# token lives in the home dir, which is NOT mounted into the container (--no-container-mount-home);
# read it out here and let srun propagate it
set -a; . ~/.frontier_eval/.env; set +a
export HF_TOKEN

LIMIT_ARG=""
if [ -n "${LIMIT}" ]; then LIMIT_ARG="--limit ${LIMIT}"; fi

mkdir -p $W/logs $W/lm_eval_${TAG}_base $W/lm_eval_${TAG}_step50

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
            # Each concurrent process initialises its own torch.distributed store. They inherit
            # the SAME MASTER_PORT from srun, so the second one dies with EADDRINUSE (hit on the
            # smoke, job 373666). Give each a distinct port.
            CUDA_VISIBLE_DEVICES=\$gpu MASTER_PORT=\$((29500 + gpu * 13)) \
            python lm_eval_hf.py --model hf \
                --model_args pretrained=\$model \
                --tasks mmlu,wikitext \
                --batch_size 16 \
                --output_path \$out \
                ${LIMIT_ARG} > ${W}/logs/lm_eval_${TAG}_\${tagname}.out 2>&1
            echo \"EXIT \$tagname = \$?\"
        }

        run_one 0 ${W}/models/Qwen3.5-9B ${W}/lm_eval_${TAG}_base base &
        P1=\$!
        run_one 1 ${W}/bf16_ce          ${W}/lm_eval_${TAG}_step50 step50 &
        P2=\$!
        wait \$P1; R1=\$?
        wait \$P2; R2=\$?
        echo \"base rc=\$R1  step50 rc=\$R2\"
        tail -n 25 ${W}/logs/lm_eval_${TAG}_base.out   || true
        tail -n 25 ${W}/logs/lm_eval_${TAG}_step50.out || true
        [ \$R1 -eq 0 ] && [ \$R2 -eq 0 ]
    "
