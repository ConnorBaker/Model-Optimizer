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

# Preliminary eval set, part 2: eval_ce_loss of the UNQUANTIZED Qwen3.5-9B on the SAME
# nano35_blend_jhb_8k.yaml eval split, at the SAME world_size 8, as the two numbers this
# experiment already has:
#   ptq_lsq (untrained LSQ student), job 375613 : eval_ce_loss 0.78515625
#   qad_scaleonly_slr1e-4 final eval, job 375614: eval_ce_loss 0.765625
# so all three sit on identical eval samples and are comparable.
#
# teacher = the same unquantized model, so eval_loss (the KD term) is trivially ~0 by
# construction; eval_ce_loss (plain CE against ground-truth labels) is the real signal here.
#
# world_size 8 matters: _merge_distributed_shards concatenates rank-major before the seed-42
# shuffle, so a 4-rank run would score a DIFFERENT 535-sample split and the number would not be
# comparable. 2 nodes x 4 GB300.
#
# The blend14000 tokenized cache already exists (built by job 375613), so this is a cache hit.
#SBATCH --job-name=coreai_dlalgo_modelopt-qwen35_scaleonly.eval_baseline_ce
#SBATCH --account=coreai_dlalgo_modelopt
#SBATCH --partition=batch_long
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=4:00:00
#SBATCH --output=/lustre/fsw/portfolios/coreai/users/rohjoshi/qwen35_scaleonly/logs/eval_baseline_ce_%j.log

R=/lustre/fsw/portfolios/coreai/users/rohjoshi
W=$R/qwen35_scaleonly
BF16W=$R/qwen35_bf16
CODE_DIR=$BF16W/code/Model-Optimizer
mkdir -p $W/logs

MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_ADDR=$(getent hosts "$MASTER_HOST" | awk '{print $1}' | head -n1)
[ -z "$MASTER_ADDR" ] && MASTER_ADDR=$MASTER_HOST
MASTER_PORT=29517
echo "MASTER_HOST=$MASTER_HOST MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

srun \
    --container-image="nvcr.io#nvidia/tensorrt-llm/release:1.2.0" \
    --container-mounts="/lustre/fsw/portfolios/coreai:/lustre/fsw/portfolios/coreai" \
    --container-workdir="${CODE_DIR}/examples/llm_qat" \
    --no-container-mount-home \
    bash -c "
        set -ex
        export HF_HOME=${W}/hf_home
        export HF_HUB_OFFLINE=1

        python -m pip install --no-cache-dir -q -U 'transformers>=5.0,<5.15'
        python -m pip install --no-cache-dir -q -e '${CODE_DIR}[hf]' --no-deps
        python -m pip install --no-cache-dir -q -r requirements.txt

        accelerate launch --config-file configs/accelerate/fsdp2.yaml \
            --num_machines 2 \
            --num_processes 8 \
            --machine_rank \$SLURM_NODEID \
            --main_process_ip ${MASTER_ADDR} \
            --main_process_port ${MASTER_PORT} \
            --fsdp_transformer_layer_cls_to_wrap Qwen3_5DecoderLayer \
            train.py \
            --model_name_or_path ${BF16W}/models/Qwen3.5-9B \
            --teacher_model ${BF16W}/models/Qwen3.5-9B \
            --distill true \
            --dataset_config configs/dataset/nano35_blend_jhb_8k.yaml \
            --model_max_length 32768 \
            --attn_implementation flash_attention_2 \
            --train_samples 8000 \
            --eval_samples 50 \
            --do_train False \
            --do_eval True \
            --per_device_eval_batch_size 1 \
            --report_to none \
            --output_dir ${W}/eval_baseline_ce
    "
