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

# Step 3: standalone iteration-0 eval of the UNTRAINED LSQ student (ptq_lsq/), on the exact
# dataset_config / eval_samples / model_max_length / world_size the training run uses.
#
# This is the second, independent iter-0 measurement required by CLAUDE.md's "no baseline,
# no claim" and feedback_training_metric_baseline: qad_scaleonly.sh separately sets
# --eval_on_start True, and BOTH are run, always. do_train=False means HF Trainer writes no
# trainer_state.json, so the printed "Evaluation results" line is captured by hand.
#
# Side benefit: this job builds the tokenized blend cache that qad_scaleonly.sh then reuses
# (same blend config, same max_length, same world_size 8).
#
# world_size 8 is deliberate: _merge_distributed_shards concatenates rank-major before the
# seed-42 shuffle, so only 8 ranks reproduce the gcp-nrt qad_5x_budget train/eval split and
# make this run's eval numbers comparable to that run's.
#SBATCH --job-name=coreai_dlalgo_modelopt-qwen35_scaleonly.iter0_eval
#SBATCH --account=coreai_dlalgo_modelopt
#SBATCH --partition=batch_long
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=10:00:00
#SBATCH --output=/lustre/fsw/portfolios/coreai/users/rohjoshi/qwen35_scaleonly/logs/iter0_eval_%j.log

R=/lustre/fsw/portfolios/coreai/users/rohjoshi
W=$R/qwen35_scaleonly
BF16W=$R/qwen35_bf16
CODE_DIR=$BF16W/code/Model-Optimizer
mkdir -p $W/logs

MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_ADDR=$(getent hosts "$MASTER_HOST" | awk '{print $1}' | head -n1)
[ -z "$MASTER_ADDR" ] && MASTER_ADDR=$MASTER_HOST
MASTER_PORT=29514
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
            --model_name_or_path ${W}/ptq_lsq \
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
            --output_dir ${W}/iter0_eval
    "
