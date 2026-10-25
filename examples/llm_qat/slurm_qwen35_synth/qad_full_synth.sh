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

# QAD (distillation, NVFP4 student) of Qwen3.5-9B on the 2026-09-15 synthetic-data
# snapshot, on OCI-JHB. Student: qwen35_qad_eval/ptq_full (real PTQ NVFP4 checkpoint,
# BF16 weights + modelopt_state.pth). Teacher: unquantized Qwen3.5-9B.
#
# Hyperparameters are copied verbatim from the ORIGINAL 50-step run
# (slurm_qwen35_qad/qad_full.sh on gcp-nrt) -- user-specified, not re-derived:
#   seq 32768, lr 1e-5, cosine, warmup_ratio 0.05, eval/save every 10, 1 epoch,
#   train_samples 1600 -> 50 optimizer steps. (The 5x run differed only in
#   train_samples 8000 -> 250 steps; this is deliberately the 50-step version.)
#
# THE ONE ADAPTATION: gcp-nrt had 8 GPUs per node, JHB has 4. Two nodes x 4 GPUs
# keeps world_size 8 AND gradient_accumulation_steps 4, so the effective global
# batch is 1 x 4 x 8 = 32 exactly as before, and 1600/32 = 50 steps is preserved.
# Memory is not a concern: 8x GB300 (~2200 GB) vs the original 8x B200 (~1440 GB).
#
# DATA: the tokenized blend cache must already exist at max_length 32768
# (build_blend_32k.sh). build_blend_dataset() returns a disk cache before any
# rank-aware loading, so the train/eval split is fixed by the cache and does NOT
# depend on this job's world_size.
#SBATCH --job-name=coreai_dlalgo_modelopt-qwen35_qad_synth.qad_full
#SBATCH --account=coreai_dlalgo_modelopt
#SBATCH --partition=batch_long
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=10:00:00
#SBATCH --output=/lustre/fsw/portfolios/coreai/users/rohjoshi/qwen35_qad_synth/logs/qad_full_%j.log

R=/lustre/fsw/portfolios/coreai/users/rohjoshi
W=$R/qwen35_qad_synth
CODE_DIR=$R/qwen35_bf16/code/Model-Optimizer
STUDENT_DIR="${STUDENT_DIR:-$R/qwen35_qad_eval/ptq_full}"
TEACHER_DIR="${TEACHER_DIR:-$R/qwen35_bf16/models/Qwen3.5-9B}"
mkdir -p $W/logs

# scontrol/getent do NOT exist inside the TRT-LLM container, so the rendezvous address is
# resolved here in the sbatch body and substituted into the srun string by the outer shell.
# Matches the proven 2-node pattern in slurm_qwen35_scaleonly/qad_scaleonly.sh.
MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_ADDR=$(getent hosts "$MASTER_HOST" | awk '{print $1}' | head -n1)
[ -z "$MASTER_ADDR" ] && MASTER_ADDR=$MASTER_HOST
MASTER_PORT=29516
echo "MASTER_HOST=$MASTER_HOST MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

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

        accelerate launch --config-file configs/accelerate/fsdp2.yaml \
            --num_processes 8 \
            --num_machines 2 \
            --machine_rank \$SLURM_NODEID \
            --main_process_ip ${MASTER_ADDR} \
            --main_process_port ${MASTER_PORT} \
            --fsdp_transformer_layer_cls_to_wrap Qwen3_5DecoderLayer \
            train.py \
            --model_name_or_path ${STUDENT_DIR} \
            --teacher_model ${TEACHER_DIR} \
            --distill true \
            --dataset_config configs/dataset/qwen35_synth_blend.yaml \
            --dataset_cache_dir ${W}/dataset_cache \
            --model_max_length 32768 \
            --attn_implementation flash_attention_2 \
            --train_samples 1600 \
            --eval_samples 50 \
            --do_train True \
            --do_eval True \
            --eval_on_start True \
            --eval_strategy steps \
            --eval_steps 10 \
            --num_train_epochs 1 \
            --per_device_train_batch_size 1 \
            --per_device_eval_batch_size 1 \
            --gradient_accumulation_steps 4 \
            --learning_rate 1e-5 \
            --lr_scheduler_type cosine \
            --warmup_ratio 0.05 \
            --logging_steps 2 \
            --save_strategy steps \
            --save_steps 10 \
            --save_total_limit 3 \
            --report_to tensorboard \
            --output_dir ${W}/qad_full
    "
