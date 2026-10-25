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

# Step 2b: second 2-node / 8-GPU smoke test, re-run at the REAL run's eval shape.
#
# smoke_scaleonly.sh (job 375032) passed on plumbing -- multi-node FSDP2 came up, the log
# showed "Trainable params: 200/627" (exactly the 200 tied-LSQ _amax_post Parameters, all
# 432,013,312 block scales, everything else frozen), and real loss/grad_norm appeared
# (0.05553 -> 0.05029, grad_norm 0.3262/0.5039) with a checkpoint saved.
#
# BUT its eval emitted no eval_loss key at all, only eval_runtime=0.0137s. Its blend
# (nano35_blend_jhb_smoke.yaml, blend_size 40) materializes train=30, eval=2, test=1, so
# 2 eval samples were split across 8 ranks and most ranks saw zero batches.
#
# An --eval_on_start pass that yields no eval_loss would silently destroy the iteration-0
# anchor, which is the one thing this project cannot ship without. So this run reproduces the
# REAL eval shape -- nano35_blend_jhb.yaml (train=2757, eval=153) with --eval_samples 50 over
# world_size 8, identical to qad_scaleonly.sh -- and the pass/fail criterion is simply:
# does eval_loss (and eval_ce_loss) appear in the log and trainer_state.json?
#
# Training side stays minimal (32 train samples, 2 steps); this is not a measurement.
#SBATCH --job-name=coreai_dlalgo_modelopt-qwen35_scaleonly.smoke2
#SBATCH --account=coreai_dlalgo_modelopt
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=2:00:00
#SBATCH --output=/lustre/fsw/portfolios/coreai/users/rohjoshi/qwen35_scaleonly/logs/smoke2_%j.log

R=/lustre/fsw/portfolios/coreai/users/rohjoshi
W=$R/qwen35_scaleonly
BF16W=$R/qwen35_bf16
CODE_DIR=$BF16W/code/Model-Optimizer
mkdir -p $W/logs

# Rendezvous host for the multi-node accelerate launch. Resolve to an IP so the container's
# DNS view does not matter; fall back to the hostname if resolution fails.
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
            --dataset_config configs/dataset/nano35_blend_jhb.yaml \
            --model_max_length 32768 \
            --attn_implementation flash_attention_2 \
            --train_samples 32 \
            --eval_samples 50 \
            --trainable_params '*weight_quantizer._amax_pre' '*weight_quantizer._amax_post' \
            --do_train True \
            --do_eval True \
            --eval_on_start True \
            --eval_strategy steps \
            --eval_steps 2 \
            --max_steps 2 \
            --per_device_train_batch_size 1 \
            --per_device_eval_batch_size 1 \
            --gradient_accumulation_steps 1 \
            --learning_rate 1e-4 \
            --lr_scheduler_type cosine \
            --warmup_ratio 0.05 \
            --logging_steps 1 \
            --save_strategy steps \
            --save_steps 2 \
            --save_total_limit 1 \
            --report_to none \
            --output_dir ${W}/smoke2
    "
