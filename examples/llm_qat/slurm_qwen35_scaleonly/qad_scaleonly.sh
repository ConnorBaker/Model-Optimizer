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

# Step 4: Qwen3.5-9B NVFP4 SCALE-ONLY (LSQ) QAD -- 250 steps, scale LR 1e-4, on OCI-JHB.
#
# Weights are FROZEN; the only trainable parameters are the 432,013,312 tied per-block NVFP4
# weight-scale amax values (one _amax_post Parameter per each of the 200 static weight
# quantizers in ptq_lsq/). Freezing on the HF path is done with --trainable_params, i.e.
# requires_grad=False on everything else -- safe here because there is no MCore fused-linear
# path (see reference_lsq_hf_vs_mcore_scale_lr).
#
# --learning_rate IS the scale LR on this path (1e-4, the HF-tuned default in
# configs/train/qad_scale_only.yaml). Do NOT carry over the MCore optimum of 3e-5: MCore's KD
# loss is SUM-reduced and HF's is MEAN-reduced, so the two are not comparable.
#
# Everything else is held identical to the gcp-nrt qad_5x_budget run so the comparison is
# controlled: 250 steps / 1 epoch, 8000 train + 50 eval samples, GBS 32 (1 x accum 4 x 8 GPUs),
# seq 32768 (262.1M tokens), cosine sized to the 250-step horizon with 5% warmup, eval/save
# every 25 steps. The student differs only where it must: static per-block NVFP4 weight scales
# with tied LSQ instead of dynamic-block scales (a dynamic-block quantizer has no persistent
# per-block tensor to learn -- see ptq_lsq.sh).
#
# world_size 8 (2 nodes x 4 GB300) reproduces the gcp-nrt run's train/eval split, which is
# rank-major-concat dependent -- see iter0_eval_scaleonly.sh.
#
# --eval_on_start True is mandatory (feedback_training_metric_baseline); iter0_eval_scaleonly.sh
# independently measures the same anchor, and BOTH are run, always.
#
# Single job, no afterany continuation: cross-job resume is broken in this flow on modelopt
# 0.44.0 ("Model already has modelopt state!", reference_qad_resume_bug), so the walltime is
# sized to cover the whole run.
#SBATCH --job-name=coreai_dlalgo_modelopt-qwen35_scaleonly.qad_slr1e-4
#SBATCH --account=coreai_dlalgo_modelopt
#SBATCH --partition=batch_long
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=2-00:00:00
#SBATCH --output=/lustre/fsw/portfolios/coreai/users/rohjoshi/qwen35_scaleonly/logs/qad_slr1e-4_%j.log

R=/lustre/fsw/portfolios/coreai/users/rohjoshi
W=$R/qwen35_scaleonly
BF16W=$R/qwen35_bf16
CODE_DIR=$BF16W/code/Model-Optimizer
mkdir -p $W/logs

MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_ADDR=$(getent hosts "$MASTER_HOST" | awk '{print $1}' | head -n1)
[ -z "$MASTER_ADDR" ] && MASTER_ADDR=$MASTER_HOST
MASTER_PORT=29515
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
            --trainable_params '*weight_quantizer._amax_pre' '*weight_quantizer._amax_post' \
            --do_train True \
            --do_eval True \
            --eval_on_start True \
            --eval_strategy steps \
            --eval_steps 25 \
            --num_train_epochs 1 \
            --per_device_train_batch_size 1 \
            --per_device_eval_batch_size 1 \
            --gradient_accumulation_steps 4 \
            --learning_rate 1e-4 \
            --weight_decay 0.0 \
            --lr_scheduler_type cosine \
            --warmup_ratio 0.05 \
            --logging_steps 5 \
            --save_strategy steps \
            --save_steps 25 \
            --save_total_limit 3 \
            --report_to tensorboard \
            --output_dir ${W}/qad_scaleonly_slr1e-4
    "
