# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Promote an existing (dynamic-block NVFP4) PTQ checkpoint's weight quantizers to static-block
LSQ (learnable per-block scales), without rerunning PTQ's calibration.

The source checkpoint's weight quantizers use "dynamic" NVFP4 block scaling: a single scalar
per-tensor amax is calibrated, and per-block scales are derived fresh at every forward pass
from real block data (see reference_qwen35_9b_qad memory / project investigation). That leaves
no persistent per-block amax to make learnable. This script runs a DATA-FREE weight-only
recalibration (MSE, operates purely on weight values -- no calibration data, no forward passes,
unlike PTQ's activation calibration) to derive genuine per-block amax values, promotes each
eligible weight quantizer to StaticBlockScaleQuantizer, and enables LSQ (learnable post-amax)
on it via ModelOpt's model-level `lsq()` calibration function. Activation/KV quantizers are
untouched (dynamic, no persistent amax to learn -- see the same investigation).

Usage:
    python enable_lsq_on_ptq.py \\
        --model_name_or_path <ptq_full checkpoint dir> \\
        --output_dir <output dir>
"""

import argparse

import torch
import transformers

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.model_calib import lsq
from modelopt.torch.utils import print_rank_0

mto.enable_huggingface_checkpointing()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--learnable_amax",
        nargs="+",
        default=["post"],
        help="Which LSQ amax params are learnable: 'pre', 'post', or both.",
    )
    parser.add_argument(
        "--tied_amax",
        action="store_true",
        help="If set, pre and post amax share a single learnable tensor.",
    )
    parser.add_argument(
        "--no_quantize_pre_scale",
        action="store_true",
        help="If set, skip FP8-quantizing the LSQ pre scale.",
    )
    return parser.parse_args()


def main():
    args = get_args()

    print_rank_0(f"Loading PTQ checkpoint: {args.model_name_or_path}")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)

    print_rank_0("Quant summary BEFORE LSQ promotion:")
    mtq.print_quant_summary(model)

    print_rank_0(
        "Running data-free MSE weight recalibration + static-block LSQ promotion "
        f"(learnable_amax={args.learnable_amax}, tied_amax={args.tied_amax})..."
    )
    lsq(
        model,
        forward_loop=None,
        scale_algorithm={"method": "mse"},
        learnable_amax=args.learnable_amax,
        tied_amax=args.tied_amax,
        quantize_pre_scale=not args.no_quantize_pre_scale,
    )

    print_rank_0("Quant summary AFTER LSQ promotion:")
    mtq.print_quant_summary(model)

    print_rank_0(f"Saving LSQ-enabled checkpoint to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print_rank_0("Done.")


if __name__ == "__main__":
    main()
