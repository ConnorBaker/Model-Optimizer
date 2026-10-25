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

"""Export a tied-LSQ (scale-only QAD) checkpoint to a deployable NVFP4 HF checkpoint.

`export.py` cannot do this directly. The static NVFP4 export path in
`NVFP4QTensor.get_weights_scaling_factor_from_quantizer` reads the RAW `_amax` attribute:

    per_block_amax = weight_quantizer._amax.float()      # nvfp4_tensor.py:142

but `TensorQuantizer.enable_lsq()` does `delattr(self, "_amax")` and replaces it with the
learnable `_amax_post` Parameter, so that line raises AttributeError on an LSQ checkpoint.
(The `amax` PROPERTY does resolve to `_amax_post` under tied LSQ -- it is only this one raw
attribute read on the static path that breaks. The W4A8_NVFP4_FP8 branches in
`export/quant_utils.py` read `_amax` raw too, but this recipe is plain NVFP4, so they are
not on our path.)

This is the same gap the MCore exporter had, and the fix is the same one recorded in the
Nemotron scale-only runbook: fold tied-LSQ back to a static quantizer before exporting --
set `_amax = _amax_post`, drop the Parameter, clear `_lsq`. That is lossless here because
`tied_amax: true` means the pre- and post-quantization amax ARE one tensor, which is exactly
what a single static NVFP4 per-block scale represents.

An UNTIED LSQ checkpoint is refused rather than silently mangled: it learned separate pre and
post scales, and a single static NVFP4 scale cannot represent both.

Usage:
    python export_lsq.py --pyt_ckpt_path <lsq ckpt> --export_path <deploy dir>
"""

import argparse
import json
import warnings
from pathlib import Path

import torch
import transformers

import modelopt.torch.opt as mto
from modelopt.torch.export.convert_hf_config import convert_hf_quant_config_format
from modelopt.torch.export.unified_export_hf import _export_transformers_checkpoint
from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer
from modelopt.torch.utils import print_rank_0

mto.enable_huggingface_checkpointing()


def fold_lsq_to_static(model):
    """Turn every tied-LSQ quantizer back into a plain static-block quantizer.

    Returns (folded, total_scales) for verification.
    """
    folded = 0
    total_scales = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, TensorQuantizer) or not getattr(mod, "_lsq", False):
            continue
        if not getattr(mod, "_tied_amax", False):
            raise RuntimeError(
                f"{name}: untied LSQ learns separate pre/post amax, which a single static "
                "NVFP4 per-block scale cannot represent. Refusing to export."
            )
        # _amax_post is an nn.Parameter when 'post' is learnable, else a buffer.
        if "_amax_post" in mod._parameters:
            post = mod._parameters.pop("_amax_post").detach().clone().float()
        elif "_amax_post" in mod._buffers:
            post = mod._buffers.pop("_amax_post").detach().clone().float()
        else:
            raise RuntimeError(f"{name}: _lsq is set but no _amax_post found.")

        mod._lsq = False
        mod._tied_amax = False
        mod._learnable_amax = []
        # setter registers _amax as a buffer when absent and preserves fp32 on
        # StaticBlockScaleQuantizer (_amax_setter_helper override).
        mod.amax = post

        folded += 1
        total_scales += post.numel()
    return folded, total_scales


def main(args):
    print_rank_0(f"Loading LSQ checkpoint: {args.pyt_ckpt_path}")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.pyt_ckpt_path, dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.pyt_ckpt_path)

    folded, total_scales = fold_lsq_to_static(model)
    print_rank_0(f"Folded tied-LSQ -> static on {folded} quantizers, {total_scales} block scales")
    if folded == 0:
        raise RuntimeError(
            "No LSQ quantizers found -- this checkpoint is not a scale-only/LSQ checkpoint. "
            "Use export.py instead."
        )

    export_dir = Path(args.export_path)
    export_dir.mkdir(parents=True, exist_ok=True)

    try:
        post_state_dict, hf_quant_config = _export_transformers_checkpoint(model)

        with open(f"{export_dir}/hf_quant_config.json", "w") as file:
            json.dump(hf_quant_config, file, indent=4)

        hf_quant_config = convert_hf_quant_config_format(hf_quant_config)
        model.save_pretrained(export_dir, state_dict=post_state_dict)

        config_data = model.config.to_dict()
        config_data["quantization_config"] = hf_quant_config
        with open(f"{export_dir}/config.json", "w") as file:
            json.dump(config_data, file, indent=4)

        tokenizer.save_pretrained(export_dir)
    except Exception as e:
        warnings.warn("Cannot export model to the model_config.")
        raise e

    print_rank_0(f"Exported to {export_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyt_ckpt_path", required=True)
    parser.add_argument("--export_path", default="exported_model")
    args = parser.parse_args()
    main(args)
