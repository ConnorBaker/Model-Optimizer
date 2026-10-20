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

"""Tests for translating a transformers-targeted quant_cfg to Megatron-Core names."""

from types import SimpleNamespace

import pytest
from torch import nn

from modelopt.torch.quantization.config import normalize_quant_cfg_list
from modelopt.torch.quantization.conversion import set_quantizer_by_cfg
from modelopt.torch.quantization.hf_to_megatron import (
    ModuleMapping,
    module_mappings_from_registry,
    translate_quant_cfg_to_megatron,
)
from modelopt.torch.quantization.nn import TensorQuantizer

FP8 = {"num_bits": (4, 3), "axis": None}
DISABLE_ALL = {"quantizer_name": "*", "enable": False}
LINEAR = ("weight_quantizer", "input_quantizer", "output_quantizer")
KV = ("q_bmm_quantizer", "k_bmm_quantizer", "v_bmm_quantizer")
LAYERS = (0, 1)
EXPERTS = 3


def _mapping(megatron_param, hf_param):
    return SimpleNamespace(megatron_param=megatron_param, hf_param=hf_param)


# A stand-in shaped like a Megatron-Bridge registry of a Qwen3-MoE-like model, plus mappings to be
# skipped. The real registries are read in tests/gpu_megatron/torch/utils/plugins/test_mbridge.py.
REGISTRY = SimpleNamespace(
    mappings=[
        _mapping("output_layer.weight", "lm_head.weight"),
        _mapping(
            "decoder.layers.*.self_attention.linear_proj.weight",
            "model.layers.*.self_attn.o_proj.weight",
        ),
        _mapping(
            "decoder.layers.*.self_attention.linear_qkv.weight",
            {f"{p}": f"model.layers.*.self_attn.{p}_proj.weight" for p in "qkv"},
        ),
        _mapping(
            "decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight",
            {p: f"model.layers.*.mlp.experts.*.{p}_proj.weight" for p in ("gate", "up")},
        ),
        _mapping(
            "decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight",
            "model.layers.*.mlp.experts.*.down_proj.weight",
        ),
        _mapping(
            "decoder.layers.*.mlp.router.expert_bias",
            "model.layers.*.mlp.gate.e_score_correction_bias",
        ),
        _mapping(
            "mtp.layers.*.mtp_model_layer.mlp.linear_fc2.weight",
            "mtp.layers.*.mlp.down_proj.weight",
        ),
        _mapping("vision_tower.**", "vision_tower.**"),
    ]
)
MAPPINGS = module_mappings_from_registry(REGISTRY)

# Grouped experts of a VLM whose transformers experts are one fused parameter.
FUSED_EXPERTS_REGISTRY = SimpleNamespace(
    mappings=[
        _mapping(
            "language_model.decoder.layers.*.mlp.experts.linear_fc1.weight*",
            "model.language_model.layers.*.mlp.experts.gate_up_proj",
        ),
        _mapping(
            "language_model.decoder.layers.*.mlp.experts.linear_fc2.weight*",
            "model.language_model.layers.*.mlp.experts.down_proj",
        ),
    ]
)


def _names(template, suffixes, layers=LAYERS):
    return {
        f"{template.format(l=layer, e=expert)}.{suffix}"
        for layer in layers
        for expert in range(EXPERTS)
        for suffix in suffixes
    }


QKV = "decoder.layers.{l}.self_attention.linear_qkv"
PROJ = "decoder.layers.{l}.self_attention.linear_proj"
ATTN = "decoder.layers.{l}.self_attention.core_attention"
FC1 = "decoder.layers.{l}.mlp.experts.local_experts.{e}.linear_fc1"
FC2 = "decoder.layers.{l}.mlp.experts.local_experts.{e}.linear_fc2"
ALL_NAMES = (
    _names("output_layer", LINEAR)
    | _names(QKV, LINEAR)
    | _names(PROJ, LINEAR)
    | _names(ATTN, KV)
    | _names(FC1, LINEAR)
    | _names(FC2, LINEAR)
)


def _apply(cfg, mappings, names, layer_indices=LAYERS):
    """Apply the translated cfg to a module tree holding one quantizer per name; return enabled."""
    model = nn.Module()
    for name in names:
        *path, quantizer = name.split(".")
        parent = model
        for part in path:
            if not hasattr(parent, part):
                setattr(parent, part, nn.Module())
            parent = getattr(parent, part)
        setattr(parent, quantizer, TensorQuantizer())
    translated = translate_quant_cfg_to_megatron([DISABLE_ALL, *cfg], mappings, layer_indices)
    set_quantizer_by_cfg(model, translated)
    return {n for n, m in model.named_modules() if isinstance(m, TensorQuantizer) and m.is_enabled}


def _entry(glob, **kwargs):
    return {"quantizer_name": glob, "cfg": FP8, **kwargs}


@pytest.mark.parametrize(
    ("entries", "expected"),
    [
        pytest.param([_entry("*lm_head*")], _names("output_layer", LINEAR), id="rename"),
        pytest.param(
            [_entry("*[qkvo]_proj*weight_quantizer")],
            _names(QKV, LINEAR[:1]) | _names(PROJ, LINEAR[:1]),
            id="fused_qkv_and_glob_class",
        ),
        pytest.param(
            [_entry("*layers.1.self_attn*weight_quantizer")],
            _names(QKV, LINEAR[:1], (1,)) | _names(PROJ, LINEAR[:1], (1,)),
            id="layer_index",
        ),
        pytest.param(
            [_entry("*self_attn*[kv]_bmm_quantizer")], _names(ATTN, KV[1:]), id="kv_cache"
        ),
        pytest.param(
            [_entry("*experts*gate_proj*"), _entry("*experts*up_proj*")],
            _names(FC1, LINEAR),
            id="fused_pieces_selected_by_separate_entries",
        ),
        pytest.param(
            [_entry("*mlp.experts*weight_quantizer"), _entry("*linear_proj*weight_quantizer")],
            _names(FC1, LINEAR[:1]) | _names(FC2, LINEAR[:1]) | _names(PROJ, LINEAR[:1]),
            id="already_matching_or_megatron_spelled",
        ),
    ],
)
def test_translated_entries_select_the_megatron_quantizers(entries, expected):
    assert _apply(entries, MAPPINGS, ALL_NAMES) == expected


def test_entries_that_already_match_are_kept_unchanged():
    cfg = [
        DISABLE_ALL,
        _entry("*weight_quantizer"),
        _entry("*linear_proj*input_quantizer"),
        # Selects by class, not by name, so its transformers-spelled glob must not be translated.
        {"parent_class": "nn.Linear", "quantizer_name": "*lm_head*", "enable": False},
    ]
    assert translate_quant_cfg_to_megatron(cfg, MAPPINGS, LAYERS) == normalize_quant_cfg_list(cfg)


def test_layer_independent_entries_add_one_glob_per_module_not_per_layer():
    translated = translate_quant_cfg_to_megatron([_entry("*[qkv]_proj*")], MAPPINGS, range(48))
    assert len(translated) == 1 + len(LINEAR)


def test_translation_follows_the_pipeline_stage_layer_offset():
    entries = [
        _entry("*layers.3.self_attn*weight_quantizer"),
        _entry("*layers.3.*input_quantizer"),
    ]
    # This rank holds global layers 2 and 3 as its local layers 0 and 1.
    names = _apply(entries, MAPPINGS, ALL_NAMES, layer_indices=(2, 3))
    inputs = LINEAR[1:2]
    assert names == (
        _names(QKV, LINEAR[:1] + inputs, (1,))
        | _names(PROJ, LINEAR[:1] + inputs, (1,))
        | _names(FC1, inputs, (1,))
        | _names(FC2, inputs, (1,))
    )


def test_fused_experts_parameter_maps_onto_grouped_experts():
    mappings = module_mappings_from_registry(FUSED_EXPERTS_REGISTRY)
    fc1 = "language_model.decoder.layers.{l}.mlp.experts.linear_fc1"
    names = _names(fc1, LINEAR) | _names(fc1.replace("fc1", "fc2"), LINEAR)
    assert _apply([_entry("*gate_up_proj_weight_quantizer")], mappings, names) == _names(
        fc1, LINEAR[:1]
    )


@pytest.mark.parametrize(
    ("entries", "megatron_module"),
    [
        ([_entry("*experts*gate_proj*")], "linear_fc1"),
        ([_entry("*q_proj*"), _entry("*k_proj*")], "linear_qkv"),
        ([_entry("*"), {"quantizer_name": "*v_proj*", "enable": False}], "linear_qkv"),
        ([_entry("*experts.0.down_proj*")], "linear_fc2"),
    ],
    ids=["gate_without_up", "q_k_without_v", "v_disabled", "one_expert_of_a_module"],
)
def test_entries_a_megatron_module_cannot_express_raise(entries, megatron_module):
    with pytest.raises(ValueError, match=megatron_module):
        translate_quant_cfg_to_megatron([DISABLE_ALL, *entries], MAPPINGS, LAYERS)


def test_module_mappings_skip_what_is_not_a_decoder_module_weight():
    assert [m.megatron for m in MAPPINGS] == [
        "output_layer",
        "decoder.layers.*.self_attention.linear_proj",
        "decoder.layers.*.self_attention.linear_qkv",
        "decoder.layers.*.self_attention.core_attention",
        "decoder.layers.*.mlp.experts.local_experts.*.linear_fc1",
        "decoder.layers.*.mlp.experts.local_experts.*.linear_fc2",
    ]
    assert (
        ModuleMapping(
            "decoder.layers.*.mlp.experts.local_experts.*.linear_fc1",
            ("model.layers.*.mlp.experts.*.gate_proj", "model.layers.*.mlp.experts.*.up_proj"),
        )
        in MAPPINGS
    )
