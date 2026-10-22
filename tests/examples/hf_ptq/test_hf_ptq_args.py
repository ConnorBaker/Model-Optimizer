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

import importlib
import sys
from pathlib import Path

import pytest
import torch

from modelopt.recipe import load_recipe
from modelopt.recipe.config import AutoQuantizeConfig, AutoQuantizeConstraints
from modelopt.recipe.presets import QUANT_CFG_CHOICES
from modelopt.torch.quantization.config import QuantizeConfig

_EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples" / "hf_ptq"


def _import_hf_ptq(monkeypatch):
    monkeypatch.syspath_prepend(str(_EXAMPLES_DIR))
    return importlib.import_module("hf_ptq")


def _parse_hf_ptq_args(monkeypatch, *args):
    hf_ptq = _import_hf_ptq(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["hf_ptq.py", *args])
    parsed_args = hf_ptq.parse_args()
    parsed_args.dataset = (
        parsed_args.dataset.split(",")
        if isinstance(parsed_args.dataset, str)
        else parsed_args.dataset
    )
    parsed_args.calib_size = [int(num_sample) for num_sample in parsed_args.calib_size.split(",")]
    return hf_ptq, parsed_args


def test_autoquant_recipe_builds_mtq_inputs(monkeypatch):
    """The recipe path maps an AutoQuantizeConfig to the expected mtq.auto_quantize inputs."""
    hf_ptq, args = _parse_hf_ptq_args(
        monkeypatch, "--pyt_ckpt_path", "dummy", "--kv_cache_qformat", "none"
    )
    aq = load_recipe("general/auto_quantize/nvfp4_fp8_at_5p4bits").auto_quantize
    inputs = hf_ptq._mtq_inputs_from_auto_quantize_config(aq, args)

    assert inputs["constraints"] == {"effective_bits": 5.4, "cost_model": "weight"}
    assert inputs["kv_cache_quant_cfg"] is None
    assert inputs["method"] == "gradient"
    assert inputs["score_boundary"] is None
    assert inputs["score_size"] == 128
    assert inputs["kl_log_prob_fallback"] is None
    assert inputs["num_candidates"] == 1
    assert inputs["candidate_pool_size"] is None
    assert inputs["candidate_min_hamming_distance"] == 1
    assert inputs["candidate_diversity_group_patterns"] is None
    assert inputs["candidate_max_objective_regret"] is None
    assert inputs["candidate_pool_selection"] == "objective"
    assert inputs["candidate_distance_fallback"] == "none"
    assert inputs["candidate_selection_size"] == 0
    assert inputs["candidate_selection_metric"] == "kl_div"
    assert inputs["fixed_quantization_config"] is None
    assert inputs["module_search_spaces"] == []
    # disabled_layers come straight from the recipe (no model introspection).
    assert inputs["disabled_layers"] == aq.disabled_layers
    assert "*output_layer*" in inputs["disabled_layers"]
    # Candidates resolve to the exact preset dicts mtq expects (preset identity preserved).
    assert inputs["quantization_formats"][0] == QUANT_CFG_CHOICES["nvfp4"]
    assert inputs["quantization_formats"][1] == QUANT_CFG_CHOICES["fp8"]


def test_autoquant_recipe_cost_excluded_layers_map_into_cost(monkeypatch):
    """Top-level cost_excluded_layers maps to the mtq constraints.cost.excluded_module_name_patterns
    key (distinct from disabled_layers), so a cost-exclusion recipe matches the nested mtq dict."""
    hf_ptq, args = _parse_hf_ptq_args(
        monkeypatch, "--pyt_ckpt_path", "dummy", "--kv_cache_qformat", "none"
    )
    aq = load_recipe(
        "huggingface/qwen3_6_moe/auto_quantize/w4a16_nvfp4_fp8_at_6p0bits-active_moe"
    ).auto_quantize
    inputs = hf_ptq._mtq_inputs_from_auto_quantize_config(aq, args)

    # cost-exclusion is hoisted to a sibling of disabled_layers but still reaches the mtq cost dict.
    assert aq.cost_excluded_layers == ["*visual*", "*mtp*", "*vision_tower*"]
    assert inputs["constraints"]["cost"] == {
        "active_moe_expert_ratio": 0.03125,
        "excluded_module_name_patterns": ["*visual*", "*mtp*", "*vision_tower*"],
    }
    # The two exclusions are independent: cost-excluded patterns are also disabled here, but the
    # roles (cost-accounting vs search) are tracked separately.
    assert "*visual*" in inputs["disabled_layers"]


def test_autoquant_recipe_maps_module_search_spaces(monkeypatch):
    """Fixed PTQ baseline and explicit recipe candidates map to mtq inputs."""
    hf_ptq, args = _parse_hf_ptq_args(
        monkeypatch, "--pyt_ckpt_path", "dummy", "--kv_cache_qformat", "none"
    )
    recipe = load_recipe(
        "huggingface/qwen3_6_moe/auto_quantize/w4a16_nvfp4_fp8_module_spaces_at_6p0bits-active_moe"
    )
    inputs = hf_ptq._mtq_inputs_from_auto_quantize_config(
        recipe.auto_quantize, args, fixed_quantize_config=recipe.quantize
    )
    model_ptq = load_recipe("huggingface/qwen3_5_moe/ptq/w4a16_nvfp4-fp8_attn-kv_fp8_cast")

    assert inputs["quantization_formats"] == []
    assert inputs["fixed_quantization_config"] == model_ptq.quantize.model_dump()
    (searched,) = inputs["module_search_spaces"]
    assert searched["module_name_patterns"] == [
        "*mlp.shared_expert*",
        "*linear_attn*",
        "*self_attn*",
        "*lm_head*",
    ]
    assert searched["quantization_formats"] == [
        QUANT_CFG_CHOICES["w4a16_nvfp4"],
        QUANT_CFG_CHOICES["fp8"],
    ]
    assert searched["allow_no_quant"] is False


def test_autoquant_rejects_non_export_safe_candidate(monkeypatch):
    """A candidate that resolves to a preset outside the export-safe set is rejected before search."""
    hf_ptq, args = _parse_hf_ptq_args(
        monkeypatch, "--pyt_ckpt_path", "dummy", "--kv_cache_qformat", "none"
    )
    non_safe = next(k for k in QUANT_CFG_CHOICES if k not in hf_ptq._AUTO_QUANTIZE_QFORMATS)
    aq = AutoQuantizeConfig(
        constraints=AutoQuantizeConstraints(effective_bits=4.8),
        candidate_formats=[
            QuantizeConfig(**QUANT_CFG_CHOICES["fp8"]),
            QuantizeConfig(**QUANT_CFG_CHOICES[non_safe]),
        ],
    )
    with pytest.raises(ValueError, match="not supported for unified checkpoint export"):
        hf_ptq._mtq_inputs_from_auto_quantize_config(aq, args)


def test_autoquant_warns_on_custom_candidate(monkeypatch):
    """A candidate matching no shipped preset can't be export-verified, so it warns (not blocks)."""
    hf_ptq, args = _parse_hf_ptq_args(
        monkeypatch, "--pyt_ckpt_path", "dummy", "--kv_cache_qformat", "none"
    )
    custom = QuantizeConfig(quant_cfg=[{"quantizer_name": "*", "enable": False}])
    aq = AutoQuantizeConfig(
        constraints=AutoQuantizeConstraints(effective_bits=4.8),
        candidate_formats=[QuantizeConfig(**QUANT_CFG_CHOICES["fp8"]), custom],
    )
    with pytest.warns(UserWarning, match="export compatibility cannot be verified"):
        hf_ptq._mtq_inputs_from_auto_quantize_config(aq, args)


def test_autoquant_two_stage_hidden_recon_maps_to_mtq_inputs(monkeypatch):
    hf_ptq, args = _parse_hf_ptq_args(
        monkeypatch, "--pyt_ckpt_path", "dummy", "--kv_cache_qformat", "none"
    )
    aq = AutoQuantizeConfig(
        constraints=AutoQuantizeConstraints(effective_bits=5.5),
        candidate_formats=[QuantizeConfig(**QUANT_CFG_CHOICES["fp8"])],
        auto_quantize_method="hidden_recon",
        num_candidates=20,
        candidate_selection_size=512,
    )

    inputs = hf_ptq._mtq_inputs_from_auto_quantize_config(aq, args)

    assert inputs["method"] == "hidden_recon"
    assert inputs["num_candidates"] == 20
    assert inputs["candidate_selection_size"] == 512


def test_autoquant_candidate_selection_split_is_disjoint(monkeypatch):
    hf_ptq = _import_hf_ptq(monkeypatch)
    source = torch.utils.data.DataLoader(list(range(10)), batch_size=2, shuffle=False)

    first_stage, held_out = hf_ptq._split_dataloader_for_candidate_selection(source, 6, 4)

    assert torch.cat(list(first_stage)).tolist() == list(range(6))
    assert torch.cat(list(held_out)).tolist() == list(range(6, 10))
    assert set(first_stage.dataset.indices).isdisjoint(held_out.dataset.indices)


def test_skip_export_flag_is_opt_in(monkeypatch):
    _, default_args = _parse_hf_ptq_args(monkeypatch, "--pyt_ckpt_path", "dummy")
    _, skip_args = _parse_hf_ptq_args(
        monkeypatch,
        "--pyt_ckpt_path",
        "dummy",
        "--skip_export",
    )

    assert default_args.skip_export is False
    assert skip_args.skip_export is True


@pytest.mark.parametrize(
    ("num_candidates", "selection_size", "error"),
    [
        (0, 0, "num_candidates must be positive"),
        (2, 0, "candidate_selection_size must be positive"),
        (1, 1, "requires num_candidates greater than 1"),
    ],
)
def test_autoquant_two_stage_config_validation(num_candidates, selection_size, error):
    with pytest.raises(ValueError, match=error):
        AutoQuantizeConfig(
            constraints=AutoQuantizeConstraints(effective_bits=5.5),
            candidate_formats=[QuantizeConfig(**QUANT_CFG_CHOICES["fp8"])],
            num_candidates=num_candidates,
            candidate_selection_size=selection_size,
        )


def test_autoquant_diverse_candidate_pool_maps_to_mtq_inputs(monkeypatch):
    hf_ptq, args = _parse_hf_ptq_args(
        monkeypatch, "--pyt_ckpt_path", "dummy", "--kv_cache_qformat", "none"
    )
    aq = AutoQuantizeConfig(
        constraints=AutoQuantizeConstraints(effective_bits=5.5),
        candidate_formats=[QuantizeConfig(**QUANT_CFG_CHOICES["fp8"])],
        num_candidates=10,
        candidate_pool_size=40,
        candidate_min_hamming_distance=8,
        candidate_diversity_group_patterns=["*self_attn*", "*linear_attn*"],
        candidate_max_objective_regret=0.05,
        candidate_pool_selection="farthest",
        candidate_distance_fallback="relax",
        candidate_selection_size=512,
        candidate_selection_metric="ear",
    )

    inputs = hf_ptq._mtq_inputs_from_auto_quantize_config(aq, args)

    assert inputs["candidate_pool_size"] == 40
    assert inputs["candidate_min_hamming_distance"] == 8
    assert inputs["candidate_diversity_group_patterns"] == [
        "*self_attn*",
        "*linear_attn*",
    ]
    assert inputs["candidate_max_objective_regret"] == pytest.approx(0.05)
    assert inputs["candidate_pool_selection"] == "farthest"
    assert inputs["candidate_distance_fallback"] == "relax"
    assert inputs["candidate_selection_metric"] == "ear"


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"candidate_pool_size": 9}, "candidate_pool_size must be at least"),
        ({"candidate_min_hamming_distance": 0}, "candidate_min_hamming_distance must be positive"),
        ({"candidate_max_objective_regret": -0.01}, "candidate_max_objective_regret cannot"),
        ({"candidate_diversity_group_patterns": []}, "cannot be empty"),
    ],
)
def test_autoquant_diverse_candidate_pool_validation(overrides, error):
    with pytest.raises(ValueError, match=error):
        AutoQuantizeConfig(
            constraints=AutoQuantizeConstraints(effective_bits=5.5),
            candidate_formats=[QuantizeConfig(**QUANT_CFG_CHOICES["fp8"])],
            num_candidates=10,
            candidate_selection_size=512,
            **overrides,
        )


def test_autoquant_export_guard_not_bypassed_by_effective_bits(monkeypatch):
    """A non-export-safe preset can't dodge the guard by adding a cost-only effective_bits override."""
    hf_ptq, args = _parse_hf_ptq_args(
        monkeypatch, "--pyt_ckpt_path", "dummy", "--kv_cache_qformat", "none"
    )
    non_safe = next(k for k in QUANT_CFG_CHOICES if k not in hf_ptq._AUTO_QUANTIZE_QFORMATS)
    tampered = QuantizeConfig(**{**QUANT_CFG_CHOICES[non_safe], "effective_bits": 4.5})
    aq = AutoQuantizeConfig(
        constraints=AutoQuantizeConstraints(effective_bits=5.4),
        candidate_formats=[QuantizeConfig(**QUANT_CFG_CHOICES["fp8"]), tampered],
    )
    with pytest.raises(ValueError, match="not supported for unified checkpoint export"):
        hf_ptq._mtq_inputs_from_auto_quantize_config(aq, args)


def test_autoquant_config_from_deprecated_cli_flags(monkeypatch):
    """The deprecated --auto_quantize_* flags convert to an AutoQuantizeConfig with the shared
    base disabled + cost-excluded patterns appended (no new flags, no model introspection)."""
    hf_ptq, args = _parse_hf_ptq_args(
        monkeypatch,
        "--pyt_ckpt_path",
        "dummy",
        "--qformat",
        "fp8,nvfp4",
        "--auto_quantize_bits",
        "5.4",
        "--auto_quantize_cost_model",
        "active_moe",
        "--auto_quantize_active_moe_expert_ratio",
        "0.03125",
        "--kv_cache_qformat",
        "none",
    )
    aq = hf_ptq._auto_quantize_config_from_cli(args)

    assert aq.constraints.effective_bits == 5.4
    assert aq.constraints.cost_model == "active_moe"
    assert aq.constraints.cost.active_moe_expert_ratio == 0.03125
    assert aq.auto_quantize_method == "gradient"
    assert aq.score_boundary is None
    assert aq.score_size == 128
    # candidates come from --qformat and resolve to their shipped presets.
    assert [hf_ptq._match_candidate_to_preset(f)[0] for f in aq.candidate_formats] == [
        "fp8",
        "nvfp4",
    ]
    # base disabled + base cost-excluded appended from the shared units (no introspection).
    assert "*output_layer*" in aq.disabled_layers
    assert aq.cost_excluded_layers == ["*visual*", "*mtp*", "*vision_tower*"]
