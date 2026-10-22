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

import copy
import io
import json
import math
from types import SimpleNamespace

import pytest
import torch
from _test_utils.torch.distributed.utils import DistributedWorkerPool, spawn_multiprocess_job
from _test_utils.torch.quantization.models import SimpleConv, SimpleConvLinear, SimpleLinear

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
import modelopt.torch.quantization.model_quant as model_quant
from modelopt.torch.opt.searcher import LPS
from modelopt.torch.quantization._auto_quantize_cost import (
    EXCLUDED_MODULE_NAME_PATTERNS_KEY,
    _get_module_weight_numel,
    get_auto_quantize_cost_model,
    infer_active_moe_expert_ratio,
)
from modelopt.torch.quantization.algorithms import (
    AutoQuantizeGradientSearcher,
    AutoQuantizeHiddenReconSearcher,
    AutoQuantizeKLDivSearcher,
    QuantRecipe,
    QuantRecipeHparam,
    _AutoQuantizeBaseSearcher,
    _get_expected_acceptance_rate_from_log_probs,
    _get_hidden_recon_score,
    _get_kl_div_from_log_probs,
    _get_kl_div_loss,
    _get_kl_div_per_token,
    _module_search_space_signature,
    estimate_quant_compression,
)
from modelopt.torch.quantization.config import _base_disable_all, _default_disabled_quantizer_cfg
from modelopt.torch.utils import safe_load, safe_save
from modelopt.torch.utils.distributed import DistributedProcessGroup, ParallelState


class _AttentionLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(32, 32)
        self.k_proj = torch.nn.Linear(32, 32)
        self.v_proj = torch.nn.Linear(32, 32)
        self.o_proj = torch.nn.Linear(32, 32)

    def forward(self, x):
        for layer in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            x = layer(x)
        return x


class TransformerBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _AttentionLayer()
        self.mlp = torch.nn.Linear(32, 32)

    def forward(self, x):
        x = self.attn(x)
        x = self.mlp(x)
        return x

    def get_input(self):
        return torch.randn(1, 4, 32)


class _LinearAttentionLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj_qkv = torch.nn.Linear(32, 32)
        self.in_proj_z = torch.nn.Linear(32, 32)
        self.in_proj_a = torch.nn.Linear(32, 32)
        self.in_proj_b = torch.nn.Linear(32, 32)
        self.out_proj = torch.nn.Linear(32, 32)

    def forward(self, x):
        x = self.in_proj_qkv(x) + self.in_proj_z(x)
        x = x + self.in_proj_a(x) + self.in_proj_b(x)
        return self.out_proj(x)


class _GroupBoundaryModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            use_cache=True,
            text_config=SimpleNamespace(use_cache=True),
        )
        self.use_cache_seen = []
        self.self_attn = _AttentionLayer()
        self.linear_attn = _LinearAttentionLayer()

    def forward(self, x):
        self.use_cache_seen.append((self.config.use_cache, self.config.text_config.use_cache))
        return self.linear_attn(x=self.self_attn(x=x))

    def get_input(self):
        return torch.randn(1, 4, 32)


class _ExplicitCacheGroupBoundaryModel(torch.nn.Module):
    class CacheAwareAttention(_AttentionLayer):
        def forward(self, x, cache_params=None):
            if cache_params is not None:
                cache_params["positions"].append(len(cache_params["positions"]) + 1)
            return super().forward(x)

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(use_cache=False)
        self.cache = {"positions": []}
        self.self_attn = self.CacheAwareAttention()

    def forward(self, x):
        # Pass the mutable cache positionally to verify signature binding, not just kwargs.
        return self.self_attn(x, self.cache)

    def get_input(self):
        return torch.randn(1, 4, 32)


class _ExplicitMutableStateGroupBoundaryModel(torch.nn.Module):
    class StateAwareAttention(_AttentionLayer):
        def forward(self, x, state=None, cache=None):
            mutable = state if state is not None else cache
            if mutable is not None:
                mutable["positions"].append(len(mutable["positions"]) + 1)
            return super().forward(x)

    def __init__(self, input_name, as_keyword):
        super().__init__()
        self.config = SimpleNamespace(use_cache=False)
        self.mutable = {"positions": []}
        self.input_name = input_name
        self.as_keyword = as_keyword
        self.self_attn = self.StateAwareAttention()

    def forward(self, x):
        if self.as_keyword:
            return self.self_attn(x, **{self.input_name: self.mutable})
        if self.input_name == "state":
            return self.self_attn(x, self.mutable)
        return self.self_attn(x, None, self.mutable)

    def get_input(self):
        return torch.randn(1, 4, 32)


class _ExplicitUseCacheGroupBoundaryModel(torch.nn.Module):
    class CacheFlagAttention(_AttentionLayer):
        def forward(self, x, use_cache=False):
            return super().forward(x)

    def __init__(self, use_cache):
        super().__init__()
        self.config = SimpleNamespace(use_cache=False)
        self.use_cache = use_cache
        self.self_attn = self.CacheFlagAttention()

    def forward(self, x):
        return self.self_attn(x, use_cache=self.use_cache)

    def get_input(self):
        return torch.randn(1, 4, 32)


class _NestedAttentionContainerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(use_cache=False)
        self.self_attn = torch.nn.ModuleList([_AttentionLayer(), _AttentionLayer()])

    def forward(self, x):
        for attention in self.self_attn:
            x = attention(x)
        return x

    def get_input(self):
        return torch.randn(1, 4, 32)


class _FailingUseCacheConfig:
    def __init__(self):
        self._use_cache = True

    @property
    def use_cache(self):
        return self._use_cache

    @use_cache.setter
    def use_cache(self, value):
        self._use_cache = value
        if value is False:
            raise RuntimeError("cannot disable use_cache")


class _PartiallyFailingCacheConfigModel(_GroupBoundaryModel):
    def __init__(self):
        super().__init__()
        self.failing_config_module = torch.nn.Module()
        self.failing_config_module.config = _FailingUseCacheConfig()


class _AutoQuantMoeModel(torch.nn.Module):
    def __init__(self, num_experts_attr="num_experts"):
        super().__init__()
        self.config = SimpleNamespace(text_config=SimpleNamespace(num_experts_per_tok=2))
        setattr(self.config.text_config, num_experts_attr, 8)
        self.mlp = torch.nn.Module()
        self.mlp.experts = torch.nn.ModuleList()
        for _ in range(2):
            expert = torch.nn.Module()
            expert.gate_proj = torch.nn.Linear(32, 32)
            expert.up_proj = torch.nn.Linear(32, 32)
            expert.down_proj = torch.nn.Linear(32, 32)
            self.mlp.experts.append(expert)
        self.mlp.shared_expert = torch.nn.Linear(32, 32)

    def forward(self, x):
        y = self.mlp.shared_expert(x)
        for expert in self.mlp.experts:
            y = y + expert.down_proj(expert.gate_proj(x) + expert.up_proj(x))
        return y

    def get_input(self):
        return torch.randn(1, 4, 32)


@pytest.mark.parametrize(
    ("quant_cfg", "other_quant_cfg", "is_less_than"),
    [
        (mtq.FP8_DEFAULT_CFG, None, True),
        (mtq.NVFP4_DEFAULT_CFG, mtq.FP8_DEFAULT_CFG, True),
        (None, mtq.INT8_DEFAULT_CFG, False),
    ],
)
def test_quant_recipe(quant_cfg, other_quant_cfg, is_less_than):
    qr_this = QuantRecipe(quant_cfg)
    qr_other = QuantRecipe(other_quant_cfg)
    assert (qr_this < qr_other) == is_less_than

    qr_this_duplicate = QuantRecipe(quant_cfg)
    assert qr_this_duplicate == qr_this


def test_module_search_space_signature_is_canonical_and_uses_full_configs():
    fp8_recipe = QuantRecipe(mtq.FP8_DEFAULT_CFG)
    int8_recipe = QuantRecipe(mtq.INT8_DEFAULT_CFG)

    def signature(recipes):
        return _module_search_space_signature(
            [
                {
                    "module_name_patterns": ("*mlp*",),
                    "quant_recipes": recipes,
                    "allow_no_quant": False,
                }
            ]
        )

    assert signature([fp8_recipe, int8_recipe]) == signature([int8_recipe, fp8_recipe])

    custom_max_cfg = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    custom_max_cfg["quant_cfg"].append({"quantizer_name": "unused_quantizer", "enable": False})
    custom_mse_cfg = copy.deepcopy(custom_max_cfg)
    custom_mse_cfg["algorithm"] = "mse"
    max_recipe = QuantRecipe(custom_max_cfg, name="CUSTOM_MODULE_0_0")
    mse_recipe = QuantRecipe(custom_mse_cfg, name="CUSTOM_MODULE_0_0")

    assert str(max_recipe) == str(mse_recipe)
    assert max_recipe != mse_recipe
    assert signature([max_recipe]) != signature([mse_recipe])


def test_quant_recipe_hparam():
    model_test = torch.nn.Linear(4, 16)
    model_ref = torch.nn.Linear(4, 16)
    model_ref.load_state_dict(model_test.state_dict())

    model_test = mtq.quantize(model_test, mtq.INT8_DEFAULT_CFG)
    model_ref = mtq.quantize(model_ref, mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG)

    search_recipes = [
        QuantRecipe(mtq.INT8_DEFAULT_CFG),
        QuantRecipe(mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG),
    ]
    hparam = QuantRecipeHparam(
        search_recipes,
        quant_modules=[model_test],
    )
    model_test._register_hparam("quant_recipe", hparam)
    assert model_test.quant_recipe == QuantRecipe(mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG)
    assert model_test.get_hparam("quant_recipe").choices == sorted(
        [*search_recipes, QuantRecipe(quant_cfg=None)]
    )

    model_test.quant_recipe = QuantRecipe(mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG)
    inputs = torch.randn(1, 4, 4)
    output_test = model_test(inputs)
    output_ref = model_ref(inputs)

    assert torch.allclose(output_test, output_ref)


def test_quant_recipe_hparam_cost_weight():
    model_test = mtq.quantize(torch.nn.Linear(4, 16), mtq.INT8_DEFAULT_CFG)
    search_recipes = [QuantRecipe(mtq.INT8_DEFAULT_CFG)]
    hparam = QuantRecipeHparam(
        search_recipes,
        quant_modules=[model_test],
        quant_module_names=["layers.0.mlp.experts.0.down_proj"],
        cost_weight=0.25,
    )

    dense_cost = hparam.get_cost(QuantRecipe(quant_cfg=None))
    int8_cost = hparam.get_cost(QuantRecipe(mtq.INT8_DEFAULT_CFG))

    assert dense_cost == pytest.approx(model_test.weight.numel() * 0.25)
    assert int8_cost == pytest.approx(model_test.weight.numel() * 0.25 * 0.5)


def test_quant_recipe_hparam_zero_cost_weight():
    model_test = mtq.quantize(torch.nn.Linear(4, 16), mtq.INT8_DEFAULT_CFG)
    hparam = QuantRecipeHparam(
        [QuantRecipe(mtq.INT8_DEFAULT_CFG)],
        quant_modules=[model_test],
        quant_module_names=["visual.blocks.0.attn.qkv"],
        cost_weight=0.0,
    )

    assert hparam.get_cost(QuantRecipe(quant_cfg=None)) == pytest.approx(0.0)
    assert hparam.get_cost(QuantRecipe(mtq.INT8_DEFAULT_CFG)) == pytest.approx(0.0)


def test_quant_recipe_hparam_cost_weight_and_effective_bits_compose():
    """cost_weight (active_moe) and effective_bits stack multiplicatively in get_cost."""
    model_test = mtq.quantize(torch.nn.Linear(4, 16), mtq.NVFP4_DEFAULT_CFG)
    numel = model_test.weight.numel()
    hparam = QuantRecipeHparam(
        [QuantRecipe(mtq.NVFP4_DEFAULT_CFG)],
        quant_modules=[model_test],
        quant_module_names=["layers.0.mlp.experts.0.down_proj"],
        cost_weight=0.03125,
    )

    # NVFP4's library-default effective_bits (4.5, from configs/numerics/nvfp4) stacks with cost_weight:
    # numel * cost_weight * (effective_bits / 16).
    assert hparam.get_cost(QuantRecipe(mtq.NVFP4_DEFAULT_CFG)) == pytest.approx(
        numel * 0.03125 * (4.5 / 16)
    )
    # A recipe-level effective_bits override (8.0) wins over the library default and still stacks.
    override = QuantRecipe({**mtq.NVFP4_DEFAULT_CFG, "effective_bits": 8.0}, name="NVFP4_8B")
    assert hparam.get_cost(override) == pytest.approx(numel * 0.03125 * 0.5)


def test_lps_solve_top_k_returns_objective_ranked_distinct_solutions():
    solver = LPS(
        name="top_k",
        constraints={"cost": 2.0},
        constraints_to_candidate_costs={
            "cost": [
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        },
        candidate_scores=[
            [0.0, 1.0],
            [0.0, 2.0],
        ],
    )

    assert solver.solve_top_k(3) == [
        ([0, 0], "Optimal"),
        ([1, 0], "Optimal"),
        ([0, 1], "Optimal"),
    ]


def test_lps_solve_top_k_enforces_pairwise_minimum_hamming_distance():
    solver = LPS(
        name="hamming",
        constraints={"cost": 3.0},
        constraints_to_candidate_costs={"cost": [[1.0, 1.0]] * 3},
        candidate_scores=[[0.0, 1.0]] * 3,
    )

    solutions = solver.solve_top_k(4, min_hamming_distance=2)
    selections = [selection for selection, _ in solutions]

    assert len(selections) == 4
    assert all(
        sum(lhs_idx != rhs_idx for lhs_idx, rhs_idx in zip(lhs, rhs)) >= 2
        for lhs_pos, lhs in enumerate(selections)
        for rhs in selections[lhs_pos + 1 :]
    )
    assert solver.last_solve_metadata["min_hamming_distance"] == 2


def test_lps_infeasible_hamming_distance_returns_rank_zero_and_reports_partial_pool():
    solver = LPS(
        name="infeasible_hamming",
        constraints={"cost": 3.0},
        constraints_to_candidate_costs={"cost": [[1.0, 1.0]] * 3},
        candidate_scores=[[0.0, 1.0]] * 3,
    )

    assert solver.solve_top_k(3, min_hamming_distance=4) == [([0, 0, 0], "Optimal")]
    assert solver.last_solve_metadata["returned_solutions"] == 1
    assert solver.last_solve_metadata["terminal_status"] == "Infeasible"


def test_lps_objective_regret_and_cost_bounds_are_preserved():
    solver = LPS(
        name="regret",
        constraints={"cost": 3.0},
        constraints_to_candidate_costs={"cost": [[1.0, 2.0], [1.0, 2.0]]},
        candidate_scores=[[1.0, 2.0], [1.0, 2.0]],
    )

    solutions = solver.solve_top_k(4, max_objective_regret=0.5)

    objectives = [
        sum(solver.candidate_scores[i][choice] for i, choice in enumerate(selection))
        for selection, _ in solutions
    ]
    assert objectives == sorted(objectives)
    assert objectives == [2.0, 3.0, 3.0]
    assert all(
        sum(
            solver.constraints_to_candidate_costs["cost"][i][choice]
            for i, choice in enumerate(selection)
        )
        <= 3.0
        for selection, _ in solutions
    )


def _make_candidate_pool_searcher(**config_overrides):
    searcher = AutoQuantizeGradientSearcher()
    searcher.candidate_stats = {
        f"layer_{layer_id}.quant_recipe": {
            "formats": ["low", "high"],
            "scores": [1.0, 2.0 + layer_id / 10],
            "costs": [1.0, 1.0],
            "module_names": [f"layer_{layer_id}"],
        }
        for layer_id in range(4)
    }
    searcher.cost_model = "weight"
    searcher.config = {
        **searcher.default_search_config,
        "candidate_pool_size": 4,
        "candidate_min_hamming_distance": 1,
        "candidate_pool_selection": "farthest",
        **config_overrides,
    }
    return searcher


def test_autoquant_candidate_pool_farthest_selection_is_deterministic_and_serializable():
    first = _make_candidate_pool_searcher()
    second = _make_candidate_pool_searcher()

    first_packets, first_satisfied = first._enumerate_candidate_packets(4.0, 4.0, 3, False)
    second_packets, second_satisfied = second._enumerate_candidate_packets(4.0, 4.0, 3, False)

    assert first_satisfied and second_satisfied
    assert [packet["selection"] for packet in first_packets] == [
        packet["selection"] for packet in second_packets
    ]
    assert first_packets[0]["pool_rank"] == 0
    assert [packet["pool_selection_rank"] for packet in first_packets] == [0, 1, 2]
    json.dumps(first_packets)
    json.dumps(first._candidate_pool_provenance)


def test_autoquant_candidate_pool_relaxes_infeasible_distance_with_provenance():
    searcher = _make_candidate_pool_searcher(
        candidate_min_hamming_distance=5,
        candidate_distance_fallback="relax",
    )

    with pytest.warns(UserWarning, match="relaxed the distance"):
        packets, is_satisfied = searcher._enumerate_candidate_packets(4.0, 4.0, 3, False)

    assert is_satisfied
    assert len(packets) == 3
    assert searcher._candidate_pool_provenance["strict_pool_size"] == 1
    assert searcher._candidate_pool_provenance["fallback_used"] is True
    assert searcher._candidate_pool_provenance["selected_pairwise_hamming_distance"]["minimum"] < 5
    assert all(packet["effective_bits"] <= 16.0 for packet in packets)


def test_autoquant_candidate_diversity_patterns_scope_hamming_decisions():
    searcher = _make_candidate_pool_searcher(
        candidate_pool_size=2,
        candidate_min_hamming_distance=2,
        candidate_diversity_group_patterns=["layer_[01]"],
        candidate_pool_selection="objective",
    )

    packets, is_satisfied = searcher._enumerate_candidate_packets(4.0, 4.0, 2, False)

    assert is_satisfied
    assert len(packets) == 2
    assert searcher._candidate_pool_provenance["diversity_group_count"] == 2
    assert (
        sum(
            packets[0]["selection"][f"layer_{layer_id}.quant_recipe"]
            != packets[1]["selection"][f"layer_{layer_id}.quant_recipe"]
            for layer_id in range(2)
        )
        == 2
    )


def test_kl_top_k_preserves_established_threshold_recipe_at_rank_zero():
    searcher = AutoQuantizeKLDivSearcher()
    searcher.candidate_stats = {
        "layer_0": {
            "formats": ["low", "mid", "high"],
            "scores": [10.0, 4.0, 0.0],
            "costs": [2.0, 5.0, 8.0],
        },
        "layer_1": {
            "formats": ["low", "mid", "high"],
            "scores": [10.0, 4.0, 0.0],
            "costs": [2.0, 5.0, 8.0],
        },
    }
    default_recipe, is_satisfied = searcher.run_search_with_stats(10.0)

    packets, candidates_satisfied = searcher._enumerate_candidate_packets(
        max_weight_size=10.0,
        total_weight_size=16.0,
        num_candidates=2,
        verbose=False,
    )

    assert is_satisfied and candidates_satisfied
    assert packets[0]["recipe"] == {name: info["format"] for name, info in default_recipe.items()}
    assert [packet["first_stage_score"] for packet in packets] == [4.0, 10.0]
    assert [packet["first_stage_rank"] for packet in packets] == [0, 1]


@pytest.mark.parametrize(
    ("distance", "fallback", "regret", "patterns", "expected"),
    [
        (3, "none", None, None, [(1, 1, 0), (0, 0, 1)]),
        (3, "none", 0.0, None, [(1, 1, 0)]),
        (4, "relax", None, None, [(1, 1, 0), (1, 0, 0)]),
        (2, "relax", None, ["layer_0"], [(1, 1, 0), (0, 0, 0)]),
    ],
)
def test_kl_top_k_fills_pool_after_diversity_filtering(
    distance, fallback, regret, patterns, expected
):
    searcher = AutoQuantizeKLDivSearcher()
    searcher.config = {
        **searcher.default_search_config,
        "candidate_pool_size": 2,
        "candidate_min_hamming_distance": distance,
        "candidate_distance_fallback": fallback,
        "candidate_max_objective_regret": regret,
        "candidate_diversity_group_patterns": patterns,
    }
    searcher.candidate_stats = {
        f"layer_{i}": {"formats": ["low", "high"], "scores": [score, 0], "costs": [1, 2]}
        for i, score in enumerate((10, 9, 8))
    }

    packets, satisfied = searcher._enumerate_candidate_packets(5, 6, 2, False)

    assert satisfied
    assert [tuple(packet["selection"].values()) for packet in packets] == expected
    assert all(packet["cost"] <= 5 for packet in packets)


def test_hidden_recon_score_is_normalized_output_error():
    reference = torch.tensor([1.0, 2.0])
    quantized = torch.tensor([2.0, 2.0])

    assert _get_hidden_recon_score(reference, quantized).item() == pytest.approx(0.2)


@pytest.mark.parametrize("flatten", [False, True])
def test_hidden_recon_score_excludes_padding(flatten):
    reference = torch.tensor([[[1.0, 1.0], [10.0, 10.0]]])
    quantized = torch.tensor([[[2.0, 1.0], [100.0, 100.0]]])
    attention_mask = torch.tensor([[1, 0]])
    if flatten:
        reference = reference.flatten(0, 1)
        quantized = quantized.flatten(0, 1)

    assert _get_hidden_recon_score(
        reference,
        quantized,
        attention_mask,
    ).item() == pytest.approx(0.5)


def test_hidden_recon_parent_replays_do_not_rescore_children(monkeypatch):
    model = torch.nn.Sequential()
    model.add_module("block", torch.nn.Sequential())
    for name in ("leaf", "first", "last"):
        model.block.add_module(name, torch.nn.Linear(32, 32))
    inputs = torch.randn(1, 4, 32)

    def score(rules):
        monkeypatch.setattr(AutoQuantizeHiddenReconSearcher, "score_module_rules", rules)
        return mtq.auto_quantize(
            copy.deepcopy(model),
            constraints={"effective_bits": 8.0},
            quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG],
            data_loader=[inputs],
            forward_step=lambda model, batch: model(batch),
            num_calib_steps=1,
            num_score_steps=1,
            method="hidden_recon",
            score_boundary="local",
        )[1]["candidate_stats"]["block.leaf.quant_recipe"]["scores"]

    local_scores = score([])
    assert any(value > 0 for value in local_scores)
    assert score([r"^(block)\.(?:first|last)$"]) == pytest.approx(local_scores)


def test_kl_div_ignores_zero_probability_reference_terms():
    """Exact zero target probabilities contribute zero instead of producing NaN."""
    log_prob_unquant = torch.tensor([[[-torch.inf, 0.0]]])
    logits_quant = torch.tensor([[[0.0, 0.0]]])

    token_kl = _get_kl_div_per_token(log_prob_unquant, logits_quant)

    assert torch.isfinite(token_kl).all()
    assert token_kl.item() == pytest.approx(torch.log(torch.tensor(2.0)).item())
    assert _get_kl_div_loss(log_prob_unquant, logits_quant).item() == pytest.approx(token_kl.item())


def test_expected_acceptance_rate_is_exact_probability_overlap():
    log_prob_unquant = torch.log(torch.tensor([[[0.6, 0.4], [1.0, 0.0]]]))
    log_prob_quant = torch.log(torch.tensor([[[0.5, 0.5], [0.0, 1.0]]]))

    ear = _get_expected_acceptance_rate_from_log_probs(log_prob_unquant, log_prob_quant)

    torch.testing.assert_close(ear, torch.tensor([[0.9, 0.0]]))


def test_expected_acceptance_rate_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="must have the same shape"):
        _get_expected_acceptance_rate_from_log_probs(
            torch.zeros(1, 2, 3),
            torch.zeros(1, 2, 4),
        )


def test_kl_div_retains_infinite_penalty_for_missing_candidate_support():
    """A candidate zero probability remains infinite when the reference is positive."""
    log_prob_unquant = torch.tensor([[[0.0, -torch.inf]]])
    logits_quant = torch.tensor([[[-torch.inf, 0.0]]])

    token_kl = _get_kl_div_per_token(log_prob_unquant, logits_quant)

    assert torch.isposinf(token_kl).all()


def test_kl_div_nonfinite_candidate_fallback_is_opt_in_and_finite():
    """The recovery fallback replaces only non-finite candidate log-probabilities."""
    log_prob_unquant = torch.tensor([[[0.0, -torch.inf]]])
    log_prob_quant = torch.tensor([[[-torch.inf, torch.nan]]])

    exact = _get_kl_div_from_log_probs(log_prob_unquant, log_prob_quant)
    recovered = _get_kl_div_from_log_probs(
        log_prob_unquant,
        log_prob_quant,
        log_prob_fallback=-100.0,
    )

    assert torch.isposinf(exact).all()
    assert torch.isfinite(recovered).all()
    assert recovered.item() == pytest.approx(100.0)


def test_kl_div_nonfinite_candidate_fallback_preserves_finite_terms():
    """Enabling the fallback does not clip or otherwise change finite scores."""
    log_prob_unquant = torch.log_softmax(torch.tensor([[[1.0, 2.0, 3.0]]]), dim=-1)
    log_prob_quant = torch.log_softmax(torch.tensor([[[3.0, -200.0, 1.0]]]), dim=-1)

    exact = _get_kl_div_from_log_probs(log_prob_unquant, log_prob_quant)
    recovered = _get_kl_div_from_log_probs(
        log_prob_unquant,
        log_prob_quant,
        log_prob_fallback=-100.0,
    )

    torch.testing.assert_close(recovered, exact)


def test_kl_div_nonfinite_candidate_fallback_recovers_overflowed_logits():
    """The opt-in path converts an overflowed candidate forward into a finite penalty."""
    log_prob_unquant = torch.log_softmax(torch.tensor([[[1.0, 2.0]]]), dim=-1)
    logits_quant = torch.tensor([[[torch.inf, 0.0]]])

    exact = _get_kl_div_loss(log_prob_unquant, logits_quant)
    recovered = _get_kl_div_loss(
        log_prob_unquant,
        logits_quant,
        log_prob_fallback=-100.0,
    )

    assert torch.isnan(exact)
    assert torch.isfinite(recovered)


@pytest.mark.parametrize("fallback", [0.0, 1.0, math.inf, math.nan, True])
def test_kl_div_rejects_invalid_nonfinite_candidate_fallback(fallback):
    log_prob = torch.tensor([[[0.0]]])

    with pytest.raises(ValueError, match="finite negative"):
        _get_kl_div_from_log_probs(log_prob, log_prob, log_prob_fallback=fallback)


def test_auto_quantize_cost_model_excludes_module_name_patterns():
    cost_model = get_auto_quantize_cost_model("weight")
    cost_constraints = {EXCLUDED_MODULE_NAME_PATTERNS_KEY: ["*visual*", "*vision_tower*", "*mtp*"]}

    # Modules whose name matches an excluded pattern contribute zero cost weight.
    assert cost_model.module_cost_weight(["model.visual.blocks.0.attn.qkv"], cost_constraints) == 0
    assert cost_model.module_cost_weight(["model.mtp.layers.0.mlp"], cost_constraints) == 0
    # Non-excluded modules keep full weight.
    assert cost_model.module_cost_weight(["lm_head"], cost_constraints) == 1.0
    # A group is only excluded when *all* of its module names match; a mixed group is not.
    assert (
        cost_model.module_cost_weight(
            ["model.visual.blocks.0.attn.qkv", "lm_head"], cost_constraints
        )
        == 1.0
    )


def test_active_moe_cost_model_counts_fused_experts_without_weight():
    fused_experts = torch.nn.Module()
    fused_experts.gate_up_proj = torch.nn.Parameter(torch.empty(2, 3, 5))
    fused_experts.down_proj = torch.nn.Parameter(torch.empty(2, 5, 3))
    cost_model = get_auto_quantize_cost_model("active_moe")
    cost_constraints = {
        "active_moe_expert_ratio": 0.25,
        EXCLUDED_MODULE_NAME_PATTERNS_KEY: ["*visual*"],
    }

    # Fused experts expose no `.weight`; their size is summed across all parameters.
    assert _get_module_weight_numel(fused_experts) == (
        fused_experts.gate_up_proj.numel() + fused_experts.down_proj.numel()
    )
    # Routed MoE experts are scaled by the active-expert ratio; excluded modules drop to zero.
    assert cost_model.module_cost_weight(["layers.0.mlp.experts"], cost_constraints) == 0.25
    assert cost_model.module_cost_weight(["model.visual.blocks.0.attn.qkv"], cost_constraints) == 0


@pytest.mark.parametrize("num_experts_attr", ["num_experts", "num_local_experts"])
def test_auto_quantize_active_moe_cost_model(num_experts_attr):
    model = _AutoQuantMoeModel(num_experts_attr)

    _, search_history = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 6.0, "cost_model": "active_moe"},
        quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG],
        data_loader=[model.get_input() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=2,
        num_score_steps=2,
    )

    assert search_history["cost_model"] == "active_moe"
    assert search_history["active_moe_expert_ratio"] == pytest.approx(0.25)
    weighted_no_quant_cost = sum(
        stats["costs"][-1] for stats in search_history["candidate_stats"].values()
    )
    assert search_history["cost_denominator"] == pytest.approx(weighted_no_quant_cost)
    routed_stats = [
        stats
        for stats in search_history["candidate_stats"].values()
        if any("mlp.experts" in name for name in stats["module_names"])
    ]
    shared_stats = [
        stats
        for stats in search_history["candidate_stats"].values()
        if any("mlp.shared_expert" in name for name in stats["module_names"])
    ]
    assert routed_stats
    assert shared_stats
    assert all(stats["cost_weight"] == pytest.approx(0.25) for stats in routed_stats)
    assert all(stats["cost_weight"] == pytest.approx(1.0) for stats in shared_stats)
    assert all("active_costs" not in stats for stats in search_history["candidate_stats"].values())


def test_auto_quantize_module_search_spaces_keep_fixed_routed_experts_costed():
    """A one-format routed rule is fixed in the LP but retained in active-MoE cost."""
    model = _AutoQuantMoeModel()
    int4_recipe = QuantRecipe(mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG)
    int8_recipe = QuantRecipe(mtq.INT8_DEFAULT_CFG)
    no_quant_recipe = QuantRecipe(None)

    searched_model, search_history = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 6.0, "cost_model": "active_moe"},
        quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG],
        module_search_spaces=[
            {
                "module_name_patterns": ["*mlp.experts*"],
                "quantization_formats": [mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG],
                "allow_no_quant": False,
            },
            {
                "module_name_patterns": ["*mlp.shared_expert*"],
                "quantization_formats": [
                    mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
                    mtq.INT8_DEFAULT_CFG,
                ],
                "allow_no_quant": False,
            },
        ],
        data_loader=[model.get_input() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=2,
        num_score_steps=2,
    )

    stats = search_history["candidate_stats"]
    routed = next(
        candidate
        for candidate in stats.values()
        if any("mlp.experts" in name for name in candidate["module_names"])
    )
    shared = next(
        candidate
        for candidate in stats.values()
        if any("mlp.shared_expert" in name for name in candidate["module_names"])
    )

    assert routed["formats"] == [int4_recipe]
    assert routed["allow_no_quant"] is False
    assert routed["cost_weight"] == pytest.approx(0.25)
    assert routed["costs"] == pytest.approx([routed["uncompressed_cost"] * 0.25])
    assert no_quant_recipe not in routed["formats"]
    assert shared["formats"] == [int4_recipe, int8_recipe]
    assert shared["allow_no_quant"] is False
    assert no_quant_recipe not in shared["formats"]
    assert search_history["cost_denominator"] == pytest.approx(
        sum(candidate["uncompressed_cost"] for candidate in stats.values())
    )
    routed_best = [
        recipe
        for name, recipe in search_history["best"]["recipe"].items()
        if any("mlp.experts" in module for module in stats[name]["module_names"])
    ]
    assert routed_best and all(recipe == int4_recipe for recipe in routed_best)
    routed_hparam = searched_model.mlp.experts[0].gate_proj.get_hparam("quant_recipe")
    assert not routed_hparam.is_configurable
    assert routed_hparam.active == int4_recipe


def test_auto_quantize_fixed_ptq_baseline_keeps_unmatched_routed_experts_costed():
    """A normal PTQ config fixes unmatched groups while explicit families remain searched."""
    model = _AutoQuantMoeModel()
    int4_recipe = QuantRecipe(mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG)
    no_quant_recipe = QuantRecipe(None)

    searched_model, search_history = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 6.0, "cost_model": "active_moe"},
        fixed_quantization_config=mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
        module_search_spaces=[
            {
                "module_name_patterns": ["*mlp.shared_expert*"],
                "quantization_formats": [
                    mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
                    mtq.INT8_DEFAULT_CFG,
                ],
                "allow_no_quant": False,
            }
        ],
        data_loader=[model.get_input() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=2,
        num_score_steps=2,
    )

    stats = search_history["candidate_stats"]
    routed_name, routed = next(
        (name, candidate)
        for name, candidate in stats.items()
        if any("mlp.experts" in module for module in candidate["module_names"])
    )
    shared = next(
        candidate
        for candidate in stats.values()
        if any("mlp.shared_expert" in module for module in candidate["module_names"])
    )

    assert search_history["quantization_formats_signature"] == ()
    assert search_history["fixed_quantization_config_signature"] == int4_recipe.checkpoint_signature
    assert routed["formats"] == [int4_recipe]
    assert routed["is_fixed"] is True
    assert routed["allow_no_quant"] is False
    assert routed["cost_weight"] == pytest.approx(0.25)
    assert no_quant_recipe not in routed["formats"]
    assert shared["is_fixed"] is False
    assert search_history["best"]["recipe"][routed_name] == int4_recipe
    routed_hparam = searched_model.mlp.experts[0].gate_proj.get_hparam("quant_recipe")
    assert not routed_hparam.is_configurable
    assert routed_hparam.is_fixed
    assert routed_hparam.active == int4_recipe

    exported = mtq.get_auto_quantize_config(search_history)
    fixed_entries = search_history["fixed_quantization_config"]["quant_cfg"]
    assert exported["quant_cfg"][: len(fixed_entries)] == fixed_entries


def test_auto_quantize_fixed_ptq_baseline_honors_full_model_disables():
    model = TransformerBlock()
    fixed_config = copy.deepcopy(mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG)
    fixed_config["quant_cfg"].append({"quantizer_name": "*attn.o_proj*", "enable": False})

    searched_model, search_history = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 8.0},
        fixed_quantization_config=fixed_config,
        module_search_spaces=[
            {
                "module_name_patterns": ["*mlp*"],
                "quantization_formats": [
                    mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
                    mtq.INT8_DEFAULT_CFG,
                ],
            }
        ],
        data_loader=[model.get_input()],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=1,
        num_score_steps=1,
    )

    o_proj_hparam = searched_model.attn.o_proj.get_hparam("quant_recipe")
    assert o_proj_hparam.is_fixed
    assert o_proj_hparam.active == QuantRecipe(None)
    o_proj_stats = next(
        candidate
        for candidate in search_history["candidate_stats"].values()
        if candidate["module_names"] == ["attn.o_proj"]
    )
    assert o_proj_stats["costs"] == [o_proj_stats["uncompressed_cost"]]
    exported = mtq.get_auto_quantize_config(search_history)
    assert any(
        entry["quantizer_name"] == "*attn.o_proj*" and entry["enable"] is False
        for entry in exported["quant_cfg"]
    )


def test_auto_quantize_fixed_ptq_baseline_isolated_from_search_calibration(monkeypatch):
    model = TransformerBlock()
    calibration_states = []
    original_calibrate = model_quant.calibrate

    def recording_calibrate(model, algorithm="max", forward_loop=None):
        algorithm_name = algorithm.get("method") if isinstance(algorithm, dict) else algorithm
        calibration_states.append(
            {
                "algorithm": algorithm_name,
                "fixed_enabled": model.attn.q_proj.weight_quantizer.is_enabled,
                "fixed_num_bits": model.attn.q_proj.weight_quantizer.num_bits,
            }
        )
        return original_calibrate(model, algorithm=algorithm, forward_loop=forward_loop)

    monkeypatch.setattr(model_quant, "calibrate", recording_calibrate)

    searched_model, _ = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 8.0},
        fixed_quantization_config=mtq.FP8_DEFAULT_CFG,
        module_search_spaces=[
            {
                "module_name_patterns": ["*mlp*"],
                "quantization_formats": [mtq.INT4_AWQ_CFG],
            }
        ],
        data_loader=[model.get_input()],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=1,
        num_score_steps=1,
    )

    awq_state = next(state for state in calibration_states if "awq" in state["algorithm"])
    fp8_state = next(state for state in calibration_states if state["algorithm"] == "max")
    assert not awq_state["fixed_enabled"]
    assert fp8_state["fixed_enabled"]
    assert fp8_state["fixed_num_bits"] == (4, 3)
    assert searched_model.attn.q_proj.weight_quantizer.is_enabled
    assert searched_model.attn.q_proj.weight_quantizer.num_bits == (4, 3)


def test_auto_quantize_fixed_ptq_baseline_rejects_global_formats():
    with pytest.raises(ValueError, match="cannot be combined"):
        mtq.auto_quantize(
            TransformerBlock(),
            quantization_formats=[mtq.INT8_DEFAULT_CFG],
            fixed_quantization_config=mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
            module_search_spaces=[
                {
                    "module_name_patterns": ["*mlp*"],
                    "quantization_formats": [mtq.INT8_DEFAULT_CFG],
                }
            ],
        )


def test_auto_quantize_fixed_ptq_baseline_requires_explicit_search_spaces():
    with pytest.raises(ValueError, match="requires at least one explicit"):
        mtq.auto_quantize(
            TransformerBlock(),
            fixed_quantization_config=mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
        )


@pytest.mark.parametrize("formats", ["FP8_DEFAULT_CFG", mtq.FP8_DEFAULT_CFG, ()])
def test_auto_quantize_rejects_non_list_global_formats(formats):
    with pytest.raises(TypeError, match="`quantization_formats` must be a list"):
        mtq.auto_quantize(TransformerBlock(), quantization_formats=formats)


@pytest.mark.parametrize("formats", [[], [None]])
def test_auto_quantize_rejects_empty_global_formats(formats):
    with pytest.raises(ValueError, match="`quantization_formats` must"):
        mtq.auto_quantize(TransformerBlock(), quantization_formats=formats)


@pytest.mark.parametrize("formats", ["FP8_DEFAULT_CFG", mtq.FP8_DEFAULT_CFG, ()])
def test_auto_quantize_rejects_non_list_module_formats(formats):
    with pytest.raises(
        TypeError, match=r"module_search_spaces\.quantization_formats must be a list"
    ):
        mtq.auto_quantize(
            TransformerBlock(),
            quantization_formats=[mtq.INT8_DEFAULT_CFG],
            module_search_spaces=[
                {
                    "module_name_patterns": ["*mlp*"],
                    "quantization_formats": formats,
                }
            ],
        )


@pytest.mark.parametrize("formats", [[], [None]])
def test_auto_quantize_rejects_empty_module_formats(formats):
    with pytest.raises(ValueError, match=r"module_search_spaces\.quantization_formats must"):
        mtq.auto_quantize(
            TransformerBlock(),
            quantization_formats=[mtq.INT8_DEFAULT_CFG],
            module_search_spaces=[
                {
                    "module_name_patterns": ["*mlp*"],
                    "quantization_formats": formats,
                }
            ],
        )


def test_auto_quantize_fixed_module_isolated_from_unrelated_calibration(monkeypatch):
    model = TransformerBlock()
    calibration_states = []
    original_calibrate = model_quant.calibrate

    def recording_calibrate(model, algorithm="max", forward_loop=None):
        algorithm_name = algorithm.get("method") if isinstance(algorithm, dict) else algorithm
        weight_quantizer = model.mlp.weight_quantizer
        calibration_states.append(
            {
                "algorithm": algorithm_name,
                "enabled": weight_quantizer.is_enabled,
                "num_bits": weight_quantizer.num_bits,
                "quantizer_id": id(weight_quantizer),
            }
        )
        return original_calibrate(model, algorithm=algorithm, forward_loop=forward_loop)

    monkeypatch.setattr(model_quant, "calibrate", recording_calibrate)

    searched_model, _ = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 6.0},
        quantization_formats=[mtq.INT4_AWQ_CFG],
        module_search_spaces=[
            {
                "module_name_patterns": ["*mlp*"],
                "quantization_formats": [mtq.FP8_DEFAULT_CFG],
                "allow_no_quant": False,
            }
        ],
        data_loader=[model.get_input()],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=1,
        num_score_steps=1,
    )

    awq_state = next(state for state in calibration_states if "awq" in state["algorithm"])
    fp8_state = next(state for state in calibration_states if state["algorithm"] == "max")
    assert not awq_state["enabled"]
    assert fp8_state["enabled"]
    assert fp8_state["num_bits"] == (4, 3)
    assert awq_state["quantizer_id"] != fp8_state["quantizer_id"]
    assert id(searched_model.mlp.weight_quantizer) == fp8_state["quantizer_id"]
    assert searched_model.mlp.weight_quantizer.pre_quant_scale is None
    assert not searched_model.mlp.get_hparam("quant_recipe").is_configurable
    assert searched_model.mlp.get_hparam("quant_recipe").active == QuantRecipe(mtq.FP8_DEFAULT_CFG)


def test_auto_quantize_rejects_infeasible_resolved_module_search_space(monkeypatch):
    def unexpected_calibration(*args, **kwargs):
        pytest.fail("infeasible search should fail before calibration")

    monkeypatch.setattr(model_quant, "calibrate", unexpected_calibration)
    model = TransformerBlock()

    with pytest.raises(ValueError, match=r"minimum achievable effective bits is 8\.0000"):
        mtq.auto_quantize(
            model,
            constraints={"effective_bits": 6.0},
            quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG],
            module_search_spaces=[
                {
                    "module_name_patterns": ["*"],
                    "quantization_formats": [mtq.FP8_DEFAULT_CFG],
                    "allow_no_quant": False,
                }
            ],
            data_loader=[model.get_input()],
            forward_step=lambda model, batch: model(batch),
            loss_func=lambda output, data: output.sum(),
            num_calib_steps=1,
            num_score_steps=1,
        )


def test_auto_quantize_module_search_space_cannot_split_runtime_group():
    model = TransformerBlock()

    with pytest.raises(ValueError, match="partially matches runtime-grouped modules"):
        mtq.auto_quantize(
            model,
            constraints={"effective_bits": 6.0},
            quantization_formats=[
                mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
                mtq.INT8_DEFAULT_CFG,
            ],
            module_search_spaces=[
                {
                    "module_name_patterns": ["*q_proj"],
                    "quantization_formats": [mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG],
                    "allow_no_quant": False,
                }
            ],
            data_loader=[model.get_input()],
            forward_step=lambda model, batch: model(batch),
            loss_func=lambda output, data: output.sum(),
            num_calib_steps=1,
            num_score_steps=1,
        )


def test_active_moe_ratio_requires_single_config_object():
    model = torch.nn.Module()
    model.config = SimpleNamespace(
        num_experts_per_tok=2,
        text_config=SimpleNamespace(num_experts=8),
    )

    assert infer_active_moe_expert_ratio(model) is None


def test_active_moe_search_prefers_budget_lower_bound():
    searcher = AutoQuantizeGradientSearcher()
    searcher.config = {"cost_model": "active_moe"}
    searcher.cost_model = "active_moe"
    searcher.candidate_stats = {
        "layers.0.mlp.quant_recipe": {
            "formats": ["under_budget", "near_budget"],
            "costs": [1.0, 4.95],
            "scores": [0.0, 10.0],
        }
    }

    best_recipes, is_satisfied = searcher.run_search_with_stats(5.0)

    assert is_satisfied
    assert best_recipes["layers.0.mlp.quant_recipe"]["format"] == "near_budget"


def test_auto_quantize_local_boundary_scores_shared_expert_at_parent_mlp():
    searcher = AutoQuantizeGradientSearcher()
    searcher.config = {"score_boundary": "local"}

    score_modules = []
    for projection in ("gate_proj", "up_proj", "down_proj"):
        name = f"model.layers.0.mlp.shared_expert.{projection}"
        score_module = next(
            (
                result
                for rule in searcher._get_score_module_rules()
                if (result := searcher._apply_score_group_rule(name, rule)) is not None
            ),
            name,
        )
        score_modules.append(score_module)

    assert score_modules == ["model.layers.0.mlp"] * 3


@pytest.mark.parametrize("parent_name", ["mlp", "mixer"])
def test_auto_quantize_local_boundary_scores_fused_experts_at_parent(monkeypatch, parent_name):
    model = torch.nn.Module()
    parent = torch.nn.Module()
    parent.experts = torch.nn.Module()
    setattr(model, parent_name, parent)
    monkeypatch.setattr(
        "modelopt.torch.quantization.algorithms._is_hf_quant_fused_experts_module",
        lambda module: module is parent.experts,
    )
    searcher = AutoQuantizeGradientSearcher()
    searcher.model = model
    searcher.config = {"score_boundary": "local"}

    score_module = next(
        result
        for rule in searcher._get_score_module_rules()
        if (
            result := searcher._apply_score_group_rule(
                f"{parent_name}.experts",
                rule,
            )
        )
        is not None
    )

    assert score_module == parent_name


@pytest.mark.parametrize(
    ("module_name", "expected_parent"),
    [
        ("model.layers.0.self_attn.qkv_proj", "model.layers.0.self_attn"),
        ("model.layers.0.self_attn.o_proj", "model.layers.0.self_attn"),
        ("decoder.layers.0.self_attention.linear_qkv", "decoder.layers.0.self_attention"),
        ("decoder.layers.0.self_attention.linear_proj", "decoder.layers.0.self_attention"),
        ("transformer.h.0.attn.c_attn", "transformer.h.0.attn"),
        ("transformer.h.0.crossattention.c_attn", "transformer.h.0.crossattention"),
        ("encoder.layer.0.attention.self.query", "encoder.layer.0.attention"),
        ("model.layers.0.linear_attn.in_proj_qkv", "model.layers.0.linear_attn"),
    ],
)
def test_auto_quantize_group_boundary_maps_supported_attention_layouts(
    module_name, expected_parent
):
    class LookupModel:
        def get_submodule(self, name):
            assert name == expected_parent
            return _AttentionLayer()

    searcher = AutoQuantizeGradientSearcher()
    searcher.model = LookupModel()
    searcher.config = {"score_boundary": "group"}

    score_module = next(
        result
        for rule in searcher._get_score_module_rules()
        if (result := searcher._apply_score_group_rule(module_name, rule)) is not None
    )

    assert score_module == expected_parent


def test_auto_quantize_group_boundary_resolves_invoked_child_under_attention_container():
    model = _NestedAttentionContainerModel()
    _, search_state = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 8.0},
        quantization_formats=[
            mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
            mtq.INT8_WEIGHT_ONLY_CFG,
        ],
        data_loader=[model.get_input()],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=1,
        num_score_steps=1,
        method="gradient",
    )

    first_q_proj = model.self_attn[0].q_proj.get_hparam("quant_recipe")
    second_o_proj = model.self_attn[1].o_proj.get_hparam("quant_recipe")
    assert first_q_proj.score_modules == [model.self_attn[0]]
    assert second_o_proj.score_modules == [model.self_attn[1]]
    assert not hasattr(model.self_attn, "_hparams_for_scoring")
    assert any(
        score > 0 for stats in search_state["candidate_stats"].values() for score in stats["scores"]
    )


def test_auto_quantize_group_boundary_warns_on_unmapped_attention_projection():
    class RootAttentionProjection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(32, 32)

        def forward(self, x):
            return self.q_proj(x)

    model = RootAttentionProjection()
    batch = torch.randn(1, 4, 32)
    with pytest.warns(UserWarning, match=r"could not map.*q_proj"):
        mtq.auto_quantize(
            model,
            constraints={"effective_bits": 8.0},
            quantization_formats=[
                mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
                mtq.INT8_WEIGHT_ONLY_CFG,
            ],
            data_loader=[batch],
            forward_step=lambda model, data: model(data),
            loss_func=lambda output, data: output.sum(),
            num_calib_steps=1,
            num_score_steps=1,
        )


def test_auto_quantize_group_score_boundary_does_not_group_recipe_decisions():
    model = _GroupBoundaryModel()
    _, search_state = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 8.0},
        quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG],
        data_loader=[model.get_input()],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=1,
        num_score_steps=1,
        method="gradient",
    )

    self_qkv = model.self_attn.q_proj.get_hparam("quant_recipe")
    self_o = model.self_attn.o_proj.get_hparam("quant_recipe")
    linear_qkvz = model.linear_attn.in_proj_qkv.get_hparam("quant_recipe")
    linear_ba = model.linear_attn.in_proj_a.get_hparam("quant_recipe")
    linear_out = model.linear_attn.out_proj.get_hparam("quant_recipe")

    assert self_qkv is not self_o
    assert linear_qkvz is not linear_ba
    assert linear_qkvz is not linear_out
    assert linear_ba is not linear_out
    assert self_qkv.score_modules == [model.self_attn]
    assert self_o.score_modules == [model.self_attn]
    assert linear_qkvz.score_modules == [model.linear_attn]
    assert linear_ba.score_modules == [model.linear_attn]
    assert linear_out.score_modules == [model.linear_attn]
    assert search_state["score_boundary"] == "group"
    assert search_state["method"] == "gradient"
    assert any(
        score > 0 for stats in search_state["candidate_stats"].values() for score in stats["scores"]
    )
    assert model.config.use_cache is True
    assert model.config.text_config.use_cache is True
    assert (False, False) in model.use_cache_seen


@pytest.mark.parametrize("method", ["gradient", "hidden_recon"])
def test_auto_quantize_group_boundary_rejects_explicit_mutable_cache_before_replay(method):
    model = _ExplicitCacheGroupBoundaryModel()

    with pytest.raises(ValueError, match=r"mutable state/cache inputs \(cache_params\)"):
        mtq.auto_quantize(
            model,
            constraints={"effective_bits": 8.0},
            quantization_formats=[
                mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
                mtq.INT8_WEIGHT_ONLY_CFG,
            ],
            data_loader=[model.get_input()],
            forward_step=lambda model, batch: model(batch),
            loss_func=lambda output, data: output.sum(),
            num_calib_steps=1,
            num_score_steps=1,
            method=method,
        )

    # Each format runs one calibration forward. The parent replay must fail before a third,
    # baseline/candidate forward can mutate and reuse the same cache.
    assert model.cache["positions"] == [1, 2]


@pytest.mark.parametrize("input_name", ["state", "cache"])
@pytest.mark.parametrize("as_keyword", [False, True])
@pytest.mark.parametrize("method", ["gradient", "hidden_recon"])
def test_auto_quantize_group_boundary_rejects_generic_mutable_state_before_replay(
    input_name, as_keyword, method
):
    model = _ExplicitMutableStateGroupBoundaryModel(input_name, as_keyword)

    with pytest.raises(ValueError, match=rf"mutable state/cache inputs \({input_name}\)"):
        mtq.auto_quantize(
            model,
            constraints={"effective_bits": 8.0},
            quantization_formats=[
                mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
                mtq.INT8_WEIGHT_ONLY_CFG,
            ],
            data_loader=[model.get_input()],
            forward_step=lambda model, batch: model(batch),
            loss_func=lambda output, data: output.sum(),
            num_calib_steps=1,
            num_score_steps=1,
            method=method,
        )

    assert model.mutable["positions"] == [1, 2]


@pytest.mark.parametrize("use_cache", [False, True])
@pytest.mark.parametrize("method", ["gradient", "hidden_recon"])
def test_auto_quantize_group_boundary_handles_explicit_use_cache_flag(use_cache, method):
    model = _ExplicitUseCacheGroupBoundaryModel(use_cache)
    kwargs = {
        "constraints": {"effective_bits": 8.0},
        "quantization_formats": [
            mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
            mtq.INT8_WEIGHT_ONLY_CFG,
        ],
        "data_loader": [model.get_input()],
        "forward_step": lambda model, batch: model(batch),
        "loss_func": lambda output, data: output.sum(),
        "num_calib_steps": 1,
        "num_score_steps": 1,
        "method": method,
    }

    if use_cache:
        with pytest.raises(ValueError, match=r"mutable state/cache inputs \(use_cache=True\)"):
            mtq.auto_quantize(model, **kwargs)
    else:
        _, search_state = mtq.auto_quantize(model, **kwargs)
        assert any(
            score > 0
            for stats in search_state["candidate_stats"].values()
            for score in stats["scores"]
        )


@pytest.mark.parametrize("method", ["gradient", "hidden_recon"])
def test_auto_quantize_group_boundary_restores_cleanup_when_cache_disabling_fails(method):
    model = _PartiallyFailingCacheConfigModel()

    with pytest.raises(RuntimeError, match="cannot disable use_cache"):
        mtq.auto_quantize(
            model,
            constraints={"effective_bits": 8.0},
            quantization_formats=[
                mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
                mtq.INT8_WEIGHT_ONLY_CFG,
            ],
            data_loader=[model.get_input()],
            forward_step=lambda model, batch: model(batch),
            loss_func=lambda output, data: output.sum(),
            num_calib_steps=1,
            num_score_steps=1,
            method=method,
        )

    assert model.config.use_cache is True
    assert model.config.text_config.use_cache is True
    assert model.failing_config_module.config.use_cache is True
    for score_module in (model.self_attn, model.linear_attn):
        assert not hasattr(score_module, "_forward_original")
        assert not hasattr(score_module, "_backward_hook_handle")


def test_auto_quantize_gradient_defaults_to_group_boundary_and_supports_keyword_calls():
    model = _GroupBoundaryModel()
    _, search_state = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 8.0},
        quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG],
        data_loader=[model.get_input()],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=1,
        num_score_steps=1,
        method="gradient",
    )

    assert search_state["score_boundary"] == "group"
    assert search_state["method"] == "gradient"
    assert any(
        score > 0 for stats in search_state["candidate_stats"].values() for score in stats["scores"]
    )
    assert model.config.use_cache is True
    assert model.config.text_config.use_cache is True
    assert (False, False) in model.use_cache_seen


def test_auto_quantize_gradient_local_boundary_opt_out():
    model = _GroupBoundaryModel()
    _, search_state = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 8.0},
        quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG],
        data_loader=[model.get_input()],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=1,
        num_score_steps=1,
        method="gradient",
        score_boundary="local",
    )

    assert search_state["score_boundary"] == "local"
    assert (False, False) not in model.use_cache_seen


@pytest.mark.parametrize(
    ("method", "score_boundary", "match"),
    [
        ("gradient", "unsupported", "score_boundary must be one of"),
        ("kl_div", "group", "supported only with method='gradient'"),
    ],
)
def test_auto_quantize_validates_score_boundary_before_mutating_model(
    method, score_boundary, match
):
    model = TransformerBlock()
    original_mlp_type = type(model.mlp)

    with pytest.raises(ValueError, match=match):
        mtq.auto_quantize(model, method=method, score_boundary=score_boundary)

    assert type(model.mlp) is original_mlp_type
    assert not hasattr(model.mlp, "weight_quantizer")


def test_auto_quantize_can_retry_after_rejected_score_boundary():
    model = TransformerBlock()
    with pytest.raises(ValueError, match="supported only with method='gradient'"):
        mtq.auto_quantize(model, method="kl_div", score_boundary="group")

    _, search_state = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 8.0},
        quantization_formats=[
            mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
            mtq.INT8_WEIGHT_ONLY_CFG,
        ],
        data_loader=[model.get_input()],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=1,
        num_score_steps=1,
        method="gradient",
        score_boundary="local",
    )

    assert search_state["score_boundary"] == "local"


# use this config to test custom quantization config
INT8_CUSTOM_QUANT_TEST_CFG = {
    "quant_cfg": [
        *_base_disable_all,
        {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}},
        {"quantizer_name": "*input_quantizer", "cfg": {"num_bits": 8, "axis": None}},
        *_default_disabled_quantizer_cfg,
    ],
    "algorithm": "smoothquant",
}


@pytest.mark.parametrize(
    "model_cls",
    [SimpleConv, SimpleConvLinear, SimpleLinear, TransformerBlock],
)
@pytest.mark.parametrize(
    ("search_formats", "min_bits", "search_bits"),
    [
        ([mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG], 4.0, 6.0),
        ([mtq.INT4_AWQ_CFG, mtq.INT8_SMOOTHQUANT_CFG], 4.0, 6.0),
        ([mtq.INT4_AWQ_CFG, INT8_CUSTOM_QUANT_TEST_CFG], 4.0, 6.0),
        ([mtq.INT8_SMOOTHQUANT_CFG], 8.0, 11.0),
        ([None, mtq.INT8_SMOOTHQUANT_CFG], 8.0, 11.0),
    ],
)
@pytest.mark.parametrize(
    "method",
    ["gradient", "hidden_recon", "kl_div"],
)
def test_auto_quantize(model_cls, search_formats, min_bits, search_bits, method):
    model = model_cls()

    def loss_func(output):
        return output.sum()

    best_model, search_history = mtq.auto_quantize(
        model,
        constraints={"effective_bits": search_bits},
        quantization_formats=search_formats,
        data_loader=[model.get_input() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=2,
        num_score_steps=2,
        verbose=True,
        method=method,
    )
    assert isinstance(search_history, dict)
    assert search_history["best"]["is_satisfied"]
    effective_bits_from_search = search_history["best"]["constraints"]["effective_bits"]
    assert effective_bits_from_search <= search_bits and effective_bits_from_search >= min_bits, (
        "Search failed!"
    )
    for candidate_stat in search_history["candidate_stats"].values():
        assert len(candidate_stat["raw_scores"]) == len(candidate_stat["scores"])
        expected_scores = []
        previous_score = float("inf")
        for raw_score in candidate_stat["raw_scores"]:
            previous_score = min(raw_score, previous_score)
            expected_scores.append(previous_score)
        assert candidate_stat["scores"] == expected_scores

    if model_cls == TransformerBlock:
        hparam = model.attn.q_proj.get_hparam("quant_recipe")
        for layer in [model.attn.k_proj, model.attn.v_proj]:
            assert layer.get_hparam("quant_recipe") == hparam
        assert ("attn.q_proj.quant_recipe" in search_history["candidate_stats"]) != (
            "attn.k_proj.quant_recipe" in search_history["candidate_stats"]
        )

    # test restore
    buffer = io.BytesIO()
    mto.save(best_model, buffer)
    buffer.seek(0)
    new_model = model_cls()
    new_model = mto.restore(new_model, buffer)

    input = model.get_input()
    output_ref = best_model(input)
    output_test = new_model(input)
    assert torch.allclose(output_ref, output_test)


@pytest.mark.parametrize("method", ["gradient", "hidden_recon", "kl_div"])
def test_auto_quantize_two_stage_selects_and_records_held_out_kl(method):
    model = TransformerBlock()
    first_stage_data = [model.get_input()]
    held_out_data = [
        {
            "inputs": model.get_input(),
            "attention_mask": torch.tensor([[1, 1, 0, 0]]),
        }
    ]

    _, search_history = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 8.0},
        quantization_formats=[
            mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
            mtq.INT8_DEFAULT_CFG,
        ],
        data_loader=first_stage_data,
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=1,
        num_score_steps=1,
        num_candidates=3,
        candidate_selection_data_loader=held_out_data,
        candidate_selection_forward_step=lambda model, batch: model(batch["inputs"]),
        num_candidate_selection_steps=1,
        method=method,
    )

    candidates = search_history["best"]["candidates"]
    selection = search_history["best"]["candidate_selection"]
    assert len(candidates) == 3
    assert {
        key: selection[key]
        for key in (
            "metric",
            "requested_candidates",
            "evaluated_candidates",
            "requested_steps",
            "evaluated_steps",
            "requires_disjoint_samples",
        )
    } == {
        "metric": "kl_div",
        "requested_candidates": 3,
        "evaluated_candidates": 3,
        "requested_steps": 1,
        "evaluated_steps": 1,
        "requires_disjoint_samples": True,
    }
    assert selection["requested_pool_size"] == 3
    assert selection["enumerated_pool_size"] == 3
    assert selection["min_hamming_distance"] == 1
    assert selection["fallback_used"] is False
    json.dumps(selection)
    assert [candidate["selection_rank"] for candidate in candidates] == [0, 1, 2]
    assert candidates[0]["selected"]
    assert not any(candidate["selected"] for candidate in candidates[1:])
    assert candidates[0]["validation_kl_div"] == min(
        candidate["validation_kl_div"] for candidate in candidates
    )
    assert selection["reported_metrics"] == ["kl_div", "ear"]
    assert selection["ear_definition"] == "full_vocabulary_probability_overlap"
    assert all(0.0 <= candidate["validation_ear"] <= 1.0 for candidate in candidates)
    assert all(
        candidate["validation_ear_sum"]
        == pytest.approx(candidate["validation_ear"] * candidate["validation_token_count"])
        for candidate in candidates
    )
    assert candidates[0]["validation_token_count"] == 2
    assert search_history["best"]["recipe"] == candidates[0]["recipe"]


def test_auto_quantize_two_stage_can_select_maximum_ear():
    model = TransformerBlock()
    first_stage_data = [model.get_input()]
    held_out_data = [{"inputs": model.get_input(), "attention_mask": torch.ones((1, 4))}]

    _, search_history = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 8.0},
        quantization_formats=[
            mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
            mtq.INT8_DEFAULT_CFG,
        ],
        data_loader=first_stage_data,
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=1,
        num_score_steps=1,
        num_candidates=3,
        candidate_selection_data_loader=held_out_data,
        candidate_selection_forward_step=lambda model, batch: model(batch["inputs"]),
        candidate_selection_metric="ear",
        num_candidate_selection_steps=1,
    )

    candidates = search_history["best"]["candidates"]
    assert search_history["best"]["candidate_selection"]["metric"] == "ear"
    assert candidates[0]["validation_ear"] == max(
        candidate["validation_ear"] for candidate in candidates
    )
    assert search_history["best"]["recipe"] == candidates[0]["recipe"]


@pytest.mark.parametrize("selection_metric", ["kl_div", "ear"])
def test_auto_quantize_two_stage_metric_ties_preserve_first_stage_order(selection_metric):
    searcher = AutoQuantizeGradientSearcher()
    searcher.model = torch.nn.Identity()
    logits = torch.tensor([[[1.0, 2.0]]])
    searcher.config = {
        "candidate_selection_data_loader": [{"logits": logits}],
        "candidate_selection_forward_step": lambda model, batch: model(batch["logits"]),
        "candidate_selection_metric": selection_metric,
        "num_candidate_selection_steps": 1,
    }
    packets = [
        {"recipe_info": {}, "first_stage_rank": first_stage_rank} for first_stage_rank in (2, 0, 1)
    ]

    _, candidates, _ = searcher._select_candidate_with_kl_div(packets)

    assert [candidate["first_stage_rank"] for candidate in candidates] == [0, 1, 2]


@pytest.mark.parametrize("metric", ["kl_div", "ear"])
@pytest.mark.parametrize("invalid", ["candidate", "reference", "all", "padding"])
def test_auto_quantize_two_stage_nonfinite_metrics(metric, invalid):
    searcher = AutoQuantizeGradientSearcher()
    searcher.model = torch.nn.Identity()
    logits = torch.tensor([[[1.0, 2.0]]])
    bad_logits = torch.full_like(logits, torch.nan)
    data = logits
    if invalid == "padding":
        logits = bad_logits = torch.cat([logits, bad_logits], dim=1)
        data = {"attention_mask": torch.tensor([[1, 0]])}
    outputs = iter(
        [
            bad_logits if invalid == "reference" else logits,
            bad_logits,
            bad_logits if invalid == "all" else logits,
        ]
    )
    searcher.config = {
        "candidate_selection_data_loader": [data],
        "candidate_selection_forward_step": lambda model, batch: next(outputs),
        "candidate_selection_metric": metric,
        "num_candidate_selection_steps": 1,
    }
    packets = [{"recipe_info": {}, "first_stage_rank": rank} for rank in (0, 1)]

    if invalid in ("candidate", "padding"):
        _, candidates, _ = searcher._select_candidate_with_kl_div(packets)
        assert candidates[0]["first_stage_rank"] == int(invalid == "candidate")
        assert candidates[0]["validation_kl_div"] == pytest.approx(0)
        assert candidates[0]["validation_ear"] == pytest.approx(1)
    else:
        with pytest.raises(ValueError, match=r"reference|No candidate"):
            searcher._select_candidate_with_kl_div(packets)


def _check_distributed_autoquant_scores_and_selection(rank, world_size):
    torch.set_num_threads(1)
    model = _GroupBoundaryModel()
    torch.manual_seed(rank + 10)
    _, state = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 8.0},
        quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG],
        data_loader=[model.get_input()],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, batch: output.sum(),
        num_calib_steps=1,
        num_score_steps=1,
        num_candidates=2,
        candidate_selection_data_loader=[model.get_input()[:, : 2 * rank + 1]],
        candidate_selection_forward_step=lambda model, batch: model(batch),
        num_candidate_selection_steps=1,
    )
    gathered = [None] * world_size
    torch.distributed.all_gather_object(gathered, state["candidate_stats"])
    grouped_scores_agree = all(stats == gathered[0] for stats in gathered)
    assert all(
        candidate["validation_token_count"] == 4 for candidate in state["best"]["candidates"]
    )

    reference = torch.tensor([[[2.0, 0.0]]]).repeat(1, 1 if rank == 0 else 3, 1)
    flipped = reference.flip(-1)
    bad_kl = _get_kl_div_per_token(reference.log_softmax(-1), flipped)[0, 0].item()
    bad_ear = _get_expected_acceptance_rate_from_log_probs(
        reference.log_softmax(-1), flipped.log_softmax(-1)
    )[0, 0].item()
    for metric in ("kl_div", "ear"):
        for invalid in (None, "candidate", "reference", "all"):
            searcher = AutoQuantizeGradientSearcher()
            searcher.model = torch.nn.Identity()
            searcher.model.output_layer = torch.nn.Identity()
            searcher.model.output_layer.parallel_state = ParallelState()
            bad = torch.full_like(reference, torch.nan)
            outputs = iter(
                [
                    bad if invalid == "reference" and rank == 1 else reference,
                    bad
                    if invalid in ("candidate", "all") and rank == 1
                    else (reference if rank == 0 else flipped),
                    bad
                    if invalid == "all" and rank == 0
                    else (flipped if rank == 0 else reference),
                ]
            )
            searcher.config = {
                "candidate_selection_data_loader": [reference],
                "candidate_selection_forward_step": lambda model, batch: next(outputs),
                "candidate_selection_metric": metric,
                "num_candidate_selection_steps": 1,
            }
            packets = [{"recipe_info": {}, "first_stage_rank": i} for i in (0, 1)]
            if invalid in ("reference", "all"):
                with pytest.raises(ValueError, match=r"reference|No candidate"):
                    searcher._select_candidate_with_kl_div(packets)
                continue
            _, candidates, _ = searcher._select_candidate_with_kl_div(packets)
            winner = candidates[0]
            assert winner["first_stage_rank"] == 1
            assert winner["validation_token_count"] == 4
            assert winner["validation_kl_div"] == pytest.approx(bad_kl / 4)
            assert winner["validation_ear"] == pytest.approx((3 + bad_ear) / 4)
    assert grouped_scores_agree


@pytest.mark.timeout(180)
def test_auto_quantize_distributed_group_scores_and_candidate_selection():
    pool = DistributedWorkerPool(world_size=2, backend="gloo")
    try:
        pool.run(_check_distributed_autoquant_scores_and_selection)
    finally:
        pool.shutdown()


def test_auto_quantize_two_stage_rejects_unknown_selection_metric():
    model = TransformerBlock()
    first_stage_data = [model.get_input()]

    with pytest.raises(ValueError, match="candidate_selection_metric"):
        mtq.auto_quantize(
            model,
            constraints={"effective_bits": 8.0},
            quantization_formats=[mtq.INT8_DEFAULT_CFG],
            data_loader=first_stage_data,
            forward_step=lambda model, batch: model(batch),
            loss_func=lambda output, data: output.sum(),
            num_calib_steps=1,
            num_score_steps=1,
            candidate_selection_metric="accuracy",
        )


def test_auto_quantize_two_stage_rejects_reused_first_stage_loader():
    model = TransformerBlock()
    data = [model.get_input()]

    with pytest.raises(ValueError, match="must be a distinct loader"):
        mtq.auto_quantize(
            model,
            constraints={"effective_bits": 8.0},
            quantization_formats=[
                mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
                mtq.INT8_DEFAULT_CFG,
            ],
            data_loader=data,
            forward_step=lambda model, batch: model(batch),
            loss_func=lambda output, batch: output.sum(),
            num_calib_steps=1,
            num_score_steps=1,
            num_candidates=2,
            candidate_selection_data_loader=data,
            candidate_selection_forward_step=lambda model, batch: model(batch),
            num_candidate_selection_steps=1,
        )


def test_auto_quantize_two_stage_rejects_holdout_without_valid_tokens():
    model = TransformerBlock()
    first_stage_data = [model.get_input()]
    held_out_data = [
        {
            "inputs": model.get_input(),
            "attention_mask": torch.zeros((1, 4), dtype=torch.long),
        }
    ]

    with pytest.raises(ValueError, match="no valid tokens"):
        mtq.auto_quantize(
            model,
            constraints={"effective_bits": 8.0},
            quantization_formats=[
                mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
                mtq.INT8_DEFAULT_CFG,
            ],
            data_loader=first_stage_data,
            forward_step=lambda model, batch: model(batch),
            loss_func=lambda output, data: output.sum(),
            num_calib_steps=1,
            num_score_steps=1,
            num_candidates=2,
            candidate_selection_data_loader=held_out_data,
            candidate_selection_forward_step=lambda model, batch: model(batch["inputs"]),
            num_candidate_selection_steps=1,
        )


def test_auto_quantize_disable_layers():
    model = TransformerBlock()

    def loss_func(output):
        return output.sum()

    best_model, search_history = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 8.0},
        quantization_formats=[
            mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
            mtq.INT8_DEFAULT_CFG,
        ],
        data_loader=[model.get_input() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        disabled_layers=["*mlp*"],
        num_calib_steps=2,
        num_score_steps=2,
        verbose=True,
    )

    assert not best_model.mlp.input_quantizer.is_enabled


def test_auto_quantize_disabled_layers_no_poison():
    """disabled_layers must only affect the matched layers, not all subsequent layer groups."""
    model = TransformerBlock()

    best_model, _ = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 8.0},
        quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG],
        data_loader=[model.get_input() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        disabled_layers=["*mlp*"],
        num_calib_steps=2,
        num_score_steps=2,
    )

    assert not best_model.mlp.input_quantizer.is_enabled
    hparam = best_model.attn.q_proj.get_hparam("quant_recipe")
    assert QuantRecipe(mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG) in hparam.choices


INT4INT8_AWQ_CFG = {
    "quant_cfg": [
        {"quantizer_name": "*", "enable": False},
        {
            "quantizer_name": "*weight_quantizer",
            "cfg": [
                {"num_bits": 4, "block_sizes": {-1: 128, "type": "static"}},
                {"num_bits": 8, "axis": None},
            ],
            "enable": True,
        },
        {
            "quantizer_name": "*input_quantizer",
            "cfg": {"num_bits": 8, "axis": None},
            "enable": True,
        },
    ],
    "algorithm": "awq_lite",
}


@pytest.mark.parametrize("config", [mtq.INT4_AWQ_CFG, mtq.INT8_SMOOTHQUANT_CFG, INT4INT8_AWQ_CFG])
def test_pqs_folding(config):
    model_ref = SimpleLinear()
    state_dict_ref = copy.deepcopy(model_ref.state_dict())
    inputs = model_ref.get_input()
    mtq.quantize(model_ref, config, lambda model: model(inputs))

    model_test = SimpleLinear()
    model_test.load_state_dict(state_dict_ref)
    QuantRecipe.disable_folding_pqs_to_weights()
    mtq.quantize(model_test, config, lambda model: model(inputs))

    assert torch.allclose(model_ref(inputs), model_test(inputs))

    QuantRecipe.fold_pqs_to_weights(model_test)
    assert torch.allclose(model_ref(inputs), model_test(inputs))


def _test_data_parallel_auto_quantize(rank, size):
    model = SimpleLinear()

    model, search_history = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 11.0},
        quantization_formats=[mtq.INT8_SMOOTHQUANT_CFG],
        data_loader=[model.get_input() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=2,
        num_score_steps=2,
        verbose=True,
    )

    search_history_rank0 = DistributedProcessGroup.get_dist_syncd_obj(
        search_history if rank == 0 else None,
        DistributedProcessGroup(None),
        lambda a: a[0],
    )

    # quantizer_states contains tensors which can't be compared with ==
    sh = {k: v for k, v in search_history.items() if k != "quantizer_states"}
    sh0 = {k: v for k, v in search_history_rank0.items() if k != "quantizer_states"}

    # Assert that the costs, scores and searched recipes are the same across all ranks
    assert sh == sh0

    assert search_history["best"]["is_satisfied"]


def test_data_parallel_auto_quantize(skip_on_windows):
    # 2 ranks fully exercise the cross-rank sync the test asserts; more just adds spawn overhead.
    spawn_multiprocess_job(2, _test_data_parallel_auto_quantize, backend="gloo")


def test_auto_quantize_budget_uses_no_quant_candidate_cost(monkeypatch):
    class _BudgetCaptureSearcher(AutoQuantizeGradientSearcher):
        def run_search_with_stats(self, max_weight_size, verbose=False):
            self.max_weight_size = max_weight_size
            return {}, True

    def _raise_local_total_weight_size(modules):
        pytest.fail("run_search should derive total weight size from candidate costs")

    monkeypatch.setattr(
        _AutoQuantizeBaseSearcher,
        "_get_total_weight_size",
        staticmethod(_raise_local_total_weight_size),
    )

    searcher = _BudgetCaptureSearcher()
    searcher.reset_search()
    searcher.model = torch.nn.Module()
    searcher.config = {"verbose": False}
    searcher.constraints = {"effective_bits": 8.0}
    searcher.candidate_stats = {
        "local_expert.quant_recipe": {
            "formats": [QuantRecipe(mtq.NVFP4_DEFAULT_CFG), QuantRecipe(None)],
            "scores": [1.0, 0.0],
            "costs": [25.0, 100.0],
        }
    }

    searcher.run_search()

    assert searcher.max_weight_size == 50.0


def test_estimate_quant_compression():
    # NVFP4 weight/input carry effective_bits=4.5 from configs/numerics/nvfp4 -> 4.5/16 = 0.28125.
    nvfp4_affine_kv_cfg = mtq.config.QuantizeConfig(**mtq.NVFP4_AFFINE_KV_CFG)
    assert estimate_quant_compression(nvfp4_affine_kv_cfg) == 0.28125

    nvfp4_awq_clip_cfg = mtq.config.QuantizeConfig(**mtq.NVFP4_AWQ_CLIP_CFG)
    assert estimate_quant_compression(nvfp4_awq_clip_cfg) == 0.28125

    nvfp4_awq_full_cfg = mtq.config.QuantizeConfig(**mtq.NVFP4_AWQ_FULL_CFG)
    assert estimate_quant_compression(nvfp4_awq_full_cfg) == 0.28125

    nvfp4_awq_lite_cfg = mtq.config.QuantizeConfig(**mtq.NVFP4_AWQ_LITE_CFG)
    assert estimate_quant_compression(nvfp4_awq_lite_cfg) == 0.28125

    nvfp4_default_cfg = mtq.config.QuantizeConfig(**mtq.NVFP4_DEFAULT_CFG)
    assert estimate_quant_compression(nvfp4_default_cfg) == 0.28125

    nvfp4_kv_cfg = mtq.config.QuantizeConfig(**mtq.NVFP4_KV_CFG)
    assert estimate_quant_compression(nvfp4_kv_cfg) == 0.28125

    nvfp4_kv_rotate_cfg = mtq.config.QuantizeConfig(**mtq.NVFP4_KV_ROTATE_CFG)
    assert estimate_quant_compression(nvfp4_kv_rotate_cfg) == 0.28125

    nvfp4_svdquant_default_cfg = mtq.config.QuantizeConfig(**mtq.NVFP4_SVDQUANT_DEFAULT_CFG)
    assert estimate_quant_compression(nvfp4_svdquant_default_cfg) == 0.28125

    int8_default_cfg = mtq.config.QuantizeConfig(**mtq.INT8_DEFAULT_CFG)
    assert estimate_quant_compression(int8_default_cfg) == 0.5

    int8_smoothquant_cfg = mtq.config.QuantizeConfig(**mtq.INT8_SMOOTHQUANT_CFG)
    assert estimate_quant_compression(int8_smoothquant_cfg) == 0.5

    fp8_default_cfg = mtq.config.QuantizeConfig(**mtq.FP8_DEFAULT_CFG)
    assert estimate_quant_compression(fp8_default_cfg) == 0.5

    fp8_per_channel_per_token_cfg = mtq.config.QuantizeConfig(**mtq.FP8_PER_CHANNEL_PER_TOKEN_CFG)
    assert estimate_quant_compression(fp8_per_channel_per_token_cfg) == 0.5

    fp8_2d_blockwise_weight_only_cfg = mtq.config.QuantizeConfig(
        **mtq.FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG
    )
    assert estimate_quant_compression(fp8_2d_blockwise_weight_only_cfg) == 0.5

    int4_blockwise_weight_only_cfg = mtq.config.QuantizeConfig(**mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG)
    assert estimate_quant_compression(int4_blockwise_weight_only_cfg) == 0.25

    int4_awq_cfg = mtq.config.QuantizeConfig(**mtq.INT4_AWQ_CFG)
    assert estimate_quant_compression(int4_awq_cfg) == 0.25

    w4a8_awq_beta_cfg = mtq.config.QuantizeConfig(**mtq.W4A8_AWQ_BETA_CFG)
    assert estimate_quant_compression(w4a8_awq_beta_cfg) == 0.25

    mxfp8_default_cfg = mtq.config.QuantizeConfig(**mtq.MXFP8_DEFAULT_CFG)
    assert estimate_quant_compression(mxfp8_default_cfg) == 0.5

    mxfp6_default_cfg = mtq.config.QuantizeConfig(**mtq.MXFP6_DEFAULT_CFG)
    assert estimate_quant_compression(mxfp6_default_cfg) == 0.375

    mxfp4_default_cfg = mtq.config.QuantizeConfig(**mtq.MXFP4_DEFAULT_CFG)
    assert estimate_quant_compression(mxfp4_default_cfg) == 0.25

    mxint8_default_cfg = mtq.config.QuantizeConfig(**mtq.MXINT8_DEFAULT_CFG)
    assert estimate_quant_compression(mxint8_default_cfg) == 0.5

    fp8_kv_cfg = mtq.config.QuantizeConfig(**mtq.FP8_KV_CFG)
    assert estimate_quant_compression(fp8_kv_cfg) == 0.5

    fp8_affine_kv_cfg = mtq.config.QuantizeConfig(**mtq.FP8_AFFINE_KV_CFG)
    assert estimate_quant_compression(fp8_affine_kv_cfg) == 0.5


def test_estimate_quant_compression_effective_bits_override():
    """Recipe-level ``QuantizeConfig.effective_bits`` overrides the per-entry library default."""
    # NVFP4 weight carries effective_bits=4.5 from configs/numerics/nvfp4 (per-entry default).
    nvfp4_cfg = mtq.config.QuantizeConfig(**mtq.NVFP4_DEFAULT_CFG)
    assert nvfp4_cfg.effective_bits is None  # no recipe-level override
    assert estimate_quant_compression(nvfp4_cfg) == 4.5 / 16.0  # per-entry library default

    # A recipe-level QuantizeConfig.effective_bits override wins over the per-entry default.
    nvfp4_cfg_overridden = mtq.config.QuantizeConfig(**mtq.NVFP4_DEFAULT_CFG, effective_bits=8.0)
    assert estimate_quant_compression(nvfp4_cfg_overridden) == 8.0 / 16.0

    # Override can also represent a higher cost (e.g., conservative for a sensitive recipe).
    nvfp4_cfg_high = mtq.config.QuantizeConfig(**mtq.NVFP4_DEFAULT_CFG, effective_bits=16.0)
    assert estimate_quant_compression(nvfp4_cfg_high) == 1.0

    # Out-of-range values are rejected by the Pydantic validator.
    with pytest.raises(ValueError, match="effective_bits must be in"):
        mtq.config.QuantizeConfig(**mtq.NVFP4_DEFAULT_CFG, effective_bits=0.0)
    with pytest.raises(ValueError, match="effective_bits must be in"):
        mtq.config.QuantizeConfig(**mtq.NVFP4_DEFAULT_CFG, effective_bits=17.0)


def test_estimate_quant_compression_per_entry_effective_bits():
    """Per-entry ``effective_bits`` overrides the heuristic; recipe-level wins over it; min across entries."""
    # num_bits=(2,1) -> heuristic 0.25, but per-entry library default is 4.5.
    cfg = mtq.config.QuantizeConfig(
        quant_cfg=[
            {
                "quantizer_name": "*weight_quantizer",
                "cfg": {"num_bits": (2, 1), "effective_bits": 4.5},
            },
        ],
        algorithm="max",
    )
    assert cfg.effective_bits is None
    assert estimate_quant_compression(cfg) == 4.5 / 16.0

    # Recipe-level override (layer 1) wins over the per-entry value (layer 2).
    cfg_recipe_override = mtq.config.QuantizeConfig(
        quant_cfg=[
            {
                "quantizer_name": "*weight_quantizer",
                "cfg": {"num_bits": (2, 1), "effective_bits": 4.5},
            },
        ],
        algorithm="max",
        effective_bits=8.0,
    )
    assert estimate_quant_compression(cfg_recipe_override) == 8.0 / 16.0

    # min across entries: weight 4.5/16 = 0.28125 vs heuristic fp8 input 0.5 -> 0.28125.
    cfg_mixed = mtq.config.QuantizeConfig(
        quant_cfg=[
            {
                "quantizer_name": "*weight_quantizer",
                "cfg": {"num_bits": (2, 1), "effective_bits": 4.5},
            },
            {"quantizer_name": "*input_quantizer", "cfg": {"num_bits": (4, 3)}},
        ],
        algorithm="max",
    )
    assert estimate_quant_compression(cfg_mixed) == 4.5 / 16.0

    # Per-entry out-of-range is rejected by the QuantizerAttributeConfig validator.
    with pytest.raises(ValueError, match="effective_bits must be in"):
        mtq.config.QuantizeConfig(
            quant_cfg=[
                {
                    "quantizer_name": "*weight_quantizer",
                    "cfg": {"num_bits": (2, 1), "effective_bits": 20.0},
                },
            ],
            algorithm="max",
        )


@pytest.mark.parametrize("method", ["gradient", "hidden_recon", "kl_div"])
def test_auto_quantize_checkpoint_resume(method, tmp_path, capsys):
    """Test that checkpoint can be used to resume an interrupted search."""
    model = SimpleLinear()
    checkpoint_path = str(tmp_path / "autoquant_resume_checkpoint.pth")

    # First run: save checkpoint
    model_1, state_dict_1 = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 6.0},
        quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG],
        data_loader=[model.get_input() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=2,
        num_score_steps=2,
        verbose=True,
        method=method,
        checkpoint=checkpoint_path,
    )

    # Clear captured output from first run
    capsys.readouterr()

    # Second run: resume with same constraint should produce same results
    model_2 = SimpleLinear()
    model_2, state_dict_2 = mtq.auto_quantize(
        model_2,
        constraints={"effective_bits": 6.0},  # Same constraint
        quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG],
        data_loader=[model_2.get_input() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=2,
        num_score_steps=2,
        verbose=True,
        method=method,
        checkpoint=checkpoint_path,
    )

    # Verify the restore message was printed on second run
    captured = capsys.readouterr()
    assert "Restored from checkpoint, skipping scoring" in captured.out, (
        "Expected restore message when resuming from checkpoint"
    )

    # Verify method is correctly persisted in checkpoint and state dicts
    saved = safe_load(checkpoint_path)
    assert saved["method"] == method
    assert state_dict_1["method"] == method
    assert state_dict_2["method"] == method

    # Results should be identical when using same constraint
    assert state_dict_1["candidate_stats"] == state_dict_2["candidate_stats"]
    assert state_dict_1["best"]["recipe"] == state_dict_2["best"]["recipe"]
    assert (
        pytest.approx(state_dict_1["best"]["constraints"]["effective_bits"])
        == state_dict_2["best"]["constraints"]["effective_bits"]
    )

    # Verify calibration was also restored from checkpoint
    assert "Restored calibration for" in captured.out

    # Verify quantizer_states is saved in checkpoint
    assert "quantizer_states" in saved
    assert len(saved["quantizer_states"]) > 0
    for recipe_state in saved["quantizer_states"].values():
        assert "metadata" in recipe_state
        assert "state_dict" in recipe_state

    # Verify resumed model produces identical quantizer_states
    assert state_dict_1["quantizer_states"].keys() == state_dict_2["quantizer_states"].keys()
    for recipe in state_dict_1["quantizer_states"]:
        s1 = state_dict_1["quantizer_states"][recipe]
        s2 = state_dict_2["quantizer_states"][recipe]
        # Verify metadata (quantizer properties + tensor shape/dtype info) match per quantizer
        assert s1["metadata"].keys() == s2["metadata"].keys()
        for qname in s1["metadata"]:
            assert s1["metadata"][qname] == s2["metadata"][qname], (
                f"Metadata mismatch for {qname} in {recipe}"
            )
        # Verify actual tensor values match per quantizer
        assert s1["state_dict"].keys() == s2["state_dict"].keys()
        for qname in s1["state_dict"]:
            for buf_name in s1["state_dict"][qname]:
                torch.testing.assert_close(
                    s1["state_dict"][qname][buf_name], s2["state_dict"][qname][buf_name]
                )


@pytest.mark.parametrize("method", ["gradient", "hidden_recon", "kl_div"])
def test_auto_quantize_one_stage_resume_clears_candidate_selection(tmp_path, method):
    checkpoint = str(tmp_path / "two_stage.pth")
    template = TransformerBlock()
    inputs = template.get_input()

    def run(bits, candidates):
        return mtq.auto_quantize(
            copy.deepcopy(template),
            constraints={"effective_bits": bits},
            quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG],
            data_loader=[inputs],
            forward_step=lambda model, batch: model(batch),
            loss_func=(lambda output, batch: output.sum()) if method == "gradient" else None,
            num_calib_steps=1,
            num_score_steps=1,
            method=method,
            num_candidates=candidates,
            candidate_selection_data_loader=[inputs + 0.25] if candidates > 1 else None,
            candidate_selection_forward_step=(lambda model, batch: model(batch))
            if candidates > 1
            else None,
            num_candidate_selection_steps=1,
            checkpoint=checkpoint,
        )[1]

    original = run(16.0, 2)["best"]
    assert original["candidates"][0]["selected"]
    resumed = run(8.0, 1)["best"]
    assert resumed["recipe"] != original["recipe"]
    assert resumed["constraints"]["effective_bits"] <= 8.0
    assert resumed["candidates"] == []
    assert resumed["candidate_selection"] is None
    assert safe_load(checkpoint)["best"]["candidates"] == []
    assert safe_load(checkpoint)["best"]["candidate_selection"] is None


def test_auto_quantize_checkpoint_rejects_changed_fixed_ptq_baseline(tmp_path):
    checkpoint_path = str(tmp_path / "autoquant_fixed_baseline_checkpoint.pth")

    def run(fixed_config):
        model = TransformerBlock()
        return mtq.auto_quantize(
            model,
            constraints={"effective_bits": 8.0},
            fixed_quantization_config=fixed_config,
            module_search_spaces=[
                {
                    "module_name_patterns": ["*mlp*"],
                    "quantization_formats": [
                        mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
                        mtq.INT8_DEFAULT_CFG,
                    ],
                }
            ],
            data_loader=[model.get_input()],
            forward_step=lambda model, batch: model(batch),
            loss_func=lambda output, data: output.sum(),
            num_calib_steps=1,
            num_score_steps=1,
            checkpoint=checkpoint_path,
        )

    run(mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG)
    with pytest.raises(ValueError, match="fixed_quantization_config does not match"):
        run(mtq.FP8_DEFAULT_CFG)


def test_auto_quantize_calibration_only_checkpoint_validates_module_search_spaces(
    tmp_path, monkeypatch
):
    checkpoint_path = str(tmp_path / "autoquant_calibration_only_checkpoint.pth")
    original_estimate_scores = AutoQuantizeGradientSearcher.estimate_sensitivity_scores

    def interrupt_after_calibration(self):
        raise RuntimeError("interrupt after calibration")

    monkeypatch.setattr(
        AutoQuantizeGradientSearcher,
        "estimate_sensitivity_scores",
        interrupt_after_calibration,
    )
    model = TransformerBlock()
    with pytest.raises(RuntimeError, match="interrupt after calibration"):
        mtq.auto_quantize(
            model,
            constraints={"effective_bits": 6.0},
            quantization_formats=[
                mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
                mtq.INT8_DEFAULT_CFG,
            ],
            module_search_spaces=[
                {
                    "module_name_patterns": ["*mlp*"],
                    "quantization_formats": [mtq.FP8_DEFAULT_CFG],
                    "allow_no_quant": False,
                }
            ],
            data_loader=[model.get_input()],
            forward_step=lambda model, batch: model(batch),
            loss_func=lambda output, data: output.sum(),
            num_calib_steps=1,
            num_score_steps=1,
            checkpoint=checkpoint_path,
        )

    saved = safe_load(checkpoint_path)
    assert saved["quantizer_states"]
    assert not saved["candidate_stats"]

    monkeypatch.setattr(
        AutoQuantizeGradientSearcher,
        "estimate_sensitivity_scores",
        original_estimate_scores,
    )
    resumed_model = TransformerBlock()
    with pytest.raises(ValueError, match="module_search_spaces do not match"):
        mtq.auto_quantize(
            resumed_model,
            constraints={"effective_bits": 6.0},
            quantization_formats=[
                mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
                mtq.INT8_DEFAULT_CFG,
            ],
            module_search_spaces=[
                {
                    "module_name_patterns": ["*mlp*"],
                    "quantization_formats": [mtq.INT8_DEFAULT_CFG],
                    "allow_no_quant": False,
                }
            ],
            data_loader=[resumed_model.get_input()],
            forward_step=lambda model, batch: model(batch),
            loss_func=lambda output, data: output.sum(),
            num_calib_steps=1,
            num_score_steps=1,
            checkpoint=checkpoint_path,
        )


@pytest.mark.parametrize(
    ("initial_disabled", "initial_excluded", "resumed_disabled", "resumed_excluded"),
    [
        (["*mlp*"], None, ["*o_proj*"], None),
        (None, ["*mlp*"], None, ["*o_proj*"]),
    ],
    ids=["disabled-layers", "cost-exclusions"],
)
def test_auto_quantize_calibration_checkpoint_validates_resolved_search_setup(
    tmp_path,
    monkeypatch,
    initial_disabled,
    initial_excluded,
    resumed_disabled,
    resumed_excluded,
):
    checkpoint_path = str(tmp_path / "autoquant_calibration_only_checkpoint.pth")
    original_estimate_scores = AutoQuantizeGradientSearcher.estimate_sensitivity_scores

    def interrupt_after_calibration(self):
        raise RuntimeError("interrupt after calibration")

    def constraints(excluded_patterns):
        result = {"effective_bits": 8.0}
        if excluded_patterns is not None:
            result["cost"] = {EXCLUDED_MODULE_NAME_PATTERNS_KEY: excluded_patterns}
        return result

    monkeypatch.setattr(
        AutoQuantizeGradientSearcher,
        "estimate_sensitivity_scores",
        interrupt_after_calibration,
    )
    model = TransformerBlock()
    with pytest.raises(RuntimeError, match="interrupt after calibration"):
        mtq.auto_quantize(
            model,
            constraints=constraints(initial_excluded),
            quantization_formats=[
                mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
                mtq.INT8_DEFAULT_CFG,
            ],
            disabled_layers=initial_disabled,
            data_loader=[model.get_input()],
            forward_step=lambda model, batch: model(batch),
            loss_func=lambda output, data: output.sum(),
            num_calib_steps=1,
            num_score_steps=1,
            checkpoint=checkpoint_path,
        )

    saved = safe_load(checkpoint_path)
    assert saved["quantizer_states"]
    assert saved["resolved_search_setup_signature"]
    assert not saved["candidate_stats"]

    monkeypatch.setattr(
        AutoQuantizeGradientSearcher,
        "estimate_sensitivity_scores",
        original_estimate_scores,
    )
    resumed_model = TransformerBlock()
    with pytest.raises(ValueError, match="resolved search setup does not match"):
        mtq.auto_quantize(
            resumed_model,
            constraints=constraints(resumed_excluded),
            quantization_formats=[
                mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
                mtq.INT8_DEFAULT_CFG,
            ],
            disabled_layers=resumed_disabled,
            data_loader=[resumed_model.get_input()],
            forward_step=lambda model, batch: model(batch),
            loss_func=lambda output, data: output.sum(),
            num_calib_steps=1,
            num_score_steps=1,
            checkpoint=checkpoint_path,
        )


def test_auto_quantize_calibration_checkpoint_resumes_with_cost_excluded_groups(
    tmp_path, monkeypatch
):
    checkpoint_path = str(tmp_path / "autoquant_cost_excluded_calibration.pth")
    original_estimate_scores = AutoQuantizeGradientSearcher.estimate_sensitivity_scores

    def interrupt_after_calibration(self):
        raise RuntimeError("interrupt after calibration")

    constraints = {
        "effective_bits": 8.0,
        "cost": {EXCLUDED_MODULE_NAME_PATTERNS_KEY: ["*attn*"]},
    }
    monkeypatch.setattr(
        AutoQuantizeGradientSearcher,
        "estimate_sensitivity_scores",
        interrupt_after_calibration,
    )
    model = TransformerBlock()
    with pytest.raises(RuntimeError, match="interrupt after calibration"):
        mtq.auto_quantize(
            model,
            constraints=constraints,
            quantization_formats=[
                mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
                mtq.INT8_DEFAULT_CFG,
            ],
            data_loader=[model.get_input()],
            forward_step=lambda model, batch: model(batch),
            loss_func=lambda output, data: output.sum(),
            num_calib_steps=1,
            num_score_steps=1,
            checkpoint=checkpoint_path,
        )

    saved = safe_load(checkpoint_path)
    assert type(saved["candidate_stats"]) is dict
    assert not saved["candidate_stats"]

    monkeypatch.setattr(
        AutoQuantizeGradientSearcher,
        "estimate_sensitivity_scores",
        original_estimate_scores,
    )
    resumed_model = TransformerBlock()
    _, resumed = mtq.auto_quantize(
        resumed_model,
        constraints=constraints,
        quantization_formats=[
            mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
            mtq.INT8_DEFAULT_CFG,
        ],
        data_loader=[resumed_model.get_input()],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=1,
        num_score_steps=1,
        checkpoint=checkpoint_path,
    )

    excluded_stats = [
        stats
        for stats in resumed["candidate_stats"].values()
        if all("attn" in module_name for module_name in stats["module_names"])
    ]
    assert excluded_stats
    assert all(stats["cost_weight"] == 0.0 for stats in excluded_stats)
    assert all(stats["uncompressed_cost"] == 0.0 for stats in excluded_stats)
    assert set(resumed["candidate_stats"]) == set(resumed["best"]["recipe"])


def test_auto_quantize_calibration_only_checkpoint_validates_global_formats_and_legacy(
    tmp_path, monkeypatch
):
    checkpoint_path = str(tmp_path / "autoquant_calibration_only_checkpoint.pth")
    legacy_checkpoint_path = str(tmp_path / "autoquant_legacy_calibration_only_checkpoint.pth")
    original_estimate_scores = AutoQuantizeGradientSearcher.estimate_sensitivity_scores

    def interrupt_after_calibration(self):
        raise RuntimeError("interrupt after calibration")

    monkeypatch.setattr(
        AutoQuantizeGradientSearcher,
        "estimate_sensitivity_scores",
        interrupt_after_calibration,
    )
    model = TransformerBlock()
    with pytest.raises(RuntimeError, match="interrupt after calibration"):
        mtq.auto_quantize(
            model,
            constraints={"effective_bits": 6.0},
            quantization_formats=[
                mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
                mtq.INT8_DEFAULT_CFG,
            ],
            data_loader=[model.get_input()],
            forward_step=lambda model, batch: model(batch),
            loss_func=lambda output, data: output.sum(),
            num_calib_steps=1,
            num_score_steps=1,
            checkpoint=checkpoint_path,
        )

    saved = safe_load(checkpoint_path)
    assert saved["quantizer_states"]
    assert saved["quantization_formats_signature"]
    legacy_saved = dict(saved)
    legacy_saved.pop("quantization_formats_signature")
    safe_save(legacy_saved, legacy_checkpoint_path)

    monkeypatch.setattr(
        AutoQuantizeGradientSearcher,
        "estimate_sensitivity_scores",
        original_estimate_scores,
    )
    mismatched_model = TransformerBlock()
    with pytest.raises(ValueError, match="quantization_formats do not match"):
        mtq.auto_quantize(
            mismatched_model,
            constraints={"effective_bits": 6.0},
            quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.FP8_DEFAULT_CFG],
            data_loader=[mismatched_model.get_input()],
            forward_step=lambda model, batch: model(batch),
            loss_func=lambda output, data: output.sum(),
            num_calib_steps=1,
            num_score_steps=1,
            checkpoint=checkpoint_path,
        )

    legacy_model = TransformerBlock()
    with pytest.raises(ValueError, match="does not record its quantization_formats signature"):
        mtq.auto_quantize(
            legacy_model,
            constraints={"effective_bits": 6.0},
            quantization_formats=[
                mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
                mtq.INT8_DEFAULT_CFG,
            ],
            data_loader=[legacy_model.get_input()],
            forward_step=lambda model, batch: model(batch),
            loss_func=lambda output, data: output.sum(),
            num_calib_steps=1,
            num_score_steps=1,
            checkpoint=legacy_checkpoint_path,
        )


def test_auto_quantize_checkpoint_rejects_score_boundary_mismatch(tmp_path):
    checkpoint_path = str(tmp_path / "autoquant_group_boundary.pth")
    model = _GroupBoundaryModel()
    common = {
        "constraints": {"effective_bits": 8.0},
        "quantization_formats": [
            mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
            mtq.INT8_DEFAULT_CFG,
        ],
        "forward_step": lambda model, batch: model(batch),
        "loss_func": lambda output, data: output.sum(),
        "num_calib_steps": 1,
        "num_score_steps": 1,
        "checkpoint": checkpoint_path,
    }
    mtq.auto_quantize(
        model,
        data_loader=[model.get_input()],
        score_boundary="group",
        **common,
    )

    resumed_model = _GroupBoundaryModel()
    with pytest.raises(ValueError, match="score boundary does not match"):
        mtq.auto_quantize(
            resumed_model,
            data_loader=[resumed_model.get_input()],
            score_boundary="local",
            **common,
        )


@pytest.mark.parametrize(
    ("saved_fallback", "current_fallback", "legacy"),
    [
        (None, None, True),
        (None, -30, True),
        (-30, -60, False),
        (-30, None, False),
        (-30, -30, False),
    ],
)
def test_auto_quantize_checkpoint_validates_kl_fallback(
    tmp_path, saved_fallback, current_fallback, legacy
):
    checkpoint = str(tmp_path / "kl_fallback.pth")

    def run(fallback):
        model = SimpleLinear()
        return mtq.auto_quantize(
            model,
            constraints={"effective_bits": 6.0},
            quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG],
            data_loader=[model.get_input()],
            forward_step=lambda model, batch: model(batch),
            num_calib_steps=1,
            num_score_steps=1,
            method="kl_div",
            kl_log_prob_fallback=fallback,
            checkpoint=checkpoint,
        )[1]

    original = run(saved_fallback)
    saved = safe_load(checkpoint)
    assert saved["kl_log_prob_fallback"] == saved_fallback
    if legacy:
        del saved["kl_log_prob_fallback"]
        safe_save(saved, checkpoint)
    if saved_fallback == current_fallback:
        resumed = run(current_fallback)
        assert resumed["candidate_stats"] == original["candidate_stats"]
    else:
        with pytest.raises(ValueError, match="KL log-probability fallback"):
            run(current_fallback)


@pytest.mark.parametrize("method", ["gradient", "hidden_recon", "kl_div"])
def test_get_auto_quantize_config(method):
    model = TransformerBlock()

    _, search_state = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 6.0},
        quantization_formats=[mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_DEFAULT_CFG],
        data_loader=[model.get_input() for _ in range(4)],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=2,
        num_score_steps=2,
        method=method,
    )

    # Verify search_state has method and module_names
    assert search_state["method"] == method
    assert search_state["score_boundary"] == ("local" if method == "kl_div" else "group")
    for stats in search_state["candidate_stats"].values():
        assert "module_names" in stats
        assert len(stats["module_names"]) > 0

    # Use stored best recipe
    config = mtq.get_auto_quantize_config(search_state)
    assert "quant_cfg" in config
    assert isinstance(config["quant_cfg"], list)
    assert any(
        entry["quantizer_name"] == "*" and entry.get("enable") is False
        for entry in config["quant_cfg"]
    )
    assert config["algorithm"] == "max"

    # Re-solve with different constraints
    config_resoled = mtq.get_auto_quantize_config(
        search_state, constraints={"effective_bits": 12.0}
    )
    assert "quant_cfg" in config_resoled

    # Apply config to a fresh model
    fresh_model = TransformerBlock()
    fresh_model = mtq.quantize(fresh_model, config, forward_loop=lambda m: m(model.get_input()))
    output = fresh_model(model.get_input())
    assert output is not None


def test_get_auto_quantize_config_keeps_selected_lm_head_enabled():
    recipe_config = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    recipe_config["quant_cfg"].append({"quantizer_name": "*lm_head*", "enable": False})
    recipe = QuantRecipe(recipe_config, name="explicit_lm_head_disable")
    search_state = {
        "best": {"recipe": {"lm_head.quant_recipe": recipe}},
        "candidate_stats": {"lm_head.quant_recipe": {"module_names": ["lm_head"]}},
        "disabled_layers": ["*visual*", "*mtp*"],
    }

    config = mtq.get_auto_quantize_config(search_state)
    quant_cfg = config["quant_cfg"]
    quantizer_names = [entry["quantizer_name"] for entry in quant_cfg]

    default_disable_idx = next(
        idx for idx, entry in enumerate(quant_cfg) if entry["quantizer_name"] == "*lm_head*"
    )
    weight_idx = next(
        idx
        for idx, entry in enumerate(quant_cfg)
        if entry["quantizer_name"] == "lm_head.weight_quantizer"
    )
    weight_entry = quant_cfg[weight_idx]

    assert "*visual*" in quantizer_names
    assert "*mtp*" in quantizer_names
    assert default_disable_idx < weight_idx
    assert weight_entry["enable"] is True
    assert weight_entry["cfg"]["num_bits"] == (4, 3)


@pytest.mark.parametrize("with_persisted_attrs", [True, False])
def test_get_auto_quantize_config_emits_fused_expert_quantizer_names(with_persisted_attrs):
    recipe = QuantRecipe(copy.deepcopy(mtq.FP8_DEFAULT_CFG), name="fp8")
    module_name = "layers.0.mlp.experts"
    candidate_stat = {"module_names": [module_name]}
    if with_persisted_attrs:
        candidate_stat["quantizer_attrs"] = {
            module_name: [
                "gate_up_proj_input_quantizer",
                "gate_up_proj_weight_quantizer",
                "down_proj_input_quantizer",
                "down_proj_weight_quantizer",
            ]
        }
    search_state = {
        "best": {"recipe": {f"{module_name}.quant_recipe": recipe}},
        "candidate_stats": {f"{module_name}.quant_recipe": candidate_stat},
        "disabled_layers": [],
    }

    config = mtq.get_auto_quantize_config(search_state)
    quantizer_names = {entry["quantizer_name"] for entry in config["quant_cfg"]}

    assert f"{module_name}.gate_up_proj_weight_quantizer" in quantizer_names
    assert f"{module_name}.down_proj_weight_quantizer" in quantizer_names
    assert f"{module_name}.weight_quantizer" not in quantizer_names
