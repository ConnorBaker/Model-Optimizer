# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grouped native MTP experts preserve routing and clamped SwiGLU gradients."""

import copy

import pytest
import torch

from modelopt.torch.speculative.mtp.deepseek_v4 import NativeMTPBoostModel

hf_dsv4 = pytest.importorskip("transformers.models.deepseek_v4.modeling_deepseek_v4")


def test_native_mtp_preserves_fixed_router_state():
    """FSDP placement promotion must not turn routing-only buffers into optimizer inputs."""
    config = hf_dsv4.DeepseekV4Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=8,
        num_hidden_layers=1,
        num_local_experts=4,
        num_experts_per_tok=2,
        num_attention_heads=2,
        head_dim=64,
        q_lora_rank=8,
        o_groups=2,
        o_lora_rank=8,
        index_n_heads=2,
        index_head_dim=8,
        mlp_layer_types=["moe"],
    )
    model = NativeMTPBoostModel(config, mtp_layer_index=0)
    router = model.mtp.decoder.mlp.gate
    assert isinstance(router.e_score_correction_bias, torch.nn.Parameter)
    assert not router.e_score_correction_bias.requires_grad
    assert "mtp.decoder.mlp.gate.e_score_correction_bias" in model.state_dict()
    with torch.no_grad():
        router.weight.normal_(std=0.1)
    _, weights, _ = router(torch.randn(8, config.hidden_size))
    weights[:, 0].sum().backward()
    assert router.weight.grad is not None
    assert torch.isfinite(router.weight.grad).all()
    assert router.e_score_correction_bias.grad is None


@pytest.mark.parametrize("active_experts", [2, 4])
def test_grouped_experts_match_eager_outputs_and_gradients(active_experts):
    """Exercise real HF dispatch, including empty experts and saturated gates."""
    torch.manual_seed(29)
    config = hf_dsv4.DeepseekV4Config(
        hidden_size=16, intermediate_size=8, num_local_experts=4, swiglu_limit=0.5
    )
    config._experts_implementation = "eager"
    eager = hf_dsv4.DeepseekV4Experts(config)
    with torch.no_grad():
        for parameter in eager.parameters():
            parameter.normal_(std=0.3)
    grouped = copy.deepcopy(eager)
    grouped.config._experts_implementation = "grouped_mm"
    hidden = torch.randn(17, 16, requires_grad=True)
    grouped_hidden = hidden.detach().clone().requires_grad_(True)
    # Distinct top-k choices per token, with the final experts optionally unused.
    indices = (torch.arange(17)[:, None] + torch.arange(2)) % active_experts
    weights = torch.rand(17, 2, requires_grad=True)
    grouped_weights = weights.detach().clone().requires_grad_(True)
    expected = eager(hidden, indices, weights)
    actual = grouped(grouped_hidden, indices, grouped_weights)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    upstream = torch.randn_like(expected)
    expected.backward(upstream)
    actual.backward(upstream)
    for reference, candidate in zip(
        [hidden, weights, *eager.parameters()],
        [grouped_hidden, grouped_weights, *grouped.parameters()],
    ):
        torch.testing.assert_close(candidate.grad, reference.grad, rtol=1e-5, atol=1e-6)
