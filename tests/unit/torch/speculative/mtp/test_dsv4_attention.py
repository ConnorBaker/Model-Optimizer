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

"""Native MTP attention must preserve HF outputs and gradients when chunked."""

import pytest
import torch
from torch import nn

from modelopt.torch.speculative.mtp.deepseek_v4 import _chunked_mtp_attention_forward

hf_dsv4 = pytest.importorskip("transformers.models.deepseek_v4.modeling_deepseek_v4")


@pytest.fixture(autouse=True)
def _small_tensor_thread_pool():
    # These tiny attention heads are slower with a many-core BLAS thread pool.
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("masked", [False, True])
def test_chunked_attention_matches_native_sinks_and_compressed_mask(masked):
    """Recomputed query chunks preserve Q/K/V/sink gradients and extra KV slots."""
    torch.manual_seed(17)
    module = nn.Module()
    module.num_key_value_groups = 4
    module.sinks = nn.Parameter(torch.randn(4, dtype=torch.float64))
    query = torch.randn(2, 4, 513, 8, dtype=torch.float64, requires_grad=True)
    # Extra keys exercise the extended KV axis used by the compressor.
    kv = torch.randn(2, 1, 529, 8, dtype=torch.float64, requires_grad=True)
    mask = None
    if masked:
        mask = torch.zeros(2, 1, 513, 529, dtype=torch.float64)
        mask.masked_fill_(torch.rand_like(mask) < 0.4, float("-inf"))
        mask[..., 0, :] = float("-inf")  # Sink-only row must remain finite.

    expected, _ = hf_dsv4.eager_attention_forward(module, query, kv, kv, mask, 8**-0.5)
    actual, weights = _chunked_mtp_attention_forward(module, query, kv, kv, mask, scaling=8**-0.5)
    assert weights is None
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    upstream = torch.randn_like(actual)
    expected_grads = torch.autograd.grad(expected, (query, kv, module.sinks), upstream)
    actual_grads = torch.autograd.grad(actual, (query, kv, module.sinks), upstream)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-9, atol=1e-9)
