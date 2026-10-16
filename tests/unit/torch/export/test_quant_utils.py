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

import pytest
import torch

from modelopt.torch.export.quant_utils import sync_tied_input_amax
from modelopt.torch.quantization.nn import TensorQuantizer


@pytest.mark.parametrize("first_proj_attr", ["gate_up_proj", "up_proj"])
@pytest.mark.parametrize("declared", [False, True])
@pytest.mark.parametrize(
    ("first_target", "down_target", "expected_amaxes"),
    [
        (None, None, [(2, 3), (5, 7), (11, 13)]),
        ("b", None, [(5, 3), (5, 7), (11, 13)]),
        (None, "b", [(2, 7), (5, 7), (11, 13)]),
        ("b", "b", [(5, 7), (5, 7), (11, 13)]),
        ("b", "c", [(5, 13), (5, 7), (11, 13)]),
    ],
)
def test_sync_tied_input_amax_groups_moe_projections_independently(
    first_proj_attr, declared, first_target, down_target, expected_amaxes
):
    """Only projections with their own declared or physical tie share activation ranges."""
    parent = torch.nn.Module()
    parent.all_tied_weights_keys = {}
    projections = (first_proj_attr, "down_proj")
    for name, amaxes in zip(("a", "b", "c"), ((2, 3), (5, 7), (11, 13))):
        module = torch.nn.Module()
        module._first_proj_attr = first_proj_attr
        for projection, amax in zip(projections, amaxes):
            setattr(module, projection, torch.nn.Parameter(torch.ones(2, 4, 4)))
            quantizer = TensorQuantizer()
            quantizer.amax = torch.tensor(float(amax))
            setattr(module, f"{projection}_input_quantizer", quantizer)
        parent.add_module(name, module)

    for projection, target in zip(projections, (first_target, down_target)):
        if target is not None:
            if declared:
                parent.all_tied_weights_keys[f"a.{projection}"] = f"{target}.{projection}"
            else:
                setattr(parent.a, projection, getattr(getattr(parent, target), projection))

    assert sync_tied_input_amax(parent) == sum(
        target is not None for target in (first_target, down_target)
    )
    for module, amaxes in zip((parent.a, parent.b, parent.c), expected_amaxes):
        for projection, expected in zip(projections, amaxes):
            quantizer = getattr(module, f"{projection}_input_quantizer")
            torch.testing.assert_close(quantizer.amax, torch.tensor(float(expected)))
