# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import contextlib
import os
import platform

import pytest

# Enforce no HuggingFace Hub network access for unit tests
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

with contextlib.suppress(ImportError):
    import huggingface_hub.constants as _hf_constants

    _hf_constants.HF_HUB_OFFLINE = True


@pytest.fixture(scope="session")
def _cpu_pools():
    pools = {}
    yield pools
    for pool in pools.values():
        pool.shutdown()


@pytest.fixture
def cpu_dist_workers(_cpu_pools):
    """Two persistent gloo ranks shared by every multi-rank unit test.

    A fresh spawn per test re-imports torch and transformers in each child (~10-20s on CI), so the
    workers start once (a killed pool is rebuilt). Jobs must leave the process group intact and
    undo any global state they set.
    """
    if platform.system() == "Windows":
        pytest.skip("Multi-rank tests are not run on Windows")
    # Lazy: the onnx-only partial-install job has no torch.
    from _test_utils.torch.distributed.utils import DistributedWorkerPool, get_live_pool

    # Replaced if an earlier test hung and killed it, so one hang fails one test.
    return get_live_pool(
        _cpu_pools, 2, lambda: DistributedWorkerPool(world_size=2, backend="gloo", teardown_fn=None)
    )


@pytest.hookimpl(tryfirst=True)  # xdist reads the marker in a tryfirst hook of its own
def pytest_collection_modifyitems(config, items):
    """Group the tests for ``--dist loadgroup`` so that one xdist worker holds the ``cpu_dist_workers`` pool.

    ``loadgroup`` treats an unmarked test as a group of its own, so every file becomes a group (what
    ``loadfile`` does), except that all files with a ``cpu_dist_workers`` test share one group. The
    pool (two gloo processes of about 0.7 GB) outlives its file, so with ``loadfile`` up to four
    workers held one at once, which on a 16 GB runner leaves the memory-hungry tests too little.
    Summed RSS of a CI-like unit run: 14-16 GB with ``loadfile``, about 11 GB with this group.
    """
    if not config.pluginmanager.hasplugin("xdist"):
        return
    pool_files = {item.path for item in items if "cpu_dist_workers" in item.fixturenames}
    for item in items:
        if item.get_closest_marker("xdist_group") is None:
            group = "cpu_pool" if item.path in pool_files else item.nodeid.split("::")[0]
            item.add_marker(pytest.mark.xdist_group(group))
