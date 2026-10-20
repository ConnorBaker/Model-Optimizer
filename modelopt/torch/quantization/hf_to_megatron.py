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

"""Translate a transformers-targeted ``quant_cfg`` to Megatron-Core module names.

Recipes name transformers modules (``q_proj``, ``gate_proj``, ``lm_head``) while Megatron-Core
renames and fuses them (``linear_qkv``, ``linear_fc1``, ``output_layer``). Megatron-Bridge's mapping
registry records that relation per architecture: :func:`module_mappings_from_registry` reads it and
:func:`translate_quant_cfg_to_megatron` applies it to a ``quant_cfg``. This module does not import
Megatron-Bridge; ``modelopt.torch.utils.plugins.mbridge.module_mappings_from_bridge`` is the entry
point that takes a bridge.
"""

import fnmatch
import itertools
import re
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from .config import QuantizerCfgEntry, RawQuantizeQuantCfgType, normalize_quant_cfg_list

__all__ = ["ModuleMapping", "module_mappings_from_registry", "translate_quant_cfg_to_megatron"]

_LINEAR_SUFFIXES = ("weight_quantizer", "input_quantizer", "output_quantizer")
_KV_SUFFIXES = ("q_bmm_quantizer", "k_bmm_quantizer", "v_bmm_quantizer")
# Wildcards other than the layer index (experts) are enumerated over these representative values.
_EXPERT_INDICES = (0, 1)
# Only the language-model decoder's layer index is enumerated; other layer containers (vision
# towers, MTP) are left to the original entries.
_DECODER_LAYERS = re.compile(r"^(?:language_model\.)?decoder\.layers\.\*")


@dataclass(frozen=True)
class ModuleMapping:
    """A Megatron module template and the transformers modules fused into it.

    ``*`` stands for an index and the first one is the decoder layer index. With ``hf_params`` the
    transformers names are fused parameters (``experts.gate_up_proj``) rather than modules.
    """

    megatron: str
    hf: tuple[str, ...]
    hf_params: bool = False
    suffixes: tuple[str, ...] = _LINEAR_SUFFIXES


def _strip_weight(param: str) -> str | None:
    """Return the module of a ``<module>.weight`` (or grouped ``.weight*``) name, else None."""
    for suffix in (".weight", ".weight*"):
        if param.endswith(suffix):
            return param.removesuffix(suffix)
    return None


def module_mappings_from_registry(registry: Any) -> list[ModuleMapping]:
    """Read the module-level mappings off a Megatron-Bridge ``MegatronMappingRegistry``.

    Only ``registry.mappings`` is read (each item's ``megatron_param`` and ``hf_param``), so any
    object of that shape works. Mappings that are not module weights, or that do not sit in the
    language-model decoder, are skipped. Each attention ``linear_qkv`` also yields its
    ``core_attention``, which hosts the KV-cache quantizers of the transformers attention module.
    """
    mappings: dict[ModuleMapping, None] = {}
    for mapping in registry.mappings:
        hf_names = mapping.hf_param
        hf_names = tuple(hf_names.values()) if isinstance(hf_names, dict) else (hf_names,)
        megatron = _strip_weight(mapping.megatron_param)
        modules = [_strip_weight(name) for name in hf_names]
        hf = tuple(name if module is None else module for name, module in zip(hf_names, modules))
        if megatron is None or any("**" in name for name in (megatron, *hf)):
            continue
        if megatron.count("*") and not _DECODER_LAYERS.match(megatron):
            continue
        if megatron.count("*") > min(name.count("*") for name in hf):
            continue  # flattened layouts, e.g. nested MTP layers
        mappings[ModuleMapping(megatron, hf, hf_params=None in modules)] = None
        if megatron.endswith(".self_attention.linear_qkv"):
            attention = ModuleMapping(
                megatron.removesuffix(".linear_qkv") + ".core_attention",
                (hf[0].rpartition(".")[0],),
                suffixes=_KV_SUFFIXES,
            )
            mappings[attention] = None
    return list(mappings)


@dataclass(frozen=True)
class _Quantizer:
    """A Megatron quantizer and the transformers quantizers it stands for."""

    name: str
    mapping: ModuleMapping
    suffix: str
    layer: int | None  # local layer index
    experts: tuple[int, ...]
    hf: tuple[str, ...]


def _fill(template: str, indices: Sequence[int]) -> str:
    for index in indices:
        template = template.replace("*", str(index), 1)
    return template


def _instances(
    mapping: ModuleMapping, layer_indices: Sequence[int]
) -> Iterator[tuple[int | None, tuple[int, ...], tuple[int, ...]]]:
    """Yield the (local layer, expert indices, transformers indices) of a mapping's modules."""
    wildcards = mapping.megatron.count("*")
    if not wildcards:
        yield None, (), ()
        return
    for local in range(len(layer_indices)):
        for experts in itertools.product(_EXPERT_INDICES, repeat=wildcards - 1):
            yield local, experts, (layer_indices[local], *experts)


def _hf_quantizer_names(mapping: ModuleMapping, indices: Sequence[int], suffix: str) -> list[str]:
    names = []
    for hf in mapping.hf:
        spare = hf.count("*") - len(indices)
        for extra in itertools.product(_EXPERT_INDICES, repeat=spare):
            module = _fill(hf, (*indices, *extra))
            if not mapping.hf_params:
                names.append(f"{module}.{suffix}")
            elif suffix != "output_quantizer":  # a fused parameter has weight/input quantizers only
                base, _, param = module.rpartition(".")
                names.append(f"{base}.{param}_{suffix}")
    return names


def _megatron_quantizers(
    mappings: Sequence[ModuleMapping], layer_indices: Sequence[int]
) -> list[_Quantizer]:
    quantizers = []
    for mapping in mappings:
        for layer, experts, hf_indices in _instances(mapping, layer_indices):
            module = _fill(mapping.megatron, () if layer is None else (layer, *experts))
            for suffix in mapping.suffixes:
                hf = _hf_quantizer_names(mapping, hf_indices, suffix)
                if hf:
                    quantizers.append(
                        _Quantizer(f"{module}.{suffix}", mapping, suffix, layer, experts, tuple(hf))
                    )
    return quantizers


def _matcher(entry: QuantizerCfgEntry) -> Callable[[str], Any] | None:
    """Match like ``set_quantizer_by_cfg``; None for ``parent_class`` entries, which are class-based."""
    if entry.get("parent_class"):
        return None
    return re.compile(fnmatch.translate(entry["quantizer_name"])).match


def _resolve(
    entries: Sequence[QuantizerCfgEntry], matchers: Sequence[Callable | None], name: str
) -> tuple[bool, Any] | None:
    """The (enable, cfg) the entries leave ``name`` in, applied in order like ``set_quantizer_by_cfg``."""
    state = None
    for entry, match in zip(entries, matchers):
        if match is not None and match(name):
            cfg = entry["cfg"] if entry["cfg"] is not None else (state[1] if state else None)
            state = (entry["enable"], cfg)
    return state


def _describe(state: tuple[bool, Any] | None) -> str:
    if state is None:
        return "not configured"
    enable, cfg = state
    if not enable:
        return "disabled"
    cfgs = cfg if isinstance(cfg, list) else [] if cfg is None else [cfg]
    return f"enabled {[c.model_dump(exclude_unset=True) for c in cfgs]}"[:100]


def _check_fused_consistency(
    entries: Sequence[QuantizerCfgEntry],
    matchers: Sequence[Callable | None],
    quantizers: Sequence[_Quantizer],
) -> None:
    """Raise if the entries configure the transformers pieces of a fused module differently."""
    conflicts: dict[tuple[str, str], list[tuple[_Quantizer, list]]] = defaultdict(list)
    for quantizer in quantizers:
        if len(quantizer.hf) > 1:
            states = [_resolve(entries, matchers, name) for name in quantizer.hf]
            if any(state != states[0] for state in states[1:]):
                conflicts[(quantizer.mapping.megatron, quantizer.suffix)].append(
                    (quantizer, states)
                )
    if not conflicts:
        return
    lines = []
    for (template, suffix), found in conflicts.items():
        quantizer, states = found[0]
        pieces = "".join(
            f"\n      {name}: {_describe(state)}" for name, state in zip(quantizer.hf, states)
        )
        lines.append(
            f"  {template}.{suffix} ({len(found)} instance(s)), e.g. {quantizer.name}:{pieces}"
        )
    raise ValueError(
        "Megatron-Core fuses transformers modules that the quant_cfg configures differently, which "
        "one fused quantizer cannot express. Configure the pieces alike, or exclude all of them:\n"
        + "\n".join(lines)
    )


def _globs(missing: Sequence[_Quantizer], num_layers: int) -> list[str]:
    """Megatron globs selecting exactly ``missing``: a ``*`` for the layer when all layers are in."""
    groups: dict[tuple[str, str], dict[int | None, set]] = defaultdict(lambda: defaultdict(set))
    for quantizer in missing:
        groups[(quantizer.mapping.megatron, quantizer.suffix)][quantizer.layer].add(
            quantizer.experts
        )
    globs = []
    for (template, suffix), by_layer in groups.items():
        every_expert = set(
            itertools.product(_EXPERT_INDICES, repeat=max(template.count("*") - 1, 0))
        )
        if any(experts != every_expert for experts in by_layer.values()):
            raise ValueError(
                f"A quant_cfg entry selects only some experts of {template}; Megatron-Core "
                "has them in one module, so select all or none."
            )
        layers = sorted(layer for layer in by_layer if layer is not None)
        every_layer = None in by_layer or len(layers) == num_layers
        modules = [template] if every_layer else [_fill(template, (layer,)) for layer in layers]
        globs.extend(f"*{module}.{suffix}" for module in modules)
    return globs


def translate_quant_cfg_to_megatron(
    quant_cfg: RawQuantizeQuantCfgType,
    mappings: Sequence[ModuleMapping],
    layer_indices: Sequence[int],
) -> list[QuantizerCfgEntry]:
    """Translate a ``quant_cfg`` that names transformers modules to Megatron-Core module names.

    Every original entry is kept in place. After it come copies for the Megatron quantizers that
    the entry selects through their transformers names but its own glob does not match, so
    entries that already match, such as Megatron-spelled ones and ``parent_class`` ones, pass
    through unchanged. Quantizers that no mapping covers (e.g. ``softmax_quantizer``) are matched
    by the original glob only.

    A fused Megatron quantizer, e.g. ``linear_qkv`` for ``q/k/v_proj``, has a single state, so the
    entries must leave all its transformers pieces alike. A layer index in a glob (``*layers.7.*``)
    is a transformers (global) one and the copies follow ``layer_indices`` accordingly, but the
    original entry still matches this rank's local indices, so such globs are unreliable under
    pipeline parallelism.

    Args:
        quant_cfg: The ``quant_cfg`` in any form :func:`normalize_quant_cfg_list` accepts.
        mappings: The architecture's module mappings, e.g. from :func:`module_mappings_from_registry`.
        layer_indices: The global decoder layer index of each of this rank's local layers, in order.

    Raises:
        ValueError: If the entries configure the pieces of a fused module differently, or select
            only some experts of a grouped-experts module.
    """
    entries = normalize_quant_cfg_list(quant_cfg)
    matchers = [_matcher(entry) for entry in entries]
    quantizers = _megatron_quantizers(mappings, layer_indices)
    _check_fused_consistency(entries, matchers, quantizers)

    translated = []
    for entry, match in zip(entries, matchers):
        translated.append(entry)
        if match is None:
            continue
        missing = [q for q in quantizers if not match(q.name) and any(match(name) for name in q.hf)]
        translated.extend(
            entry.model_copy(update={"quantizer_name": glob})
            for glob in _globs(missing, len(layer_indices))
        )
    return translated
