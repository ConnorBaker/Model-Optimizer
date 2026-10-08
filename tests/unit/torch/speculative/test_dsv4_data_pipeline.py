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

"""Unit tests for DSV4 native-MTP data preparation and parity validation."""

from __future__ import annotations

import importlib.util
import json
import multiprocessing
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from datasets import config as datasets_config

from tests._test_utils.torch.transformers_models import get_tiny_tokenizer

_REPO_ROOT = Path(__file__).resolve().parents[4]
_COLLECT_DIR = _REPO_ROOT / "examples/speculative_decoding/collect_hidden_states"
sys.path.insert(0, str(_COLLECT_DIR))


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, _COLLECT_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prepare = _load_script("prepare_dsv4_mtp_data")
validate = _load_script("validate_dsv4_mtp_features")
compute = _load_script("compute_hidden_states_dsv4")


def _valid_row() -> dict:
    reasoning = "We need to reason carefully about the problem."
    answer = "The answer is 42."
    return {
        "uuid": "sample-1",
        "category": "math",
        "messages": [
            {"role": "system", "content": "Be accurate."},
            {"role": "user", "content": "What is the answer?"},
        ],
        "reasoning_content": reasoning,
        "generation": answer,
        "finish_reason": "stop",
        "serialized_output": [
            {
                "role": "assistant",
                "reasoning_content": reasoning,
                "content": answer,
            }
        ],
    }


def test_normalize_generation_row_preserves_reasoning_and_answer():
    record, reason = prepare.normalize_generation_row(_valid_row())

    assert reason is None
    assert record["conversation_id"] == "sample-1"
    assert [message["role"] for message in record["messages"]] == [
        "system",
        "user",
        "assistant",
    ]
    assert record["messages"][-1]["reasoning_content"].startswith("We need to reason")
    assert record["messages"][-1]["content"] == "The answer is 42."


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        ({"finish_reason": "error"}, "unsupported_finish_reason"),
        ({"reasoning_content": "", "generation": ""}, "empty_response"),
        ({"generation": ""}, "empty_final_answer"),
        ({"generation": "different answer"}, "serialized_output_mismatch"),
        ({"uuid": "../unsafe"}, "invalid_conversation_id"),
    ],
)
def test_normalize_generation_row_rejects_unfit_rows(mutation, expected_reason):
    row = _valid_row()
    row.update(mutation)

    _, reason = prepare.normalize_generation_row(row)

    assert reason == expected_reason


@pytest.mark.parametrize(
    ("finish_reason", "reasoning", "answer", "suffix"),
    [
        ("length", "Still reasoning", "", "Still reasoning"),
        ("length", "Reasoned", "Partial answer", "Reasoned</think>Partial answer"),
        ("stop", "Reasoned", "Answer", "Reasoned</think>Answer"),
        ("stop", "", "Answer", "</think>Answer"),
    ],
)
def test_available_response_tokenization(finish_reason, reasoning, answer, suffix):
    row = _valid_row()
    row.update(finish_reason=finish_reason, reasoning_content=reasoning, generation=answer)
    row["serialized_output"][0].update(reasoning_content=reasoning, content=answer)
    record, reason = prepare.normalize_generation_row(row)
    assert reason is None
    assert record["finish_reason"] == finish_reason

    tokenizer = get_tiny_tokenizer()
    tokenizer.chat_template = (
        _REPO_ROOT / "examples/speculative_decoding/input_conversations/dsv4_thinking.jinja"
    ).read_text()
    expected = (
        tokenizer.bos_token
        + "Be accurate.<\uff5cUser\uff5c>What is the answer?<\uff5cAssistant\uff5c><think>"
        + suffix
        + (tokenizer.eos_token if finish_reason == "stop" else "")
    )
    ids, mask = prepare.tokenize_with_loss_mask(tokenizer, record["messages"], True)
    assert ids.squeeze(0).tolist() == tokenizer(expected, add_special_tokens=False)["input_ids"]
    assert mask.any()
    assert mask[-1] == 1
    assert tokenizer.apply_chat_template(record["messages"], tokenize=False) == expected


def test_category_budget_is_equal_by_default_and_exact():
    budgets = prepare._budgets(1_000_003, prepare.DEFAULT_CATEGORY_WEIGHTS)

    assert sum(budgets.values()) == 1_000_003
    assert max(budgets.values()) - min(budgets.values()) == 1


def test_zero_token_budget_streams_every_eligible_row(tmp_path, monkeypatch):
    second_row = _valid_row()
    second_row["uuid"] = "sample-2"
    second_row["generation"] = "The second answer is 43."
    second_row["serialized_output"][0]["content"] = second_row["generation"]
    rows = [_valid_row(), second_row]
    input_file = tmp_path / "input.jsonl"
    input_file.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    output_dir = tmp_path / "prepared"
    tokenizer = SimpleNamespace(pad_token=None, eos_token="<eos>", chat_template=None)

    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *_args: tokenizer)
    monkeypatch.setattr(prepare, "load_chat_template", lambda _path: "generation endgeneration")
    monkeypatch.setattr(prepare, "verify_generation_tags", lambda _template: None)
    monkeypatch.setattr(
        prepare,
        "tokenize_with_loss_mask",
        lambda *_args, **_kwargs: (torch.arange(4).unsqueeze(0), torch.tensor([0, 1, 1, 1])),
    )
    args = SimpleNamespace(
        input_file=[input_file],
        output_dir=output_dir,
        output_prefix="",
        model_dir=tmp_path,
        chat_template=tmp_path / "template.jinja",
        max_seq_len=2048,
        token_budget=0,
        category_weight=[],
        validation_fraction=0.0,
        records_per_shard=1,
        progress_interval=10_000,
        seed=0,
        deduplicate=True,
        validate_tokenization=True,
        debug_max_input_rows=None,
        write_rejects=False,
    )

    prepare.main(args)

    records = [
        json.loads(line)
        for path in sorted(output_dir.glob("train-*.jsonl"))
        for line in path.read_text().splitlines()
    ]
    report = json.loads((output_dir / "report.json").read_text())
    assert [record["conversation_id"] for record in records] == ["sample-1", "sample-2"]
    assert report["selected_rows"] == 2
    assert report["selected_tokens"] == 8


def test_deferred_tokenization_writes_prefixed_shards_without_a_tokenizer(tmp_path):
    input_file = tmp_path / "input.jsonl"
    input_file.write_text(json.dumps(_valid_row()) + "\n")
    output_dir = tmp_path / "prepared"
    args = SimpleNamespace(
        input_file=[input_file],
        output_dir=output_dir,
        output_prefix="math",
        model_dir=tmp_path / "missing-tokenizer",
        chat_template=tmp_path / "template.jinja",
        max_seq_len=8192,
        token_budget=0,
        category_weight=[],
        validation_fraction=0.0,
        records_per_shard=10_000,
        progress_interval=10_000,
        seed=0,
        deduplicate=False,
        validate_tokenization=False,
        debug_max_input_rows=None,
        write_rejects=False,
    )

    prepare.main(args)

    record = json.loads((output_dir / "math-train-00000.jsonl").read_text())
    report = json.loads((output_dir / "math-report.json").read_text())
    assert record["conversation_id"] == "sample-1"
    assert "num_tokens" not in record
    assert report["configuration"]["deduplicate"] is False
    assert report["configuration"]["validate_tokenization"] is False
    assert report["selected_rows"] == 1
    assert report["selected_tokens"] is None


def test_feature_save_respects_shared_filesystem_limit(tmp_path, monkeypatch):
    output_file = tmp_path / "sample.pt"
    payload = {"hidden_states": torch.ones(512, dtype=torch.bfloat16)}
    saved_paths = []
    gibibyte = 2**30

    monkeypatch.setattr(
        compute.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=10 * gibibyte, used=7 * gibibyte),
    )
    monkeypatch.setattr(
        compute,
        "_atomic_torch_save",
        lambda _payload, path: saved_paths.append(path),
    )

    assert compute._save_below_filesystem_limit(payload, output_file, 90)
    assert saved_paths == [output_file]
    assert not list((tmp_path / ".storage-reservations").iterdir())

    monkeypatch.setattr(
        compute.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=10 * gibibyte, used=9 * gibibyte),
    )
    assert not compute._save_below_filesystem_limit(payload, output_file, 90)
    assert saved_paths == [output_file]
    assert not list((tmp_path / ".storage-reservations").iterdir())


@pytest.mark.parametrize("interrupt_after_publish", [False, True])
def test_feature_reservation_survives_killed_writer(tmp_path, monkeypatch, interrupt_after_publish):
    payload = {"hidden_states": torch.ones(16, dtype=torch.bfloat16)}
    monkeypatch.setattr(
        compute.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=10 * 2**20, used=8 * 2**20),
    )
    context = multiprocessing.get_context("fork")
    parent_connection, child_connection = context.Pipe()

    def interrupted_save():
        replace = compute.os.replace

        def replace_and_pause(source, destination):
            if Path(destination).suffix != ".bytes":
                return replace(source, destination)
            if interrupt_after_publish:
                replace(source, destination)
            child_connection.send("reservation ready")
            child_connection.recv()

        # Control only the kill timing; run the real reservation and filesystem writes.
        compute.os.replace = replace_and_pause
        compute._save_below_filesystem_limit(payload, tmp_path / "interrupted.pt", 95)

    worker = context.Process(target=interrupted_save)
    worker.start()
    try:
        assert parent_connection.poll(10), "Writer did not reach atomic reservation publication"
        assert parent_connection.recv() == "reservation ready"
    finally:
        if worker.is_alive():
            worker.kill()
        worker.join(timeout=5)
        parent_connection.close()
        child_connection.close()
    assert not worker.is_alive()
    assert worker.exitcode != 0
    assert not (tmp_path / "interrupted.pt").exists()

    reservations = list((tmp_path / ".storage-reservations").glob("*.bytes"))
    assert len(reservations) == int(interrupt_after_publish)
    assert all(int(path.read_text()) == 2**20 + 32 for path in reservations)
    output = tmp_path / "resumed.pt"
    saved = compute._save_below_filesystem_limit(payload, output, 95)
    # An unpublished reservation is ignored; a published orphan still reserves capacity.
    assert saved is not interrupt_after_publish
    if saved:
        assert torch.equal(
            torch.load(output, weights_only=True)["hidden_states"], payload["hidden_states"]
        )
    else:
        assert not output.exists()


def test_load_completed_ids_for_archived_features(tmp_path):
    completed_ids_file = tmp_path / "completed.txt"
    completed_ids_file.write_text("sample-1\n\nsample-2\n")

    assert compute._load_completed_ids(completed_ids_file) == {"sample-1", "sample-2"}

    completed_ids_file.write_text("../unsafe\n")
    with pytest.raises(ValueError, match="Invalid conversation ID"):
        compute._load_completed_ids(completed_ids_file)


@pytest.mark.parametrize("include_new", [False, True])
def test_collector_resume_skips_quota_queries_for_completed_rows(
    tmp_path, monkeypatch, capsys, include_new
):
    source = tmp_path / "conversations.jsonl"
    identifiers = ["archived", "existing"] + (["new"] if include_new else [])
    source.write_text(
        "\n".join(json.dumps({"conversation_id": identifier}) for identifier in identifiers)
    )
    archived = tmp_path / "completed.txt"
    archived.write_text("archived\n")
    output = tmp_path / "features"
    output.mkdir()
    torch.save({"sentinel": 42}, output / "existing.pt")
    monkeypatch.setattr(datasets_config, "HF_DATASETS_CACHE", tmp_path / "dataset-cache")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    # This resume path must not tokenize or run the unavailable full target model.
    monkeypatch.setattr(compute, "_prepare_tokenizer", lambda *_args: None)
    monkeypatch.setattr(compute, "load_dsv4_target_model", lambda *_args, **_kwargs: None)
    quota_queries = []

    def full_filesystem(path):
        quota_queries.append(path)
        return SimpleNamespace(total=100, used=96)

    monkeypatch.setattr(compute.shutil, "disk_usage", full_filesystem)
    compute.main(
        SimpleNamespace(
            dp_world_size=1,
            dp_rank=0,
            max_filesystem_usage_percent=95,
            device="cpu",
            input_data=source,
            debug_max_num_conversations=None,
            completed_ids_file=archived,
            output_dir=output,
            model_dir=tmp_path,
            converted_checkpoint=tmp_path,
            max_seq_len=8192,
        )
    )
    assert quota_queries == ([output] if include_new else [])
    assert f"stopped_at_storage_limit={include_new}" in capsys.readouterr().out
    assert list(output.iterdir()) == [output / "existing.pt"]
    assert torch.load(output / "existing.pt", weights_only=True) == {"sentinel": 42}


def test_tensor_error_metrics_identical_bf16():
    tensor = torch.randn(17, 4, 8, dtype=torch.bfloat16)

    metrics = validate.tensor_error_metrics(tensor, tensor.clone(), chunk_elements=23)

    assert metrics["cosine"] == pytest.approx(1.0)
    assert metrics["relative_l2"] == 0.0
    assert metrics["max_absolute"] == 0.0


def test_tensor_error_metrics_rejects_precision_mismatch():
    with pytest.raises(TypeError, match="dtype mismatch"):
        validate.tensor_error_metrics(
            torch.ones(2, dtype=torch.bfloat16),
            torch.ones(2, dtype=torch.float32),
        )


def test_dsv4_template_masks_complete_assistant_span():
    template = (
        _REPO_ROOT / "examples/speculative_decoding/input_conversations/dsv4_thinking.jinja"
    ).read_text()

    generation_start_match = re.search(r"{%[-+]?\s*generation\s*[-+]?%}", template)
    generation_end_match = re.search(r"{%[-+]?\s*endgeneration\s*[-+]?%}", template)
    assert generation_start_match is not None
    assert generation_end_match is not None

    generation_start = generation_start_match.start()
    reasoning = template.index("reasoning_content")
    final_answer = template.index("message['content']", reasoning)
    generation_end = generation_end_match.start()
    assert generation_start < reasoning < final_answer < generation_end
