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

import json
import sys
from itertools import pairwise

import bench
import pytest

# Per turn: acceptance histogram (accepted drafts -> verify steps), streamed reply chunks.
_TURNS = [({"0": 1, "3": 1}, ["hel", "lo"]), ({"7": 2}, ["wor", "ld"])]


def _fake_aiperf(cmd, check, env, stdin):
    """Stand-in for `aiperf profile`: answers every session in the input file."""
    assert "HF_HUB_OFFLINE" not in env and "TRANSFORMERS_OFFLINE" not in env
    opt = {k: v for k, v in pairwise(cmd) if k.startswith("--")}
    extra = json.loads(opt["--extra-inputs"])
    assert extra == {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "max_tokens": 64,
        "skip_special_tokens": False,
        "chat_template_kwargs": {"reasoning_effort": "high"},
    }
    out = opt["--output-artifact-dir"]
    with (
        open(opt["--input-file"]) as f,
        open(f"{out}/profile_export.jsonl", "w") as rec,
        open(f"{out}/profile_export_raw.jsonl", "w") as raw,
    ):
        for n, line in enumerate(f):
            session = json.loads(line)
            assert [t["text"] for t in session["turns"]] == ["q1", "q2"]
            for turn, (histogram, chunks) in enumerate(_TURNS):
                meta = {
                    "conversation_id": session["session_id"],
                    "turn_index": turn,
                    "request_start_ns": (2 * n + turn + 1) * 10**9,
                }
                metrics = {
                    "time_to_first_token": {"value": 100.0, "unit": "ms"},
                    "inter_chunk_latency": {"value": [10.0, 10.0], "unit": "ms"},
                    "output_sequence_length": {
                        "value": 1 + sum((int(j) + 1) * c for j, c in histogram.items()),
                        "unit": "tokens",
                    },
                }
                stats = {"acceptance_histogram": histogram}
                row = {"metadata": meta, "metrics": metrics, "spec_decode_acceptance": stats}
                rec.write(json.dumps(row) + "\n")
                packets = [
                    {"name": "data", "value": json.dumps({"choices": [{"delta": {"content": c}}]})}
                    for c in chunks
                ] + [{"name": "data", "value": "[DONE]"}]
                raw.write(
                    json.dumps({"metadata": meta, "responses": [{"packets": packets}]}) + "\n"
                )


def test_bench_writes_specdec_bench_files(tmp_path, monkeypatch):
    questions = tmp_path / "question.jsonl"
    questions.write_text(
        "".join(
            json.dumps({"question_id": 81 + i, "category": c, "turns": ["q1", "q2"]}) + "\n"
            for i, c in enumerate(["writing", "math"])
        )
    )
    runtime = tmp_path / "runtime.yaml"
    runtime.write_text(
        "sampling_kwargs: {temperature: 0}\nchat_template_args: {reasoning_effort: high}\n"
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(bench, "_get_json", lambda url: {"data": [{"id": "target"}]})
    monkeypatch.setattr(bench.subprocess, "run", _fake_aiperf)
    argv = f"bench.py --url http://server:8000 --tokenizer {tmp_path} --mtbench {questions}"
    argv += f" --runtime_params {runtime} --temperature 1 --concurrency 2 --output_length 64"
    argv += f" --num_gpus 4 --save_dir {tmp_path / 'results'}"
    monkeypatch.setattr(sys, "argv", argv.split())

    bench.main()

    out = tmp_path / "results" / "t1_c2"
    acceptance = json.loads((out / "acceptance_rate.json").read_text())[0]
    steps = [1, 1, 4, 1, 8, 8]  # the prefill token, then 1 + accepted drafts per verify step
    assert acceptance["Average_AL"] == pytest.approx(sum(steps) / len(steps))
    assert acceptance["Category_AL"].keys() == {"writing", "math"}
    timing = json.loads((out / "timing.json").read_text())[0]
    # 4 requests starting at 1..4 s, each ending 120 ms later.
    assert timing["Output TPS"] == pytest.approx(2 * sum(steps) / (4.12 - 1.0))
    assert timing["Output TPS/gpu"] == pytest.approx(timing["Output TPS"] / 4)
    responses = [json.loads(line) for line in open(out / "mtbench_responses.jsonl")]
    assert [r["choices"][0]["turns"] for r in responses] == [["hello", "world"]] * 2


def test_metrics_match_specdec_bench(tmp_path):
    from specdec_bench import metrics

    requests = [
        {
            "question_id": 81 + i,
            "category": "writing" if i < 10 else "roleplay",
            "turns": ["a", "b"],
        }
        for i in range(12)
    ]
    steps = [[[1, 1 + i % 4, 8], [1, 2, 3 + i % 2, 8]] for i in range(12)]
    times = [[float(i), i + 0.2, i + 0.25, i + 0.31] for i in range(24)]
    tokens = [9 + i % 5 for i in range(24)]

    reference, timing = metrics.MTBench(), metrics.Timing(4)
    for metric in (reference, timing):
        metric.update_directory(str(tmp_path))
    for i, turns in enumerate(steps):
        for turn, lengths in enumerate(turns):
            reference.process_step({"output_ids": [[[0] * n for n in lengths]]}, i, turn)
            k = 2 * i + turn
            timing.process_step(
                {"output_ids": [[[0] * tokens[k]]], "token_times": times[k]}, i, turn
            )
    texts = [[{"role": "user", "content": "a"}, {"role": "assistant", "content": ""}]] * 12
    reference.process_final(texts)
    timing.process_final(texts)

    ours = bench.acceptance_metrics(requests, steps, by_index=True)
    for key in (
        "Request_AL", "Category_AL", "Average_AL", "Acceptance_Length_Histogram",
        "Conditional_Acceptance_Rate", "Joint_Acceptance_Rate",
    ):  # fmt: skip
        assert ours[key] == pytest.approx(reference.out[key]), key
    ours_timing = bench.timing_metrics(times, tokens, 4)
    assert ours_timing.keys() == timing.out.keys()
    for key, value in timing.out.items():
        assert ours_timing[key] == (pytest.approx(value) if isinstance(value, float) else value), (
            key
        )
