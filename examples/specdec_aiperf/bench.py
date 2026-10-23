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

"""Measure speculative-decoding acceptance length and throughput of a served model with AIPerf.

Results use specdec_bench's file names, fields and acceptance-length definition, so they can be
compared with in-process specdec_bench runs.
"""

import argparse
import contextlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from itertools import accumulate
from pathlib import Path

import numpy as np
import yaml

MTBENCH_TOPICS = [
    "writing", "roleplay", "reasoning", "math", "coding", "extraction", "stem", "humanities",
]  # fmt: skip
# What examples/specdec_bench/prepare_data.py leaves in SPEED-Bench turns it could not resolve.
SPEED_PLACEHOLDER = "FULL BENCHMARK DATA SHOULD BE FETCHED FROM THE SOURCE USING SPECDEC_BENCH"
RESPONSES_FILE = {
    "mtbench": "mtbench_responses.jsonl",
    "specbench": "specbench_responses.jsonl",
    "speed": "specbench_responses.jsonl",
}


def _stratified(rows, n):
    """Pick ``n`` rows round-robin across categories, as specdec_bench samples SPEED-Bench."""
    by_category = defaultdict(list)
    for row in rows:
        by_category[row.get("category")].append(row)
    if n >= len(rows) or len(by_category) <= 1:
        return rows[:n]
    queues = list(by_category.values())
    picked = []
    for i in range(max(map(len, queues))):
        picked += [queue[i] for queue in queues if i < len(queue)]
    return picked[:n]


def load_requests(dataset, path, num_requests=None):
    """Requests with question_id, category, system_prompt and turns."""
    if dataset == "speed":
        import pyarrow.parquet as pq

        path = Path(path)
        files = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
        rows = [row for f in files for row in pq.read_table(f).to_pylist()]
        if any(turn.startswith(SPEED_PLACEHOLDER) for row in rows for turn in row["turns"]):
            raise ValueError(f"{path} has unresolved turns; run specdec_bench's prepare_data.py")
        if num_requests is not None:
            rows = _stratified(rows, num_requests)
    else:
        with open(path) as f:
            rows = [json.loads(line) for line in f][:num_requests]
    requests = []
    for i, row in enumerate(rows):
        category = row.get("category")
        if category is None and dataset == "mtbench":
            category = MTBENCH_TOPICS[i // 10]
        requests.append(
            {
                "question_id": row.get("question_id", i),
                "category": category,
                "system_prompt": row.get("system_prompt"),
                "turns": list(row["turns"] if "turns" in row else row["prompt"]),
            }
        )
    return requests


def write_input(requests, path):
    """Write an AIPerf multi_turn file with one session per request, keyed by request index."""
    with open(path, "w") as f:
        for i, request in enumerate(requests):
            turns = [{"text": turn} for turn in request["turns"]]
            if request["system_prompt"] is not None:
                turns.insert(0, {"role": "system", "text": request["system_prompt"]})
            f.write(json.dumps({"session_id": str(i), "turns": turns}) + "\n")


def _get_json(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def _key(record):
    return int(record["metadata"]["conversation_id"]), record["metadata"]["turn_index"]


def _reply(raw):
    """Concatenate the streamed content of one raw-exported request."""
    parts = []
    for response in raw["responses"]:
        for packet in response.get("packets", []):
            if packet.get("name") != "data" or packet["value"] == "[DONE]":
                continue
            choices = json.loads(packet["value"]).get("choices", [])
            parts.extend((choice.get("delta") or {}).get("content") or "" for choice in choices)
    return "".join(parts)


def _seconds(metric):
    assert metric["unit"] == "ms", metric
    value = metric["value"]
    return [v / 1e3 for v in value] if isinstance(value, list) else value / 1e3


def collect(artifact_dir, requests):
    """Per request: per-turn step lengths and replies; per turn: token times and output tokens."""
    records = {_key(r): r for r in _read_jsonl(artifact_dir / "profile_export.jsonl")}
    replies = {_key(r): _reply(r) for r in _read_jsonl(artifact_dir / "profile_export_raw.jsonl")}
    steps, texts, times, tokens = [], [], [], []
    for i, request in enumerate(requests):
        request_steps, request_texts = [], []
        for turn in range(len(request["turns"])):
            record = records.get((i, turn))
            if record is None or "spec_decode_acceptance" not in record:
                raise RuntimeError(
                    f"No spec-decode stats for request {i} turn {turn}: the request failed, or the "
                    "server was started without --per-request-spec-decode-metrics."
                )
            # Counted like specdec_bench's streamed chunks: the prefill token, then each verify step.
            histogram = record["spec_decode_acceptance"]["acceptance_histogram"]
            request_steps.append([1] + [int(j) + 1 for j, n in histogram.items() for _ in range(n)])
            request_texts.append(replies[(i, turn)])
            metrics = record["metrics"]
            start = record["metadata"]["request_start_ns"] / 1e9
            first = start + _seconds(metrics["time_to_first_token"])
            gaps = (
                _seconds(metrics["inter_chunk_latency"]) if "inter_chunk_latency" in metrics else []
            )
            times.append([start, first] + [first + t for t in accumulate(gaps)])
            tokens.append(metrics["output_sequence_length"]["value"])
        steps.append(request_steps)
        texts.append(request_texts)
    return steps, texts, times, tokens


def acceptance_metrics(requests, steps, by_index=False):
    """specdec_bench's acceptance_rate.json: a request's AL pools its turns; averages are per request.

    ``Request_AL`` is keyed by request index for MT-Bench and by question_id otherwise, as there.
    """
    out = {"Request_AL": {}}
    per_category = defaultdict(list)
    histogram = defaultdict(int)
    for i, (request, turns) in enumerate(zip(requests, steps)):
        lengths = [length for turn in turns for length in turn]
        out["Request_AL"][i if by_index else request["question_id"]] = sum(lengths) / len(lengths)
        per_category[request["category"]].append(sum(lengths) / len(lengths))
        for length in lengths:
            histogram[length] += 1
    if None not in per_category:
        out["Category_AL"] = {c: float(np.mean(v)) for c, v in per_category.items()}
    out["Average_AL"] = float(np.mean(list(out["Request_AL"].values())))
    out["Acceptance_Length_Histogram"] = dict(sorted(histogram.items()))
    out["Conditional_Acceptance_Rate"], out["Joint_Acceptance_Rate"] = {}, {}
    total = remaining = sum(histogram.values())
    previous = joint = 1.0
    for length, count in out["Acceptance_Length_Histogram"].items():
        rate = remaining / total / previous
        joint *= rate
        out["Conditional_Acceptance_Rate"][length] = rate
        out["Joint_Acceptance_Rate"][length] = joint
        previous = remaining / total
        remaining -= count
    return out


def _stats(data):
    data = np.asarray(data, dtype=float)
    quantiles = np.percentile(data, [25, 50, 75])
    return {
        "min": f"{data.min():.4f}",
        "max": f"{data.max():.4f}",
        "mean": f"{data.mean():.4f}",
        "std": f"{data.std():.4f}",
        "quantiles": {q: f"{v:.4f}" for q, v in zip(["0.25", "0.5", "0.75"], quantiles)},
    }


def timing_metrics(times, tokens, num_gpus):
    """specdec_bench's timing.json from per-request [start, first token, ..., last token] seconds."""
    span = max(t[-1] for t in times) - min(t[0] for t in times)
    out = {"Output TPS": sum(tokens) / span}
    out["Output TPS/gpu"] = out["Output TPS"] / num_gpus
    e2e, ttft, step_times, rates = [], [], [], []
    for num_tokens, t in zip(tokens, times):
        if len(t) > 1:
            e2e.append(t[-1] - t[0])
            ttft.append(t[1] - t[0])
        if len(t) > 2:
            rates.append((num_tokens - 1) / (t[-1] - t[1]))
            step_times.extend(np.diff(t))
    out["E2E Request Time"] = _stats(e2e)
    out["TTFT Time"] = _stats(ttft)
    if step_times:
        out["Request Generation Step Time"] = _stats(step_times)
        out["Request Generation Tokens Per Second"] = _stats(rates)
    out["Number of Output Tokens"] = _stats(tokens)
    return out


def _write_metric(path, metric):
    # specdec_bench stores each metric file as a list of runs.
    with open(path, "w") as f:
        json.dump([metric], f, indent=4)


def _versions(url):
    versions = {"aiperf_version": None, "server_version": None}
    with contextlib.suppress(importlib.metadata.PackageNotFoundError):
        versions["aiperf_version"] = importlib.metadata.version("aiperf")
    with contextlib.suppress(OSError):
        versions["server_version"] = _get_json(f"{url}/version").get("version")
    return versions


def run_aiperf(args, model, tokenizer, extra, num_sessions, concurrency, artifact_dir):
    cmd = [
        args.aiperf_bin, "profile",
        "--model", model, "--tokenizer", tokenizer, "--url", args.url,
        "--endpoint-type", "chat", "--streaming",
        "--input-file", str(artifact_dir / "input.jsonl"),
        "--custom-dataset-type", "multi_turn",
        "--dataset-sampling-strategy", "sequential",
        # Counts sessions; without it AIPerf sends 10 requests, all first turns.
        "--num-conversations", str(num_sessions),
        "--concurrency", str(concurrency),
        "--extra-inputs", json.dumps(extra),
        "--use-server-token-count",
        "--export-level", "raw",
        "--output-artifact-dir", str(artifact_dir),
        "--ui", "none",
        "--no-gpu-telemetry",
    ]  # fmt: skip
    if args.trust_remote_code:
        cmd.append("--tokenizer-trust-remote-code")
    env = dict(os.environ)
    if os.path.isdir(tokenizer):
        # In HF offline mode AIPerf 0.13 looks every tokenizer up in the Hub cache, local paths too.
        for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
            env.pop(var, None)
    subprocess.run(cmd, check=True, env=env, stdin=subprocess.DEVNULL)
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Server base URL, e.g. http://localhost:8000")
    parser.add_argument("--model", help="Served model name (default: the first in /v1/models)")
    parser.add_argument("--tokenizer", help="Tokenizer for AIPerf (default: the model name)")
    parser.add_argument("--mtbench", help="Shorthand for --dataset mtbench --dataset_path PATH")
    parser.add_argument("--dataset", choices=["mtbench", "specbench", "speed"])
    parser.add_argument(
        "--dataset_path",
        help="JSONL questions, or for speed a split prepared by examples/specdec_bench/prepare_data.py",
    )
    parser.add_argument("--num_requests", type=int, help="Use the first N requests")
    parser.add_argument(
        "--runtime_params",
        help="specdec_bench runtime YAML; its sampling_kwargs and chat_template_args are sent",
    )
    parser.add_argument(
        "--temperature", type=float, nargs="+", help="Overrides sampling_kwargs.temperature"
    )
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1])
    parser.add_argument("--output_length", type=int, default=4096)
    parser.add_argument("--ignore_eos", action="store_true")
    parser.add_argument("--num_gpus", type=int, default=1, help="GPUs serving the model")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--aiperf_bin", help="AIPerf executable (default: next to this Python, else on PATH)"
    )
    parser.add_argument("--save_dir", default="results")
    args = parser.parse_args()
    if args.mtbench:
        args.dataset, args.dataset_path = "mtbench", args.mtbench
    if not (args.dataset and args.dataset_path):
        parser.error("pass --mtbench, or --dataset with --dataset_path")
    if args.aiperf_bin is None:
        sibling = Path(sys.executable).with_name("aiperf")
        args.aiperf_bin = str(sibling) if sibling.exists() else shutil.which("aiperf") or "aiperf"

    runtime = {}
    if args.runtime_params:
        with open(args.runtime_params) as f:
            runtime = yaml.safe_load(f) or {}
    # Same defaults as specdec_bench: greedy without sampling_kwargs, else temperature 1.
    sampling = runtime.get("sampling_kwargs", {"temperature": 0})
    model = args.model or _get_json(f"{args.url}/v1/models")["data"][0]["id"]
    tokenizer = args.tokenizer or model
    requests = load_requests(args.dataset, args.dataset_path, args.num_requests)

    for temperature in args.temperature or [sampling.get("temperature", 1.0)]:
        for concurrency in args.concurrency:
            out = Path(args.save_dir) / f"t{temperature:g}_c{concurrency}"
            artifact_dir = out / "aiperf"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            write_input(requests, artifact_dir / "input.jsonl")
            extra = {
                # Explicit, or vllm serve fills them from the model's generation_config.json.
                "temperature": temperature,
                "top_p": sampling.get("top_p", 1.0),
                "top_k": sampling.get("top_k", 0),
                "max_tokens": args.output_length,
                # Turn 2 replays the turn-1 reply with special tokens, as specdec_bench does.
                "skip_special_tokens": False,
            }
            if runtime.get("chat_template_args"):
                extra["chat_template_kwargs"] = runtime["chat_template_args"]
            if args.ignore_eos:
                extra["ignore_eos"] = True
            cmd = run_aiperf(
                args, model, tokenizer, extra, len(requests), concurrency, artifact_dir
            )

            steps, texts, times, tokens = collect(artifact_dir, requests)
            acceptance = acceptance_metrics(requests, steps, by_index=args.dataset == "mtbench")
            _write_metric(out / "acceptance_rate.json", acceptance)
            _write_metric(out / "timing.json", timing_metrics(times, tokens, args.num_gpus))
            with open(out / RESPONSES_FILE[args.dataset], "w") as f:
                for request, replies in zip(requests, texts):
                    line = {k: request[k] for k in ("question_id", "category", "turns")}
                    line["choices"] = [{"index": 0, "turns": replies}]
                    f.write(json.dumps(line) + "\n")
            config = {
                **vars(args),
                "temperature": temperature,
                "concurrency": concurrency,
                "served_model": model,
                **_versions(args.url),
                # Required by specdec_bench's upload_to_s3.py; set from the env as specdec_bench does.
                "container_image": os.environ.get("CONTAINER_IMAGE") or None,
                "aiperf_command": cmd,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            with open(out / "configuration.json", "w") as f:
                json.dump(config, f, indent=4)
            print(f"{out}: Average_AL {acceptance['Average_AL']:.4f}", flush=True)


if __name__ == "__main__":
    main()
