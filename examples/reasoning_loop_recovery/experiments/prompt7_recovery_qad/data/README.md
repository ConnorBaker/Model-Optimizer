# Create the training dataset

The `data/` folder contains settings for generating responses and a manifest describing the
resulting training dataset:

| File | Purpose |
| --- | --- |
| [data_generation.yaml](data_generation.yaml) | How candidate responses were collected: checkpoint, prompt 7, sampling settings, and seeds |
| [dataset_manifest.yaml](dataset_manifest.yaml) | How collected responses were selected and formatted, plus the identity and checksum of the resulting 334-example dataset |

The manifest references the generation settings to connect the final dataset to its source.
Neither file contains the generated responses or runs the preparation process. The
[exact prompt 7](prompt7.txt) is also stored in `data/`.

## Create the training / held-out split

1. Start with the 55 SciCode subtasks, across 38 parent problems, that previously hit the output cap
   with the original QAD checkpoint.
2. Select 11 held-out subtasks using split seed **20261018**, subject to the ordering rule below.
3. Assign the remaining 44 subtasks to training.
4. Save the exact lists in [subtask_split.yaml](subtask_split.yaml) before generating responses.

Within a parent problem, held-out subtasks must come after every training target from that parent.
For example, **12.2 is a training subtask and 12.3 is held out**. This prevents a held-out target
answer from becoming an earlier-step input to a training example. This rule concerns ordering,
not difficulty; related subtasks can still share a parent across the split.

The assignment is by subtask, so all ten attempts for a subtask stay together. The 44 training
subtasks supply **440 candidate responses**, which are filtered to **334 training examples**.
The 11 held-out subtasks supply 110 responses that are excluded from training. Post-training
evaluation uses fresh seeds and the same split to report training and held-out results separately.

## Generate candidate responses

Start from the **NVFP4 QAD Qwen3.6-35B-A3B checkpoint after 500 QAD iterations**. Generate ten
attempts for each of 55 SciCode subtasks with observed original-QAD cap hits, using seeds
60000–60009. This produces 550 target responses, of which 440 belong to training subtasks.

Append two newline characters and the following text to the original user message:

> If your reasoning begins repeating, summarize the unresolved issue once, resolve it as best you can, and continue toward a final answer.

Use temperature 0.6, top-p 0.95, thinking enabled, a 131,072-token completion cap, and a 262,144-token
server context. The server uses vLLM's `qwen3` reasoning parser. Keep the checkpoint's default KV-cache
format. No presence penalty was requested.

Each checkpoint generates its own preceding steps using the same seed and prompt variant. Supplied
reference steps are prefilled. Save prerequisite responses but exclude them from target counts.
Fail an affected chain if its prompt plus output cap would exceed context; do not lower the cap.
Preserve completed attempts when resuming, including failures.

## Select and format training responses

Use only the 440 training-target responses. Apply these filters in order:

1. Reject cap hits: finish reason `length` or completion tokens at least 131,072.
2. Reject an empty final answer or a finish reason other than `stop`.
3. Reject a saved reasoning-repetition score at least 0.5.
4. Reject any exact, case-sensitive sequence of 20 whitespace-separated words occurring eight or
   more times across reasoning and final-answer text.
5. Remove prompt 7 from the user message and serialize the original message with the thinking-enabled
   tokenizer chat template. The input ends with `<think>\n`.
6. Form the output as reasoning, `\n</think>\n\n`, and the final answer. Reject sequences exceeding
   32,768 tokens, including the input and one EOS token.
7. Deduplicate by SHA-256 of `input + NUL + output`, preserving original candidate order.

The resulting UTF-8 JSONL contains `input` and `output` string fields; it is not tokenized. EOS is
accounted for by dataset processing rather than embedded in the output string. The preparation
runtime used Transformers 5.20.1. The source repetition score and its implementation must also be
preserved when regenerating candidates; changing that detector changes the selection.

Selection retained **334/440 candidates (75.91%)**, representing all 44 training subtasks:

- 66 rejected for a repeated 20-word cycle.
- 31 rejected for hitting the cap.
- 8 rejected for high reasoning repetition.
- 1 rejected for exceeding the training sequence length.

Correctness was recorded separately: **51/334 accepted responses passed SciCode tests**. Correctness
was not a selection criterion. Held-out target responses were not used for training.

The **dataset manifest describes the training data; it does not contain the generated responses**.
It records which checkpoint and prompt produced the candidates, the filters that selected the
334 examples, the JSONL format, the file size, and its SHA-256 checksum. Use it to identify the
dataset used in this experiment and verify that a local copy is identical. It is metadata, not
a training configuration.

Matching counts alone does not establish byte-identical data; serialization, ordering, tokenizer
behavior, and source responses matter.
