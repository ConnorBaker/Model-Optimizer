# Evaluate without prompt 7

Reuse the [dataset split](../data/subtask_split.yaml) to report results separately for the 44 training
subtasks and 11 held-out subtasks.

The [evaluation settings](evaluation.yaml) record the comparison protocol and metric definitions.

Evaluation uses a **custom Python SciCode runner**, not `lm_eval` or NeMo Evaluator Launcher (NEL).
It builds requests from the SciCode prompt templates and task definitions, sends them to a vLLM
OpenAI-compatible chat-completions endpoint, and checks the generated final-answer code against
SciCode's official tests. It also saves reasoning, final-answer content, finish reason, token usage,
and repetition metrics for each response.

Compare original QAD and exported recovery QAD on the same 55 subtasks, using the **same ten seeds
for each subtask and each checkpoint** (61000–61009), with the original SciCode prompt and no presence
penalty. This is what a matched evaluation means here: subtask IDs, seeds, sampling settings, output
cap, and context limit are held constant between the two checkpoints. Keep the generation
cap, context, and sampling settings unchanged. Retain all 550 target attempts per checkpoint.
Prerequisites are generated independently for each checkpoint, so matched seeds do not imply
identical target prompts when earlier generated code differs.

Correctness comes from the official SciCode tests. Completion-token counts include reasoning.
A cap hit and a missing final answer are overlapping metrics; a response may hit the cap while
already containing final-answer content. A cap hit alone does not prove repetitive reasoning.
