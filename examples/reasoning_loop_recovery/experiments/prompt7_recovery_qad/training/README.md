# Recovery QAD training and NVFP4 export

In [training.yaml](training.yaml), `arguments` lists the command-line options passed to
the training script, such as `--train_iters 42`.

Continue the original **fake-quantized QAD checkpoint**, keeping quantizers active. Use its original
BF16 teacher. Train on the fixed generated sequences with input-masked teacher-logit KL loss,
top-k 4,096, and loss scale 1. The `--sft` flag selects the data-loading path; language-model
cross-entropy loss is skipped in this QAD recipe.

The successful run used four GB300 GPUs with TP=1, EP=4, PP=1, and CP=1. It ran 42 iterations at
GBS=8, MBS=1, sequence length 32,768, learning rate 1e-6, no warmup, and seed 20261018.
Activation recomputation was full/uniform with one layer per recompute group. Auxiliary MoE loss
was disabled. Export iteration 42 to unified NVFP4 Hugging Face format before evaluation.

[training.yaml](training.yaml) records the command-line options used in the completed run, with local paths replaced by
`${VARIABLE}` placeholders. Supply the BF16 architecture/tokenizer, original fake-quantized checkpoint,
prepared training directory, output directory, and export directory in your own workspace.

## Runtime compatibility

The successful training environment used NeMo 26.08.00 and an experiment-local distillation script.
It also corrected Transformer Engine 2.17.1's SM103 FlashAttention eligibility for head dimension 256
and bounded Adam tensor batches. Check backend selection and checkpoint/quantizer restoration in a
small run before training with a different runtime.

Current ModelOpt `main` exposes `--logit_kl_top_k` instead of the historical `--logit_kl_topk`, and
documents a residual probability bucket in the top-k loss. Other distillation arguments have also
changed. Loss equivalence and an end-to-end rerun on current `main` have not been validated.
Use the [Megatron Bridge example](../../../../megatron_bridge/README.md) as the current API reference;
retain the recorded settings when validating a port of this experiment.
