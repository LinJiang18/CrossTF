# Upstream

- **Method:** SDForger -- "SDForger: Synthetic Data Forger for time series via LLMs"
  (arXiv:2505.17103). Its reference implementation is the `time_series` databuilder of
  IBM's Data Generation and Transformation framework, **fms-dgt**.
- **Code:** https://github.com/IBM/fms-dgt (Apache-2.0; see `LICENSE`, SPDX headers kept).
- **Upstream commit:** not identifiable -- the copy we worked from was vendored without git
  history.
- **Vendored:** `fms-dgt/`, trimmed to the files on the run path of
  `python -m fms_dgt.public` with the `time_series` databuilder: the framework core
  (`fms_dgt/{base,log}`, `generate_data.py`, `index.py`, ...), the in-process vLLM block, the
  trainer block, the default dataloader/datastore/native formatter, and
  `fms_dgt/public/databuilders/time_series/`. fms-dgt discovers registered components by
  scanning `fms_dgt/core` and `fms_dgt/public` source text, so removed modules (other
  databuilders, API-based LLM providers, vLLM server mode, validators, tests, docs) are
  simply never imported. `pyproject.toml` is unchanged from the vendored copy.

## Changes vs upstream

1. `fms_dgt/core/blocks/llm/vllm/vllm_in_proc.py` -- fall back to
   `transformers.AutoTokenizer.from_pretrained` when `vllm.transformers_utils.tokenizer.get_tokenizer`
   is missing (removed in vllm > 0.9.x); needed to run on a current vllm/torch stack.
2. `fms_dgt/public/databuilders/time_series/generate.py` -- `gc.collect()` +
   `torch.cuda.empty_cache()` between fine-tuning and vLLM start-up, so the in-process vLLM
   engine finds enough free GPU memory. No effect on the algorithm.
3. `fms_dgt/public/databuilders/time_series/time_series.yaml` -- `gpu_memory_utilization: 0.10`
   for the vLLM target (memory head-room for same-process train -> infer).
4. `fms-dgt/tasks/public/time_series/<Stem>_multisample/task.yaml` -- one task per dataset
   (new files; not upstream), see settings below.
5. Our own scripts: `prepare_data.py` (npy -> parquet in the multisample layout),
   `collect_outputs.py` (jsonl -> npy), `run.sh`.

## Settings used for the paper

All upstream defaults of `time_series.yaml` except item 3 above:

- Base LM `gpt2`, learning rate 8e-5, 100 epochs, per-device batch 32, **seed 42**.
- vLLM target: dtype float32, temperature 1.3, max_tokens 1000.
- Per task (`augmentation_strategy: multisample`, channel `TOY-`):
  `train_length = T`, `train_samples = N`, `min_outputs_to_generate = max_outputs_to_generate = N`,
  `inference_batch 64`, `norms_diversity_threshold 0` (diversity early-stop off so generation
  reaches N), `embedding_type fica`, `embedding_dim auto`, `variance_explained 0.7`,
  `min_windows_number 30`, `min_windows_length = T`, `input_tokens_precision 4`, `k_bit null`.
- Data given raw (no external scaling); SDForger standard-scales internally and inverts before
  returning, so outputs are in original units.
- **Output count:** SDForger de-duplicates and applies an IQR filter to its accepted samples and
  returns slightly more than the requested minimum; on our datasets N' was 100-103% of N. The
  generated set is saved as returned (no padding, no truncation).
- One GPU, one run (seed 42) per dataset.
