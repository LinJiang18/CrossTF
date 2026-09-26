# SDForger baseline

SDForger (arXiv:2505.17103) run through its reference implementation, IBM fms-dgt
(`time_series` databuilder), on the seven datasets. See `UPSTREAM.md` for provenance,
changes and the exact settings.

```
baselines/SDForger/
  fms-dgt/            trimmed fms-dgt package (+ pyproject.toml, LICENSE, README.md)
    tasks/public/time_series/<Stem>_multisample/task.yaml   one task per dataset
  prepare_data.py     data/*.npy -> <DGT_DATA_DIR>/public/time_series/*.parquet
  collect_outputs.py  fms-dgt jsonl -> outputs/SDForger/<ds>/SDForger_<ds>_generatedata.npy
  run.sh              prepare + run + collect for all seven datasets
```

## Environment

Tested with Python 3.11, vllm 0.29.0, torch 2.13.0+cu130, transformers 5.17.0, one CUDA GPU.
Do not pin vllm 0.9.2 (the old upstream pin); the tokenizer fallback in `vllm_in_proc.py`
makes current vllm work.

```bash
conda create -n sdforger python=3.11 -y
conda activate sdforger
cd baselines/SDForger/fms-dgt
pip install -e ".[time_series,public,vllm]"
```

Instead of `pip install -e`, putting `baselines/SDForger/fms-dgt` first on `PYTHONPATH`
also works (with the dependencies installed); `run.sh` does this.

GPT-2 is downloaded from the Hugging Face Hub on first use. For machines without network
access, pre-download it (`huggingface-cli download gpt2`, with `HF_HOME` set as you like) and
run with `HF_OFFLINE=1`.

If an import fails with a libstdc++ `CXXABI_...` not found error (e.g. under `sqlite3`
imported by vllm), prepend the env's own lib: `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH`.

## Run

```bash
bash baselines/SDForger/run.sh                          # all seven datasets, seed 42
DATASETS="ecg traffic" bash baselines/SDForger/run.sh   # subset
```

Per dataset this does:

```bash
export DGT_DATA_DIR=$WORK/data DGT_OUTPUT_DIR=$WORK/dgt_output
python baselines/SDForger/prepare_data.py --datasets <ds>
cd baselines/SDForger/fms-dgt
python -m fms_dgt.public --task-paths ./tasks/public/time_series/<Stem>_multisample/task.yaml
python ../collect_outputs.py --dataset <ds> --task-out-dir $DGT_OUTPUT_DIR/time_series/<Stem>_multisample
```

Task stems: building -> `Building`, ecg -> `ECG`, electricity -> `Electricity`,
precipitation -> `precipitation`, temperature -> `temperature`, traffic -> `Traffic`,
transportation -> `transportation`.

- Work dir (`$SDFORGER_WORK`, default `baselines/SDForger/work/`): parquets, the fine-tuned
  GPT-2 checkpoint and jsonl for each dataset (~500 MB per dataset). Safe to delete after
  collecting.
- Output: `outputs/SDForger/<ds>/SDForger_<ds>_generatedata.npy`, float32, shape (N', T, 1),
  original units, plus `summary.json`. N' is 100-103% of N (SDForger's de-duplication and IQR
  filter); it is kept as returned.
- `run.sh` skips datasets whose output already exists; delete it to regenerate.
- Each dataset took 10-36 minutes on a single RTX PRO 6000 GPU.
