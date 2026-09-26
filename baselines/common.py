"""Paths shared by every baseline runner.

All methods read the same seven datasets from `<repo>/data/` and write their generated
sets to `<repo>/outputs/<Method>/[seed<N>/]<dataset>/<Method>_<dataset>_generatedata.npy`,
in the original (un-normalised) units and with the same shape (N, T, 1) as the real data.
`evaluation/evaluate.py` reads them from there.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / 'data'
OUTPUT_ROOT = REPO_ROOT / 'outputs'

DATASETS = {
    'building': 'Building_seq_144_sample_1838.npy',
    'ecg': 'ECG_seq_250_sample_1000.npy',
    'electricity': 'Electricity_seq_144_sample_1096.npy',
    'precipitation': 'precipitation_seq_90_sample_1056.npy',
    'temperature': 'temperature_seq_144_sample_1520.npy',
    'traffic': 'Traffic_seq_144_sample_1790.npy',
    'transportation': 'transportation_seq_144_sample_1320.npy',
}


def data_path(name: str) -> Path:
    """Absolute path of the (N, T, 1) .npy file of dataset `name`."""
    if name not in DATASETS:
        raise KeyError(f'unknown dataset {name!r}; choose from {sorted(DATASETS)}')
    return DATA_DIR / DATASETS[name]


def output_dir(method: str, name: str, seed: int | None = None) -> Path:
    """`<repo>/outputs/<method>/[seed<N>/]<name>/`, created if missing."""
    d = OUTPUT_ROOT / method
    if seed is not None:
        d = d / f'seed{seed}'
    d = d / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def generated_path(method: str, name: str, seed: int | None = None) -> Path:
    return output_dir(method, name, seed) / f'{method}_{name}_generatedata.npy'
