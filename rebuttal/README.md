# Reproducing Table 1

**Table 1: Extreme-sensitive cross-view asymmetry across seven datasets.**

| Dataset | Loss | Overall | Extreme | Ext./Loss | Ext./Overall |
|---|---:|---:|---:|---:|---:|
| Traffic | 4.5% | 8.4% | 290.4% | 65.22× | 34.44× |
| Transportation | 7.5% | 51.1% | 254.2% | 34.11× | 4.97× |
| Temperature | 7.3% | 12.6% | 145.6% | 19.91× | 11.52× |
| Electricity | 8.7% | 78.4% | 85.2% | 9.82× | 1.09× |
| Precipitation | 33.3% | 339.9% | 160.3% | 4.81× | 0.47× |
| ECG | 16.7% | 3.9% | 38.2% | 2.29× | 9.93× |
| Building | 12.9% | 10.8% | 22.8% | 1.77× | 2.10× |
| **Median** | 8.7% | 12.6% | 145.6% | **16.74×** | **11.56×** |

Loss, Overall and Extreme are relative max–min ranges over α ∈ [0, 1].

## Environment

One NVIDIA GPU with CUDA (tested on an RTX PRO 6000), Python 3.10:

```bash
conda create -n crosstf python=3.10 -y && conda activate crosstf
# match the index URL to your CUDA version
pip install torch==2.13.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip install numpy==2.2.6 scipy==1.15.3 omegaconf==2.3.1 pyyaml scikit-learn
```

The checkpoints (`rebuttal/pretrained/`, 28 files) and the data (`data/`) are in the
repository; nothing else needs to be downloaded.

## Run

From the repository root:

```bash
# about 15 minutes
python rebuttal/reproduce_Table1.py
```

The script prints Table 1 in the format above.
