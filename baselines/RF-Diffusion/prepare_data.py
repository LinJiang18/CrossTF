"""Convert the (N, T, 1) .npy datasets into the per-series .mat files RF-Diffusion reads.

`WiFiDataset` collects `user*.mat` and reads two keys from each: `feature`, the complex
signal, and `cond`, the condition. So one .mat per series, named `user_<i>.mat`:

    feature  (T, 1) complex64   the real series, imaginary part zero
    cond     (1, 6) complex64   all zeros -- constant condition, see README (deviation 1)

Values are written raw. The collator normalises each sample at load time
(`(x - x.mean()) / x.std()` over the stacked real/imag tensor), so nothing is normalised
here; `stats.npz` records each series' original mean and std so the generated side can be
put back on a real scale -- see README (deviation 5).

Files go to `$RFDIFF_WORK/dataset/<dataset>/` (default: `baselines/RF-Diffusion/work/`).

    python prepare_data.py                              # all seven datasets
    python prepare_data.py --datasets precipitation
"""
from __future__ import annotations
import argparse, glob, os, shutil, sys
from pathlib import Path
import numpy as np
import scipy.io as scio

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common                                             # noqa: E402

WORK = Path(os.environ.get('RFDIFF_WORK', HERE / 'work'))
OUT_ROOT = WORK / 'dataset'
COND_DIM = 6


def convert(dataset: str, force: bool = False) -> tuple[int, int, int]:
    arr = np.load(common.data_path(dataset)).astype(np.float64)      # (N, T, 1)
    n, t, c = arr.shape
    assert c == 1, f'{dataset}: expected one channel, got {c}'
    out = OUT_ROOT / dataset / 'raw'
    if out.is_dir() and not force:
        have = len(glob.glob(str(out / 'user*.mat')))
        prev = OUT_ROOT / dataset / 'stats.npz'
        if have and prev.is_file():
            return n, t, int(np.load(prev)['n_skipped'])
        shutil.rmtree(out)
    elif out.is_dir():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    # The collator divides by the per-sample std of the stacked (real, imag) tensor. With
    # the imaginary part identically zero that std is zero only for an all-zero series
    # (precipitation has one), which would turn the whole batch to NaN. Such a series
    # carries nothing to learn, so it is left out of the training set.
    keep = np.abs(arr[:, :, 0]).max(axis=1) > 0
    n_skipped = int((~keep).sum())
    cond = np.zeros((1, COND_DIM), dtype=np.complex64)
    for i in np.flatnonzero(keep):
        series = arr[i, :, 0]
        scio.savemat(out / f'user_{i:05d}.mat',
                     {'feature': series.astype(np.complex64).reshape(t, 1), 'cond': cond},
                     do_compression=False)
    # Per-series mean/std in original units. The collator discards these; the generated
    # side borrows them back, otherwise every sample would come out on an arbitrary scale.
    np.savez(OUT_ROOT / dataset / 'stats.npz',
             mean=arr[:, :, 0].mean(axis=1), std=arr[:, :, 0].std(axis=1),
             n_skipped=n_skipped)
    return n, t, n_skipped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--datasets', nargs='*', default=None, choices=sorted(common.DATASETS))
    ap.add_argument('--force', action='store_true')
    ns = ap.parse_args()
    names = ns.datasets or sorted(common.DATASETS)
    print(f'{"dataset":<16}{"N":>7}{"T":>5}{"skipped":>9}  ->')
    for d in names:
        n, t, sk = convert(d, ns.force)
        print(f'{d:<16}{n:>7}{t:>5}{sk:>9}  {OUT_ROOT / d / "raw"}')


if __name__ == '__main__':
    main()
