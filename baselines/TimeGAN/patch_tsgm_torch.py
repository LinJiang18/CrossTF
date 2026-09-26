"""Let tsgm's TimeGAN train on a CUDA GPU under the Keras torch backend.

Run once per environment, after `pip install tsgm==0.1.0` and before the first training
run. It edits the installed `tsgm/models/timeGAN.py` in place; re-run it after
reinstalling tsgm. Idempotent: running it twice is a no-op.

    python patch_tsgm_torch.py            # apply
    python patch_tsgm_torch.py --check    # report only; exit 1 if anything is unpatched

tsgm 0.1.0 has three faults on the torch path; the edits are minimal and do not touch the
model, its losses or its training schedule:

1. `self._device = "cpu"` is set unconditionally (a workaround for Apple MPS whose MPS
   branch is guarded, but this assignment is not), so every sub-model is moved to the CPU
   even when CUDA is available. It now becomes "cuda" when the torch backend is active and
   CUDA is present; the MPS branch is left as it was. Without CUDA nothing changes.
2. `float(np.sqrt(x))` -> `np.sqrt(float(x))` at the five sites that log a loss: numpy
   cannot take the sqrt of a torch tensor that still requires grad, while `float()`
   first calls `.item()`.
3. `generate()` stacks the generator's output with `np.vstack` straight off the device,
   which fails for CUDA tensors; the records are detached and moved to host first.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

MARK = '# --- patch: allow CUDA under the torch backend'

OLD_DEV = '''        # Configure device for PyTorch backend to avoid MPS broadcasting issues
        self._device = "cpu"  # Force CPU for stable training
        if _has_torch and os.environ.get("KERAS_BACKEND") == "torch":'''
NEW_DEV = '''        # Configure device for PyTorch backend to avoid MPS broadcasting issues
        ''' + MARK + '''
        # Upstream sets this to "cpu" unconditionally; the MPS workaround below is
        # guarded, this line was not, so CUDA machines lost the GPU too.
        self._device = "cpu"  # Force CPU for stable training
        if _has_torch and os.environ.get("KERAS_BACKEND") == "torch" \\
                and torch.cuda.is_available():
            self._device = "cuda"
        if _has_torch and os.environ.get("KERAS_BACKEND") == "torch":'''
DEV_DONE = '''                and torch.cuda.is_available():
            self._device = "cuda"'''

GEN_OLD = '''            records = self.generator(Z_)
            data.append(records)
        return np.array(np.vstack(data))[:n_samples]'''
GEN_NEW = '''            records = self.generator(Z_)
            # patch: on the torch backend these live on the GPU and np.vstack below cannot
            # read device memory. Detach first -- the generator output still carries grad.
            if hasattr(records, "detach"):
                records = records.detach().cpu().numpy()
            data.append(records)
        return np.array(np.vstack(data))[:n_samples]'''
GEN_DONE = 'records = records.detach().cpu().numpy()'

SQRT = [('self.training_losses_history["generator_s"] = float(np.sqrt(step_g_loss_s))',
         'self.training_losses_history["generator_s"] = np.sqrt(float(step_g_loss_s))'),
        ('self.training_losses_history["embedder"] = float(np.sqrt(step_e_loss_t0))',
         'self.training_losses_history["embedder"] = np.sqrt(float(step_e_loss_t0))'),
        ('g_loss_s: {np.round(np.sqrt(step_g_loss_s), 4)},',
         'g_loss_s: {np.round(np.sqrt(float(step_g_loss_s)), 4)},'),
        ('e_loss_t0: {np.round(np.sqrt(step_e_loss_t0), 4)}"""',
         'e_loss_t0: {np.round(np.sqrt(float(step_e_loss_t0)), 4)}"""'),
        ('f"step: {epoch}/{epochs}, s_loss: {np.round(np.sqrt(loss_val), 4)}"',
         'f"step: {epoch}/{epochs}, s_loss: {np.round(np.sqrt(float(loss_val)), 4)}"')]


def target() -> Path:
    import tsgm.models.timeGAN as m
    return Path(m.__file__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--check', action='store_true', help='report only, change nothing')
    ns = ap.parse_args()
    f = target()
    s = f.read_text()
    dev_done = DEV_DONE in s
    gen_done = GEN_DONE in s
    pending = [(a, b) for a, b in SQRT if a in s]
    print(f'target: {f}')
    print(f'device patch: {"already applied" if dev_done else "to apply"}')
    print(f'np.sqrt sites still to fix: {len(pending)} of {len(SQRT)}')
    print(f'generate() device patch: {"already applied" if gen_done else "to apply"}')
    if ns.check:
        sys.exit(0 if dev_done and gen_done and not pending else 1)
    if not dev_done:
        assert OLD_DEV in s, 'device block not found -- tsgm changed, re-read timeGAN.py'
        s = s.replace(OLD_DEV, NEW_DEV)
    if not gen_done:
        assert GEN_OLD in s, 'generate() block not found -- tsgm changed, re-read timeGAN.py'
        s = s.replace(GEN_OLD, GEN_NEW)
    for a, b in SQRT:
        s = s.replace(a, b)
    f.write_text(s)
    print('patched')


if __name__ == '__main__':
    main()
