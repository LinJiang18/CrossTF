import os
import time
import torch
import numpy as np
from Data.build_dataloader import unnormalize_sample

class Logger:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        log_path = os.path.join(save_dir, "log.txt")
        self.log_file = open(log_path, "a")

    def log_info(self, msg):
        """
        Log text information with timestamp.
        """
        time_str = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{time_str}] {msg}"
        print(line)
        self.log_file.write(line + "\n")
        self.log_file.flush()

    def add_scalar(self, tag, scalar_value, global_step):
        """
        Log scalar values (e.g., loss) to text log.
        """
        time_str = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{time_str}] {tag} | step={global_step} | value={scalar_value:.6f}"
        self.log_file.write(line + "\n")
        self.log_file.flush()

    def save_config(self, config):
        import yaml
        path = os.path.join(self.save_dir, "config.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(config, f)

    def close(self):
        self.log_file.close()


class DecompTrace:
    def __init__(self, dtype="float16"):
        self.dtype = dtype
        self.data = {}  # t -> list[{"img": np.ndarray, "x_start": np.ndarray}]

    @torch.no_grad()
    def log(self, t, img, x_start):
        t_int = int(t.flatten()[0].item()) if torch.is_tensor(t) else int(t)

        img = img.detach().cpu().numpy()
        xs = x_start.detach().cpu().numpy()

        if self.dtype is not None:
            img = img.astype(self.dtype, copy=False)
            xs = xs.astype(self.dtype, copy=False)

        if t_int not in self.data:
            self.data[t_int] = []
        self.data[t_int].append({"img": img, "x_start": xs})



def save_trace_npz(trace, path, normalizer=None):
    ts = sorted(trace.data.keys())
    if len(ts) == 0:
        raise ValueError("trace.data is empty")

    # number of logged batches (scheme A)
    n_batches = len(trace.data[ts[0]])
    if n_batches == 0:
        raise ValueError(f"trace.data[{ts[0]}] is an empty list")

    # sanity: each t should have the same number of batches
    for t in ts:
        if len(trace.data[t]) != n_batches:
            raise ValueError(
                f"Inconsistent number of batches at t={t}: "
                f"{len(trace.data[t])} vs {n_batches} (at t={ts[0]})"
            )

    img_blocks = []
    xs_blocks = []

    # reorganize: for each batch k, stack over time -> (B, T, L, D)
    for k in range(n_batches):
        imgs_t = []
        xs_t = []
        for t in ts:
            img = trace.data[t][k]["img"]       # (B, L, D) or (B, L)
            xs  = trace.data[t][k]["x_start"]   # (B, L, D) or (B, L)

            # ensure shape (B, L, C) for unnormalize_sample
            if img.ndim == 2:
                img = img[..., None]  # -> (B, L, 1)
            if xs.ndim == 2:
                xs = xs[..., None]    # -> (B, L, 1)

            # optional: unnormalize using provided normalizer
            if normalizer is not None:
                img = unnormalize_sample(img, normalizer)  # (B, L, C)
                xs = unnormalize_sample(xs, normalizer)    # (B, L, C)

                # keep trace dtype if specified
                if trace.dtype is not None:
                    img = img.astype(trace.dtype, copy=False)
                    xs = xs.astype(trace.dtype, copy=False)

            imgs_t.append(img)
            xs_t.append(xs)

        img_k = np.stack(imgs_t, axis=1)  # (B, T, L, D)
        xs_k  = np.stack(xs_t, axis=1)    # (B, T, L, D)

        img_blocks.append(img_k)
        xs_blocks.append(xs_k)

    # flatten batches into one big batch dimension -> (n_batches*B, T, L, D)
    img_4d = np.concatenate(img_blocks, axis=0)
    xs_4d  = np.concatenate(xs_blocks, axis=0)

    np.savez_compressed(
        path,
        t=np.array(ts, dtype=np.int32),
        img=img_4d,
        x_start=xs_4d,
    )


def load_trace_npz(path):
    z = np.load(path, allow_pickle=False)

    ts = z["t"].astype(int)
    order = np.argsort(ts)
    ts = ts[order]

    img = z["img"]
    x_start = z["x_start"]

    # Ensure shape (N, T, L, D)
    if img.ndim == 3:
        img = img[..., None]
    if x_start.ndim == 3:
        x_start = x_start[..., None]

    # Reorder time dimension by sorted t
    img = img[:, order, :, :]
    x_start = x_start[:, order, :, :]

    return ts, img, x_start
