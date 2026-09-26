import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def _check_shape(data: np.ndarray) -> None:
    if data.ndim != 3:
        raise ValueError(
            f"Expected array with ndim=3 (N, T, 1), got shape {data.shape}"
        )
    if data.shape[-1] != 1:
        raise ValueError(
            f"Expected last dimension 1 (univariate), got shape {data.shape}"
        )


class NpySeriesDataset(Dataset):
    """
    Loads a single .npy file of shape (N, T, 1).
    __getitem__ returns a tensor (T, 1) in the same layout as the file.
    """

    def __init__(self, data: np.ndarray):
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        _check_shape(data)
        self.data = data.astype(np.float32, copy=False)
        self.N, self.T, _ = self.data.shape

    @classmethod
    def from_path(cls, path: str) -> "NpySeriesDataset":
        data = np.load(path, allow_pickle=False)
        if isinstance(data, np.lib.npyio.NpzFile):
            raise ValueError(
                "Expected a single array .npy file, not .npz. Save as one array of shape (N, T, 1)."
            )
        return cls(np.asarray(data))

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(np.array(self.data[idx], copy=True))


def collate_unconditional(batch):
    """Stack to (B, T, 1), then convert to internal (B, 1, T) and build time grid (B, T)."""
    x = torch.stack(batch, dim=0)  # (B, T, 1)
    B, T, _ = x.shape
    observed_data = x.permute(0, 2, 1).contiguous()  # (B, 1, T)
    timepoints = torch.linspace(0.0, 1.0, T).view(1, T).expand(B, -1).clone()
    return {"observed_data": observed_data, "timepoints": timepoints}


def get_dataloader_npy(
    data_path: str,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    ds = NpySeriesDataset.from_path(data_path)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_unconditional,
        pin_memory=pin_memory,
    )
