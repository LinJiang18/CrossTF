import os
import torch
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

from Models.interpretable_diffusion.model_utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from Utils.masking_utils import noise_mask


class PreprocessedNpyDataset(Dataset):
    """Dataset for pre-segmented 3D time series stored as (M, N, C) .npy files."""

    def __init__(
        self,
        name,
        data_path,
        window=24,
        proportion=0.8,
        save2npy=True,
        neg_one_to_one=True,
        seed=123,
        period='train',
        output_dir='./OUTPUT',
        predict_length=None,
        missing_ratio=None,
        style='separate',
        distribution='geometric',
        mean_mask_length=3,
    ):
        super().__init__()
        assert period in ['train', 'test'], 'period must be train or test.'
        if period == 'train':
            assert not (predict_length is not None or missing_ratio is not None), ''

        self.name = name
        self.pred_len = predict_length
        self.missing_ratio = missing_ratio
        self.style = style
        self.distribution = distribution
        self.mean_mask_length = mean_mask_length
        self.period = period
        self.save2npy = save2npy
        self.auto_norm = neg_one_to_one

        self.dir = os.path.join(output_dir, 'samples')
        os.makedirs(self.dir, exist_ok=True)

        rawdata = np.load(data_path).astype(np.float32)
        if rawdata.ndim != 3:
            raise ValueError(
                f'Expected 3D array with shape (M, N, C), got {rawdata.shape} from {data_path}'
            )

        num_samples, seq_len, var_num = rawdata.shape
        if seq_len != window:
            raise ValueError(
                f'Sequence length mismatch: data has N={seq_len}, config window={window}'
            )

        self.window = seq_len
        self.var_num = var_num
        self.rawdata = rawdata

        train_data, test_data = self.divide(rawdata, proportion, seed)

        self.scaler = MinMaxScaler()
        self.scaler.fit(train_data.reshape(-1, self.var_num))

        train_norm = self._normalize_array(train_data)
        test_norm = self._normalize_array(test_data) if test_data.shape[0] > 0 else test_data

        if self.save2npy:
            self._save_samples(train_data, test_data, train_norm, test_norm, proportion)

        self.samples = train_norm if period == 'train' else test_norm
        self.sample_num = self.samples.shape[0]

        if period == 'test':
            if missing_ratio is not None:
                self.masking = self.mask_data(seed)
            elif predict_length is not None:
                masks = np.ones(self.samples.shape)
                masks[:, -predict_length:, :] = 0
                self.masking = masks.astype(bool)
            else:
                raise NotImplementedError()

    @staticmethod
    def divide(data, ratio, seed=2023):
        size = data.shape[0]
        st0 = np.random.get_state()
        np.random.seed(seed)

        train_num = int(np.ceil(size * ratio))
        indices = np.arange(size)
        train_data = data[indices[:train_num]]
        test_data = data[indices[train_num:]]

        np.random.set_state(st0)
        return train_data, test_data

    def _normalize_array(self, data):
        flat = data.reshape(-1, self.var_num)
        normalized = self.scaler.transform(flat)
        if self.auto_norm:
            normalized = normalize_to_neg_one_to_one(normalized)
        return normalized.reshape(data.shape)

    def _unnormalize_flat(self, data):
        if self.auto_norm:
            data = unnormalize_to_zero_to_one(data)
        return self.scaler.inverse_transform(data)

    def unnormalize(self, samples):
        flat = samples.reshape(-1, self.var_num)
        restored = self._unnormalize_flat(flat)
        return restored.reshape(samples.shape)

    def _save_samples(self, train_data, test_data, train_norm, test_norm, proportion):
        np.save(
            os.path.join(self.dir, f'{self.name}_ground_truth_{self.window}_train.npy'),
            train_data,
        )
        if 1 - proportion > 0:
            np.save(
                os.path.join(self.dir, f'{self.name}_ground_truth_{self.window}_test.npy'),
                test_data,
            )

        if self.auto_norm:
            train_save = unnormalize_to_zero_to_one(train_norm)
            test_save = unnormalize_to_zero_to_one(test_norm) if 1 - proportion > 0 else None
        else:
            train_save = train_norm
            test_save = test_norm if 1 - proportion > 0 else None

        np.save(
            os.path.join(self.dir, f'{self.name}_norm_truth_{self.window}_train.npy'),
            train_save,
        )
        if test_save is not None:
            np.save(
                os.path.join(self.dir, f'{self.name}_norm_truth_{self.window}_test.npy'),
                test_save,
            )

    def mask_data(self, seed=2023):
        masks = np.ones_like(self.samples)
        st0 = np.random.get_state()
        np.random.seed(seed)

        for idx in range(self.samples.shape[0]):
            x = self.samples[idx, :, :]
            mask = noise_mask(
                x,
                self.missing_ratio,
                self.mean_mask_length,
                self.style,
                self.distribution,
            )
            masks[idx, :, :] = mask

        if self.save2npy:
            np.save(os.path.join(self.dir, f'{self.name}_masking_{self.window}.npy'), masks)

        np.random.set_state(st0)
        return masks.astype(bool)

    def __getitem__(self, ind):
        x = self.samples[ind, :, :]
        if self.period == 'test':
            m = self.masking[ind, :, :]
            return torch.from_numpy(x).float(), torch.from_numpy(m)
        return torch.from_numpy(x).float()

    def __len__(self):
        return self.sample_num
