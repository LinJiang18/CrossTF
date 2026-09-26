from abc import ABC, abstractmethod
import torch
import torchaudio.transforms as T
from utils.utils_data import MinMaxScaler, MinMaxArgs


class TsImgEmbedder(ABC):
    """
    Abstract class for transforming time series to images and vice versa
    """

    def __init__(self, device, seq_len):
        self.device = device
        self.seq_len = seq_len

    @abstractmethod
    def ts_to_img(self, signal):
        """

        Args:
            signal: given time series

        Returns:
            image representation of the signal

        """
        pass

    @abstractmethod
    def img_to_ts(self, img):
        """

        Args:
            img: given generated image

        Returns:
            time series representation of the generated image
        """
        pass


class DelayEmbedder(TsImgEmbedder):
    """
    Delay embedding transformation
    """

    def __init__(self, device, seq_len, delay, embedding):
        super().__init__(device, seq_len)
        self.delay = delay
        self.embedding = embedding
        self.img_shape = None

    def pad_to_square(self, x, mask=0):
        """
        Pads the input tensor x to make it square along the last two dimensions.
        """
        _, _, cols, rows = x.shape
        max_side = max(cols, rows)
        padding = (
            0, max_side - rows, 0, max_side - cols)  # Padding format: (pad_left, pad_right, pad_top, pad_bottom)

        # Padding the last two dimensions to make them square
        x_padded = torch.nn.functional.pad(x, padding, mode='constant', value=mask)
        return x_padded

    def unpad(self, x, original_shape):
        """
        Removes the padding from the tensor x to get back to its original shape.
        """
        _, _, original_cols, original_rows = original_shape
        return x[:, :, :original_cols, :original_rows]

    def num_cols(self):
        """Number of delay-embedding columns; same while-loop logic as ts_to_img()."""
        i = 0
        while (i * self.delay + self.embedding) <= self.seq_len:
            i += 1
        if i * self.delay != self.seq_len and i * self.delay + self.embedding > self.seq_len:
            i += 1
        return i

    def ts_to_img(self, signal, pad=True, mask=0):

        batch, length, features = signal.shape
        #  if our sequences are of different lengths, this can happen with physionet and climate datasets
        if self.seq_len != length:
            self.seq_len = length

        x_image = torch.zeros((batch, features, self.embedding, self.embedding))
        i = 0
        while (i * self.delay + self.embedding) <= self.seq_len:
            start = i * self.delay
            end = start + self.embedding
            x_image[:, :, :, i] = signal[:, start:end].permute(0, 2, 1)
            i += 1

        ### SPECIAL CASE
        if i * self.delay != self.seq_len and i * self.delay + self.embedding > self.seq_len:
            start = i * self.delay
            end = signal[:, start:].permute(0, 2, 1).shape[-1]
            # end = start + (self.embedding - 1) - missing_vals
            x_image[:, :, :end, i] = signal[:, start:].permute(0, 2, 1)
            i += 1

        # cache the shape of the image before padding
        self.img_shape = (batch, features, self.embedding, i)
        x_image = x_image.to(self.device)[:, :, :, :i]

        if pad:
            x_image = self.pad_to_square(x_image, mask)

        return x_image

    def img_to_ts(self, img):
        # Derive non-square shape from current img + delay params (no stale batch cache).
        batch, channels = img.shape[0], img.shape[1]
        num_cols = self.num_cols()
        img_shape = (batch, channels, self.embedding, num_cols)
        self.img_shape = img_shape
        img_non_square = self.unpad(img, img_shape)

        batch, channels, rows, cols = img_non_square.shape

        reconstructed_x_time_series = torch.zeros((batch, channels, self.seq_len), device=img.device, dtype=img.dtype)

        for i in range(cols - 1):
            start = i * self.delay
            end = start + self.embedding
            reconstructed_x_time_series[:, :, start:end] = img_non_square[:, :, :, i]

        ### SPECIAL CASE
        start = (cols - 1) * self.delay
        end = reconstructed_x_time_series[:, :, start:].shape[-1]
        reconstructed_x_time_series[:, :, start:] = img_non_square[:, :, :end, cols - 1]
        reconstructed_x_time_series = reconstructed_x_time_series.permute(0, 2, 1)

        return reconstructed_x_time_series.to(self.device)


class STFTEmbedder(TsImgEmbedder):
    """
    STFT transformation
    """

    def __init__(self, device, seq_len, n_fft, hop_length):
        super().__init__(device, seq_len)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.min_real, self.max_real, self.min_imag, self.max_imag = None, None, None, None

    def stft_hw(self):
        """Native (freq_bins, time_frames) before square padding.

        torchaudio's centre-padded Spectrogram reflect-pads n_fft // 2 samples on each
        side, so the frame count is (seq_len + 2 * (n_fft // 2) - n_fft) // hop + 1.
        """
        freq_bins = self.n_fft // 2 + 1
        pad = self.n_fft // 2
        time_frames = (self.seq_len + 2 * pad - self.n_fft) // self.hop_length + 1
        return freq_bins, time_frames

    def pad_to_square(self, x, mask=0):
        _, _, cols, rows = x.shape
        max_side = max(cols, rows)
        padding = (0, max_side - rows, 0, max_side - cols)
        return torch.nn.functional.pad(x, padding, mode='constant', value=mask)

    def unpad(self, x, hw):
        h, w = hw
        return x[:, :, :h, :w]

    def cache_min_max_params(self, train_data):
        """
        Args:
            train_data: training timeseries dataset. shape: B*L*K
        this function initializes the min and max values for the real and imaginary parts.
        we'll use this function only once, before the training loop starts.
        """
        real, imag = self.stft_transform(train_data)
        # compute and cache min and max values
        real, min_real, max_real = MinMaxScaler(real.numpy(), True)
        imag, min_imag, max_imag = MinMaxScaler(imag.numpy(), True)
        self.min_real, self.max_real = torch.Tensor(min_real), torch.Tensor(max_real)
        self.min_imag, self.max_imag = torch.Tensor(min_imag), torch.Tensor(max_imag)

    def stft_transform(self, data):
        """
        Args:
            data: time series data. Shape: B*L*K
        Returns:
            real and imaginary parts of the STFT transformation
        """
        data = torch.permute(data, (0, 2, 1))  # we permute to match requirements of torchaudio.transforms.Spectrogram
        spec = T.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length, center=True, power=None).to(data.device)
        transformed_data = spec(data)
        return transformed_data.real, transformed_data.imag

    def ts_to_img(self, signal, pad=False):
        assert self.min_real is not None, "use init_norm_args() to compute scaling arguments"
        if self.seq_len != signal.shape[1]:
            self.seq_len = signal.shape[1]
        # convert to complex spectrogram
        real, imag = self.stft_transform(signal)
        # MinMax scaling
        real = (MinMaxArgs(real, self.min_real.to(self.device), self.max_real.to(self.device)) - 0.5) * 2
        imag = (MinMaxArgs(imag, self.min_imag.to(self.device), self.max_imag.to(self.device)) - 0.5) * 2
        # stack real and imag parts
        stft_out = torch.cat((real, imag), dim=1)
        if pad:
            stft_out = self.pad_to_square(stft_out)
        return stft_out

    def img_to_ts(self, x_image):
        n_fft = self.n_fft
        hop_length, length = self.hop_length, self.seq_len
        min_real, max_real, min_imag, max_imag = self.min_real.to(
            self.device), self.max_real.to(
            self.device), \
            self.min_imag.to(self.device), self.max_imag.to(
            self.device)
        # Unpad square UNet output back to native STFT (H, W) if needed.
        h, w = self.stft_hw()
        if x_image.shape[-2] != h or x_image.shape[-1] != w:
            x_image = self.unpad(x_image, (h, w))
        # -- combine real and imaginary parts --
        split = torch.split(x_image, x_image.shape[1] // 2,
                            dim=1)  # x_image.shape[1] is twice the size of the original dim

        real, imag = split[0], split[1]
        unnormalized_real = ((real / 2) + 0.5) * (max_real - min_real) + min_real
        unnormalized_imag = ((imag / 2) + 0.5) * (max_imag - min_imag) + min_imag
        unnormalized_stft = torch.complex(unnormalized_real, unnormalized_imag)
        # -- inverse stft --
        ispec = T.InverseSpectrogram(n_fft=n_fft, hop_length=hop_length, center=True).to(self.device)

        x_time_series = ispec(unnormalized_stft, length)

        return torch.permute(x_time_series, (0, 2, 1))  # B*L*K(C)
