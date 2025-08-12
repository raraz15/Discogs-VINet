from pathlib import Path

import numpy as np
import librosa
import essentia.standard as es

import torch
from torch.utils.data import Dataset

from .dataset_utils import mean_downsample_cqt


class InferenceDataset(Dataset):

    def __init__(
        self,
        path_pairs: list[tuple[Path, Path | None]],
        context_length: int,
        downsample_factor: int = 5,
        sample_rate: int = 22050,
        hop_size: int = 512,
        cqt_bins: int = 84,
        bins_per_octave: int = 12,
        scale: bool = True,
    ) -> None:
        # TODO docs

        assert cqt_bins > 0, f"Expected cqt_bins > 0, got {cqt_bins}"
        assert (
            downsample_factor > 0
        ), f"Expected downsample_factor > 0, got {downsample_factor}"

        self.path_pairs = path_pairs
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.cqt_bins = cqt_bins
        self.bins_per_octave = bins_per_octave
        self.downsample_factor = downsample_factor
        self.context_length = context_length
        self.scale = scale

    def __len__(self) -> int:
        """Returns the number of tracks to process."""

        return len(self.path_pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor | None, Path, Path]:
        """Returns the CQT of the idx-th track."""

        path_audio, path_output = self.path_pairs[idx]

        if path_output.exists():
            # If the output file already exists, we can skip processing
            return None, path_audio, path_output

        # Load the audio, convert to mono and adjust the sample rate
        try:
            audio = es.MonoLoader(
                filename=str(path_audio), sampleRate=self.sample_rate
            )()
        except Exception as e:
            print(f"Error loading {str(path_audio)}: {repr(e)}")
            return None, path_audio, path_output

        # Compute the CQT
        try:
            cqt = librosa.core.cqt(
                y=audio,
                sr=self.sample_rate,
                hop_length=self.hop_size,
                n_bins=self.cqt_bins,
                bins_per_octave=self.bins_per_octave,
            )  # (F, T)
            assert cqt.size > 0, "Empty cqt"
        except Exception as e:
            print(f"Error computing CQT for {path_audio}: {repr(e)}")
            return None, path_audio, path_output

        # Process the CQT
        try:
            # To be consistent with the rest of the code
            cqt = cqt.T  # (T, F)
            assert cqt.ndim == 2, f"Expected 2D cqt, got {cqt.ndim}D"
            assert (
                cqt.shape[1] == self.cqt_bins
            ), f"Expected {self.cqt_bins} cqts, got {cqt.shape[1]}"

            # Convert to magnitude
            cqt = np.abs(cqt)

            # Convert to np.float16 for consistency with extraction
            cqt = cqt.astype(np.float16)
            # Convert to float32
            cqt = cqt.astype(np.float32)

            if np.isnan(cqt).any():
                raise ValueError("NaN values in the CQT.")
            if np.isinf(cqt).any():
                raise ValueError("Inf values in the CQT.")

            # Pad the cqt if it is too short
            if cqt.shape[0] < self.context_length:
                cqt = np.pad(
                    cqt,
                    ((0, self.context_length - cqt.shape[0]), (0, 0)),
                    "constant",
                    constant_values=0,
                )

            # Downsample the cqt in time by taking the mean
            if self.downsample_factor > 1:
                cqt = mean_downsample_cqt(cqt, self.downsample_factor)

            # Clip the cqt below zero to be sure
            cqt = np.where(cqt < 0, 0, cqt)

            # Scale the cqt to [0,1] if specified
            if self.scale:
                cqt /= np.max(cqt) + 1e-6  # Add a small value to avoid division by zero

            # Redundant but to be consistent
            cqt = cqt.T

        except Exception as e:
            print(f"Error processing CQT for {path_audio}: {repr(e)}")
            return None, path_audio, path_output

        # Convert to tensor
        cqt = torch.as_tensor(cqt)

        return cqt, path_audio, path_output
