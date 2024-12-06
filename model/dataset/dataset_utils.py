import numpy as np


def mean_downsample_cqt(cqt: np.ndarray, mean_window_length: int) -> np.ndarray:
    """Downsamples the CQT by taking the mean of every `mean_window_length` frames without
    overlapping. Adapted from https://github.com/yzspku/TPPNet/blob/master/data/gencqt.py

    Parameters:
    -----------
        cqt: np.ndarray, shape=(T,F), CQT to downsample
        mean_window_length: int, number of frames to average together

    Returns:
    --------
        new_cqt: np.ndarray, shape=(T//mean_window_length,F), downsampled CQT
    """

    cqt_T, cqt_F = cqt.shape
    # Discard the last frame
    new_T = int(cqt_T // mean_window_length)
    new_cqt = np.zeros((new_T, cqt_F), dtype=cqt.dtype)
    for i in range(new_T):
        new_cqt[i, :] = cqt[
            i * mean_window_length : (i + 1) * mean_window_length, :
        ].mean(axis=0)
    return new_cqt
