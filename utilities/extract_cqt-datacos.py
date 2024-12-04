""" This script extracts the CQT spectrogram from a folder containing audio files and 
saves them as memmap files. Each memmap file is converted to float16 to save space. 
Power CQT-spectrograms are in dB scale and the dynamic range is 80 dB.
"""

import sys
import os
import glob
import time
import argparse
import json
from multiprocessing import Pool

import numpy as np

import librosa
import essentia.standard as es


def mean_downsample_cqt(cqt: np.ndarray, mean_window_length: int):
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


def CQT(params):
    audio_path, cqt_dir, mm_path, shape_path, args, n_bins, bins_per_octave = params
    try:
        # Get the YouTube ID of the audio file
        yt_id = os.path.basename(audio_path).split(".")[0]
        # Load the audio, convert to mono and adjust the sample rate
        audio = es.MonoLoader(filename=audio_path, sampleRate=args.sample_rate)()
        # Compute the CQT
        cqt = librosa.core.cqt(
            y=audio,
            sr=args.sample_rate,
            hop_length=args.hop_size,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
        ).T  # (T,F)
        # Convert to magnitude
        cqt = np.abs(cqt)
        # Downsample the CQT if specified
        if args.downsample_rate > 1:
            cqt = mean_downsample_cqt(cqt, args.downsample_rate)
        # # Amplitude to dB
        cqt = librosa.core.amplitude_to_db(cqt, ref=np.max)
        # Convert to np.float16
        cqt = cqt.astype(np.float16)
        # Clip below and above the dynamic range
        cqt = np.where(cqt < -80, -80, cqt)
        cqt = np.where(cqt > 0, 0, cqt)
        # We store each file as cqt_dir/yt_id[:2]/yt_id.mm
        os.makedirs(cqt_dir, exist_ok=True)
        # Save the CQT as memmap
        memmap = np.memmap(mm_path, dtype="float16", mode="w+", shape=cqt.shape)
        memmap[:] = cqt[:]
        memmap.flush()
        del memmap
        np.save(shape_path, cqt.shape)
    except KeyboardInterrupt:
        sys.exit(1)
    except Exception as e:
        print(f"Error analyzing {audio_path}")
        with open(os.path.join(log_dir, f"{yt_id}.txt"), "w") as out_f:
            out_f.write(repr(e) + "\n")


# TODO: remove these
audio_dir = "/mnt/projects/da-tacos/da-tacos/da-tacos_benchmark_coveranalysis/"
json_path = (
    "/mnt/projects/da-tacos/da-tacos_metadata/da-tacos_benchmark_subset_metadata.json"
)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Root directory to save the features. <output_dir>/cqt will be created.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Sample rate to use for the audio files",
    )
    parser.add_argument(
        "--hop-size",
        type=int,
        default=512,
        help="Hop size to use for the CQT in samples.",
    )
    parser.add_argument(
        "--n-octaves", type=int, default=7, help="Number of octaves to use for the CQT."
    )
    parser.add_argument(
        "--bins-per-semitone",
        type=int,
        default=1,
        help="Number of CQT bins per semitone.",
    )
    parser.add_argument(
        "--downsample-rate",
        type=int,
        default=1,
        help="Downsample rate to use for mean averaging the CQT in time.",
    )
    args = parser.parse_args()

    with open(json_path) as f:
        cliques = json.load(f)

    # Compute the number of bins for the CQT
    bins_per_octave = 12 * args.bins_per_semitone
    n_bins = bins_per_octave * args.n_octaves

    params = []
    for i, (clique_id, versions) in enumerate(cliques.items()):
        for version_id in versions.keys():
            audio_path = os.path.join(audio_dir, f"{version_id}.mp3")
            memmap_path = os.path.join(args.output_dir, f"{version_id}.mm")
            shape_path = os.path.join(args.output_dir, f"{version_id}.npy")
            params.append(
                (
                    audio_path,
                    args.output_dir,
                    memmap_path,
                    shape_path,
                    args,
                    n_bins,
                    bins_per_octave,
                )
            )

    # TODO: these paths are not used :()
    cqt_dir = os.path.join(args.output_dir, "datacos", "cqt")
    os.makedirs(cqt_dir, exist_ok=True)
    print(f'CQTs will be saved in "{cqt_dir}"')
    log_dir = os.path.join(args.output_dir, "datacos", "cqt_logs")
    os.makedirs(log_dir, exist_ok=True)
    print(f'Logs will be saved in "{log_dir}"')

    t0 = time.monotonic()

    print("begin")
    pool = Pool(20)
    pool.map(CQT, params)
    pool.close()
    pool.join()

    print(f"Finished in {time.monotonic()-t0} seconds.")
