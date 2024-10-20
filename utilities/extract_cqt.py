""" This script extracts the CQT spectrogram from a folder containing audio files and 
saves them as memmap files. Each memmap file is stored as float16 to save space. 
You can choose between power spectrograms and magnitude spectrograms.
Power CQT-spectrograms are in dB scale and the dynamic range is 80 dB.
You can mean downsample the CQT in time by specifying the `downsample_rate` parameter.
By default, the CQT is not downsampled."""

import sys
import os
import glob
import time
import argparse
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


def process_audio(
    audio_path: str,
    cqt_dir: str,
    log_dir: str,
    sample_rate: float,
    hop_size: int,
    n_bins: int,
    bins_per_octave: int,
    downsample_rate: int,
    convert_to_power: bool,
    dynamic_range: float,
):

    # Get the YouTube ID of the audio file
    yt_id = os.path.basename(audio_path).split(".")[0]

    try:
        # Load the audio, convert to mono and adjust the sample rate
        audio = es.MonoLoader(filename=audio_path, sampleRate=sample_rate)()

        if len(audio) == 0:
            raise ValueError("Empty audio file.")

        # Compute the CQT
        cqt = librosa.core.cqt(
            y=audio,
            sr=sample_rate,
            hop_length=hop_size,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
        ).T  # (T,F)
        if cqt.shape[0] == 0:
            raise ValueError("Empty CQT.")

        # Convert to amplitude
        cqt = np.abs(cqt)

        # Convert to np.float16 to save storage space
        cqt = cqt.astype(np.float16)

        # Convert to power if specified
        if convert_to_power:
            # Amplitude to dB
            cqt = librosa.core.amplitude_to_db(cqt, ref=np.max)
            # Clip below and above the dynamic range
            cqt = np.where(cqt < -dynamic_range, -dynamic_range, cqt)
            cqt = np.where(cqt > 0, 0, cqt)

        # Downsample the CQT if specified
        if downsample_rate > 1:
            cqt = mean_downsample_cqt(cqt, downsample_rate)

        # Check for NaN and Inf values
        if np.isnan(cqt).any():
            raise ValueError("NaN values in the CQT.")
        if np.isinf(cqt).any():
            raise ValueError("Inf values in the CQT.")

        # We store each file as cqt_dir/yt_id[:2]/yt_id.mm
        output_dir = os.path.join(cqt_dir, yt_id[:2])
        os.makedirs(output_dir, exist_ok=True)
        # Save the CQT as memmap
        output_path = os.path.join(output_dir, f"{yt_id}.mm")
        memmap = np.memmap(output_path, dtype="float16", mode="w+", shape=cqt.shape)
        memmap[:] = cqt[:]
        memmap.flush()
        del memmap
        # Save the memmap shape
        output_path = os.path.join(output_dir, f"{yt_id}.npy")
        np.save(output_path, cqt.shape)
    except KeyboardInterrupt:
        sys.exit(1)
    except Exception as e:
        print(f"Error analyzing {audio_path}:\n{repr(e)}")
        with open(os.path.join(log_dir, f"{yt_id}.txt"), "w") as out_f:
            out_f.write(repr(e) + "\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "audio_dir",
        type=str,
        help="Directory containing the audio files or a text file containing the audio paths.",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Root directory to save the features. <output_dir>/cqt/ will be created.",
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
    parser.add_argument(
        "--convert-to-power",
        action="store_true",
        help="Convert the CQT to power spectrogram (dB).",
    )
    parser.add_argument(
        "--dynamic-range",
        type=int,
        default=80,
        help="Dynamic range in dB to use in case of a Power CQT.",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=20,
        help="Number of parallel processes to use for feature extraction.",
    )
    args = parser.parse_args()

    # Load the audio paths
    print("Loading audio files...")
    if os.path.isfile(args.audio_dir):
        with open(args.audio_dir, "r") as f:
            audio_paths = sorted([p.strip() for p in f.readlines()])
    elif os.path.isdir(args.audio_dir):
        audio_paths = sorted(
            glob.glob(os.path.join(args.audio_dir, "**", "*.mp4"), recursive=True)
        )
    else:
        raise ValueError("audio_dir must be a directory or a file.")
    print(f"{len(audio_paths):,} audio files found.")

    # Skip previously computed features in output_dir
    print("Checking for previously computed features...")
    old_audio_paths = glob.glob(
        os.path.join(args.output_dir, "**", "*.mm"), recursive=True
    )
    old_audio_ids = set([os.path.basename(p).split(".")[0] for p in old_audio_paths])
    audio_paths = [
        p for p in audio_paths if os.path.basename(p).split(".")[0] not in old_audio_ids
    ]
    print(f"{len(audio_paths):,} new features will be computed.")
    del old_audio_paths, old_audio_ids

    # Create the output directories
    cqt_dir = os.path.join(args.output_dir, "cqt")
    os.makedirs(cqt_dir, exist_ok=True)
    print(f"CQTs will be saved in {cqt_dir}")
    log_dir = os.path.join(args.output_dir, "cqt_logs")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logs will be saved in {log_dir}")

    # Compute the number of bins for the CQT
    bins_per_octave = 12 * args.bins_per_semitone
    n_bins = bins_per_octave * args.n_octaves
    print(f"Using {n_bins} total CQT bins for.")

    # Extract the CQTs
    t0 = time.monotonic()
    print(f"Extracting the CQTs with {args.processes} processes...")
    with Pool(processes=args.processes) as pool:
        pool.starmap(
            process_audio,
            [
                (
                    audio_path,
                    cqt_dir,
                    log_dir,
                    args.sample_rate,
                    args.hop_size,
                    n_bins,
                    bins_per_octave,
                    args.downsample_rate,
                    args.convert_to_power,
                    args.dynamic_range,
                )
                for audio_path in audio_paths
            ],
        )
    print(f"Extraction took {time.monotonic()-t0:.2f} seconds.")
