from typing import Union, Tuple

import json
import pathlib

import numpy as np

import torch
from torch.utils.data import Dataset

from utilities.extract_cqt import mean_downsample_cqt


class TestDataset(Dataset):
    """Test dataset.

    It flattens the versions of the cliques and returns their features as CQT features.
    The features are loaded from a directory containing the features as memmap files.
    The features can be downsampled in time by taking the mean, clipped to a dynamic range,
    padded if too short, and scaled to [0,1].
    """

    def __init__(
        self,
        cliques_json_path: str,
        features_dir: str,
        mean_downsample_factor: int = 1,
        cqt_bins: int = 84,
        scale: bool = True,
    ) -> None:
        """Initializes the dataset

        Parameters:
        -----------
        cliques_json_path : str
            Path to the cliques json file
        features_dir : str
            Path to the directory containing the features
        mean_downsample_factor : int
            Factor by which to downsample the features by taking the mean
        cqt_bins : int
            Number of CQT bins in a feature array
        scale : bool
            Whether to scale the features to [0,1]
        """

        assert cqt_bins > 0, f"Expected cqt_bins > 0, got {cqt_bins}"
        assert (
            mean_downsample_factor > 0
        ), f"Expected mean_downsample_factor > 0, got {mean_downsample_factor}"

        self.cliques_json_path = cliques_json_path
        self.features_dir = pathlib.Path(features_dir)
        self.mean_downsample_factor = mean_downsample_factor
        self.scale = scale
        self.cqt_bins = cqt_bins

        # Load the cliques
        print(f"Loading cliques from {cliques_json_path}")
        with open(cliques_json_path) as f:
            self.cliques = json.load(f)

        # Count the number of cliques and versions
        self.n_cliques, self.n_versions = 0, 0
        for versions in self.cliques.values():
            self.n_cliques += 1
            self.n_versions += len(versions)
        print(f"{self.n_cliques:>7,} cliques found.")
        print(f"{self.n_versions:>7,} versions found.")

        # Determine the data source
        if "da-tacos" in cliques_json_path.lower():
            self.discogs_vi = False
            self.shs100k = False
            self.datacos = True
        elif "shs" in cliques_json_path.lower():
            self.discogs_vi = False
            self.shs100k = True
            self.datacos = False
        elif "discogsvi" in cliques_json_path.lower():
            self.discogs_vi = True
            self.shs100k = False
            self.datacos = False
        else:
            raise ValueError("Dataset not recognized.")

        # In datacos all features are present so no need to filter
        if self.discogs_vi or self.shs100k:
            # Delete versions with missing features # TODO: slow
            print("Deleting versions with missing features...")
            for clique_id in list(self.cliques.keys()):
                delete = []
                for i in range(len(self.cliques[clique_id])):
                    yt_id = self.cliques[clique_id][i]["youtube_id"]
                    if not (self.features_dir / yt_id[:2] / f"{yt_id}.mm").exists():
                        delete.append(i)
                for i in reversed(delete):
                    del self.cliques[clique_id][i]
                # If a clique is left with less than 2 versions, delete the clique
                if len(self.cliques[clique_id]) < 2:
                    del self.cliques[clique_id]

            # Count the number of cliques and versions again
            self.n_cliques, self.n_versions = 0, 0
            for versions in self.cliques.values():
                self.n_cliques += 1
                self.n_versions += len(versions)
            print(f"{self.n_cliques:>7,} cliques left.")
            print(f"{self.n_versions:>7,} versions left.")

        # Create a list of all versions together with their clique ID
        self.items = []
        if not self.datacos:
            for clique_id, versions in self.cliques.items():
                for i in range(len(versions)):
                    self.items.append((clique_id, i))
        else:
            for clique_id, versions in self.cliques.items():
                for version_id in versions.keys():
                    self.items.append((clique_id, version_id))

    def __getitem__(
        self, idx, encode_version=False
    ) -> Tuple[torch.Tensor, Union[str, int]]:
        """For a given index, returns the feature and label for the corresponding version.
        Features are loaded as full duration first and then downsampled but chunk sampling is
        not applied, i.e. a feature is returned in full duration.

        Parameters:
        -----------
        batch_idx: int
            Index of the version in the dataset
        encode_version: bool, optional
            If True, the label is a string in the format 'clique_id|version_id'.
            If False, the label is an integer obtained from the clique_id. Default is False.

        Returns:
        --------
        feature: torch.Tensor
            The CQT feature of the version shape=(F,T), dtype=float32

        label: Union[str, int]
            Label for the feature 'clique_id|version_id'. Depends on the value of encode_version.
            NOTE: Our labels are not consecutive integers. They are the clique_id.
        """

        if self.datacos:
            clique_id, version_id = self.items[idx]
            if encode_version:
                label = f"{clique_id}|{version_id}"
            else:
                label = int(clique_id.split("W_")[1])
            _id = version_id
            feature_dir = self.features_dir
        else:
            clique_id, version_idx = self.items[idx]
            version_dict = self.cliques[clique_id][version_idx]
            if encode_version:
                label = f'{clique_id}|{version_dict["version_id"]}'
            else:
                if self.discogs_vi:
                    label = int(clique_id.split("C-")[1])
                else:
                    label = int(clique_id)
            _id = version_dict["youtube_id"]
            feature_dir = self.features_dir / _id[:2]

        # We store the features as a memmap file and the shape as a separate numpy array
        feature_path = feature_dir / f"{_id}.mm"
        feature_shape_path = feature_dir / f"{_id}.npy"

        # Load the entire memmap file. As we use full tracks in evaluation
        feature_shape = tuple(np.load(feature_shape_path))
        fp = np.memmap(
            feature_path,
            dtype="float16",
            mode="r",
            shape=feature_shape,
        )
        # Convert to float32
        feature = np.array(fp, dtype=np.float32)
        del fp
        assert feature.ndim == 2, f"Expected 2D feature, got {feature.ndim}D"
        assert feature.shape[0] > 0, "Empty feature"
        assert (
            feature.shape[1] == self.cqt_bins
        ), f"Expected {self.cqt_bins} features, got {feature.shape[1]}"

        # Pad the feature if it is too short
        # NOTE: I took this value from CQTNet without giving it too much thought
        if feature.shape[0] < 4000:
            feature = np.pad(
                feature,
                ((0, 4000 - feature.shape[0]), (0, 0)),
                "constant",
                constant_values=0,
            )

        # Downsample the feature in time by taking the mean
        if self.mean_downsample_factor > 1:
            feature = mean_downsample_cqt(feature, self.mean_downsample_factor)

        # Clip the feature below zero to be sure
        feature = np.where(feature < 0, 0, feature)

        # Scale the feature to [0,1] if specified
        if self.scale:
            feature /= (
                np.max(feature) + 1e-6
            )  # Add a small value to avoid division by zero

        # Transpose to (F, T) because the CQT is stored as (T, F)
        feature = feature.T

        # Convert to tensor (view not a copy)
        feature = torch.from_numpy(feature)

        return feature, label

    def __len__(self) -> int:
        """Returns the number of versions in the dataset."""

        return len(self.items)
