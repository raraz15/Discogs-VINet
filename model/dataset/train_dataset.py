import json
import pathlib
from typing import Tuple

import numpy as np

import torch
from torch.utils.data import Dataset

from utilities.extract_cqt import mean_downsample_cqt


class TrainDataset(Dataset):
    """Training dataset.

    It samples random anchor and positive versions from the same clique.
    The anchor and positive versions are returned as CQT features. The features are loaded from
    a directory containing the features as memmap files. The features can be downsampled in time
    by taking the mean, clipped to a dynamic range, padded if too short, and scaled to [0,1].
    An epoch is defined as when each clique is seen once, regardless of their number of versions.
    """

    def __init__(
        self,
        cliques_json_path: str,
        features_dir: str,
        context_length: int,
        mean_downsample_factor: int = 1,
        cqt_bins: int = 84,
        scale: bool = True,
        versions_per_clique: int = 2,
        clique_usage_ratio: float = 1.0,
    ) -> None:
        """Initializes the training dataset.

        Parameters:
        -----------
        cliques_json_path : str
            Path to the cliques json file
        features_dir : str
            Path to the directory containing the features
        context_length : int
            Length of the context before downsampling. If a feature is longer than this,
            a chunk of this length is taken randomly. If a feature is shorter, it is padded.
        mean_downsample_factor : int
            Factor by which to downsample the features by averaging
        cqt_bins : int
            Number of CQT bins in a feature array
        scale : bool
            Whether to scale the features to [0,1]
        versions_per_clique : int
            Number of versions to sample from a clique during each iteration. If this number
            is greater than clique size it will sample some same versions multiple times.
        clique_usage_ratio: float
            Ratio of the cliques to use. If < 1.0, it will reduce the number of cliques.
            Usefull for debugging, short tests.
        """

        assert context_length > 0, f"Expected context_length > 0, got {context_length}"
        assert (
            mean_downsample_factor > 0
        ), f"Expected mean_downsample_factor > 0, got {mean_downsample_factor}"
        assert cqt_bins > 0, f"Expected cqt_bins > 0, got {cqt_bins}"
        assert clique_usage_ratio > 0, f"Expected positive, got {clique_usage_ratio}"

        self.cliques_json_path = cliques_json_path
        self.features_dir = pathlib.Path(features_dir)
        self.context_length = context_length
        self.mean_downsample_factor = mean_downsample_factor
        self.scale = scale
        self.cqt_bins = cqt_bins
        self.versions_per_clique = versions_per_clique

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

        self.clique_ids = list(self.cliques.keys())

        # Count the number of cliques and versions
        self.n_cliques, self.n_versions = 0, 0
        for versions in self.cliques.values():
            self.n_cliques += 1
            self.n_versions += len(versions)
        print(f"{self.n_cliques:>7,} cliques left.")
        print(f"{self.n_versions:>7,} versions left.")

        if clique_usage_ratio < 1.0:
            self.n_cliques = int(self.n_cliques * clique_usage_ratio)
            self.clique_ids = list(self.cliques.keys())[: self.n_cliques]
            print(f"\033[33mReducing to {len(self.clique_ids):>7,} cliques.\033[0m")

        self.indices = []
        for clique_id in self.clique_ids:
            versions = self.cliques[clique_id]
            for i in range(len(versions)):
                self.indices.append((clique_id, i))

    def __getitem__(self, index, encode_version=False) -> Tuple[torch.Tensor, list]:
        """Get self.samples_per_clique random anchor versions from a given clique.

        Parameters:
        -----------
        index : int
            Index of the clique to sample versions from
        encode_version : bool
            If True, encode the version_id in the label then it returns a list of
            'clique_id|version_id'. Otherwise it converts the clique_id string to an int.

        Returns:
        --------
        anchors : torch.Tensor
            CQT features of self.samples_per_clique versions.
            shape=(self.samples_per_clique, F, T), dtype=float32
            see self.load_cqt for more details.
        labels : list
            List of labels. The content depends on the encode_version parameter.
        """

        # Get the clique_id and the version position in the clique of the first anchor
        clique_id, version_pos = self.indices[index]

        # Get the versions of the clique
        versions = self.cliques[clique_id]

        if self.versions_per_clique == 4:

            # Sample 4 versions from the clique
            if len(versions) == 2:
                # If the clique has only 2 versions, no need to sample
                anchor1_dct, anchor2_dct = versions
                # Repeat the versions to have 4
                anchor3_dct, anchor4_dct = versions
            elif len(versions) == 3:
                anchor1_dct, anchor2_dct, anchor3_dct = versions
                # Sample the fourth version from the first three
                anchor4_dct = np.random.choice([anchor1_dct, anchor2_dct, anchor3_dct])
            elif len(versions) == 4:
                # The order does not matter, so we can just take the first 4 versions
                anchor1_dct, anchor2_dct, anchor3_dct, anchor4_dct = versions
            else:
                # First anchor is already selected
                anchor1_dct = versions[version_pos]
                # Sample 3 other versions from the clique except the selected version
                possible_indices = np.delete(np.arange(len(versions)), version_pos)
                version_pos2, version_pos3, version_pos4 = np.random.choice(
                    possible_indices, 3, replace=False
                )
                anchor2_dct, anchor3_dct, anchor4_dct = (
                    versions[version_pos2],
                    versions[version_pos3],
                    versions[version_pos4],
                )
            anchor_dicts = [anchor1_dct, anchor2_dct, anchor3_dct, anchor4_dct]

        else:
            # First anchor is already selected
            anchor1_dct = versions[version_pos]

            # Sample one other version from the clique except the current version
            possible_indices = np.delete(np.arange(len(versions)), version_pos)
            anchor2_dct = versions[np.random.choice(possible_indices)]

            anchor_dicts = [anchor1_dct, anchor2_dct]

        # Load the CQT features for anchors and stack them
        anchors, labels = [], []
        for dct in anchor_dicts:
            anchors.append(self.load_cqt(dct["youtube_id"]))
            if encode_version:
                # it may be usefull to encode the version_id in the label
                labels.append(f'{clique_id}|{dct["version_id"]}')
            else:
                labels.append(int(clique_id.split("C-")[1]))
        anchors = torch.stack(anchors, 0)

        return anchors, labels

    def __len__(self) -> int:
        """Each version appears once at each epoch."""

        return len(self.indices)

    def load_cqt(self, yt_id) -> torch.Tensor:
        """Load the magnitude CQT features for a single version with yt_id from the features
        directory. Loads the memmap file, if the feature is long enough it takes a random
        chunk or pads the feature if it is too short. Then it downsamples the feature in time,
        clips the feature to the dynamic range, and scales the feature if specified
        Converts to torch tensor float32.

        Parameters:
        -----------
        yt_id : str
            The YouTube ID of the version

        Returns:
        --------
        feature : torch.FloatTensor
            The CQT feature of the version dtype=float32, shape: (F, T)
            T is the downsampled context length, which is determined during initialization.
        """

        # Get the directory of the features
        feature_dir = self.features_dir / yt_id[:2]
        # We store the features as a memmap file
        feature_path = feature_dir / f"{yt_id}.mm"
        # And the shape as a separate numpy array
        feature_shape_path = feature_dir / f"{yt_id}.npy"
        # Load the memmap shape
        feature_shape = tuple(np.load(feature_shape_path))

        # Load the magnitude CQT
        if feature_shape[0] > self.context_length:
            # If the feature is long enough, take a random chunk
            start = np.random.randint(0, feature_shape[0] - self.context_length)
            fp = np.memmap(
                feature_path,
                dtype="float16",
                mode="r",
                shape=(self.context_length, feature_shape[1]),
                offset=start * feature_shape[1] * 2,  # 2 bytes per float16
            )
        else:
            # Load the whole feature
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
        if feature.shape[0] < self.context_length:
            feature = np.pad(
                feature,
                ((0, self.context_length - feature_shape[0]), (0, 0)),
                mode="constant",
                constant_values=0,
            )

        # Downsample the feature in time
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

        return feature

    @staticmethod
    def collate_fn(items) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collate function for the dataset. Since each item contains the
        information of 2 versions, B = 2 * number of cliques in a batch.

        NOTE: If self._get_item is called with encode_version=True, the labels
        are strings in the format 'clique_id|version_id'. In this case, the labels
        can not be put into a tensor before converted to a string.

        Parameters:
        -----------
        items: list
            List of tuples containing the features and labels of the versions

        Returns:
        --------
        anchors: torch.Tensor
            The CQT features of the anchors, shape=(B, F, T), dtype=float32
            F is the number of CQT bins.
            T is the downsampled context_length,
        labels: torch.Tensor
            1D tensor of the clique labels, shape=(B,)
        """

        anchors = torch.cat([item[0] for item in items], 0)
        labels = torch.tensor([it for item in items for it in item[1]])

        return anchors, labels
