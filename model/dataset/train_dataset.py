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
        data_usage_ratio: float = 1.0,
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
        data_usage_ratio : float
            Ratio of the data to use. If < 1.0, it will reduce the number of cliques.
        """

        assert context_length > 0, f"Expected context_length > 0, got {context_length}"
        assert (
            mean_downsample_factor > 0
        ), f"Expected mean_downsample_factor > 0, got {mean_downsample_factor}"
        assert cqt_bins > 0, f"Expected cqt_bins > 0, got {cqt_bins}"

        self.cliques_json_path = cliques_json_path
        self.features_dir = pathlib.Path(features_dir)
        self.context_length = context_length
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

        # Count the number of cliques and versions
        self.n_cliques, self.n_versions = 0, 0
        for versions in self.cliques.values():
            self.n_cliques += 1
            self.n_versions += len(versions)
        print(f"{self.n_cliques:>7,} cliques left.")
        print(f"{self.n_versions:>7,} versions left.")

        # Create a list of all clique ids
        self.clique_ids = list(self.cliques.keys())

        # Reduce the number of cliques if specified, typically used for debugging
        if data_usage_ratio < 1.0:
            self.n_cliques = int(self.n_cliques * data_usage_ratio)
            print(f"\033[33mReducing data to {self.n_cliques:>7,} cliques.\033[0m")
            self.clique_ids = self.clique_ids[: self.n_cliques]

    def __getitem__(self, index, encode_version=False) -> Tuple[torch.Tensor, list]:
        """Get 2 random anchor versions from the same clique.

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
            CQT features of 2 versions. shape=(2,F,T), dtype=float32
            see self.load_cqt for more details.
        labels : list
            List of labels. The content depends on the encode_version parameter.
        """

        # Get the versions of the clique
        clique_id = self.clique_ids[index]
        versions = self.cliques[clique_id]

        # Sample 2 random versions from the clique
        anchor1_dct, anchor2_dct = np.random.choice(versions, 2, replace=False)

        # Load the CQT features for anchors and stack them
        anchors, labels = [], []
        for dct in [anchor1_dct, anchor2_dct]:
            anchors.append(self.load_cqt(dct["youtube_id"]))
            if encode_version:
                # it may be usefull to encode the version_id in the label
                labels.append(f'{clique_id}|{dct["version_id"]}')
            else:
                labels.append(int(clique_id.split("C-")[1]))
        anchors = torch.stack(anchors, 0)

        return anchors, labels

    def __len__(self) -> int:
        """Each cliques appears once or more at each epoch, depending of its size."""

        return len(self.clique_ids)

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
