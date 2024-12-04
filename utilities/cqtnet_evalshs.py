import os
import sys
import csv
import time
import json
import pathlib
import argparse
import shutil
from typing import List, Tuple

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

sys.path.append(os.path.dirname(os.path.abspath("__file__")))
from utilities.metrics import calculate_metrics
from evaluate import extract_embeddings, load_embeddings

sys.path.append("/home/oaraz/version_identification/")

cqt_dir = "/mnt/projects/discogs-versions/features/cqtnet/"


def norm(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def cut_data_front(data, out_length):
    if out_length is not None:
        if data.shape[0] > out_length:
            max_offset = data.shape[0] - out_length
            offset = 0
            data = data[offset : (out_length + offset), :]
        else:
            offset = out_length - data.shape[0]
            data = np.pad(data, ((0, offset), (0, 0)), "constant")
    if data.shape[0] < 200:
        offset = 200 - data.shape[0]
        data = np.pad(data, ((0, offset), (0, 0)), "constant")
    return data


class CQTNetDataset(Dataset):
    def __init__(self, cliques_json_path: str, features_dir: str):

        self.cliques_json_path = cliques_json_path
        self.features_dir = pathlib.Path(features_dir)
        self.out_length = None

        # Load the cliques
        print(f"Loading cliques from {cliques_json_path}")
        with open(cliques_json_path, "r") as f:
            self.cliques = json.load(f)

        # Delete versions without features
        for clique_id, versions in list(self.cliques.items()):
            _versions = []
            for i, version in enumerate(versions):
                yt_id = version["youtube_id"]
                if os.path.exists(
                    self.features_dir / yt_id[:2] / f"{yt_id}.npy"
                ) and os.path.exists(self.features_dir / yt_id[:2] / f"{yt_id}.mm"):
                    _versions.append(version)
            if len(_versions) > 1:
                self.cliques[clique_id] = _versions
            else:
                del self.cliques[clique_id]

        # Count the number of cliques and versions
        self.n_cliques, self.n_versions = 0, 0
        for versions in self.cliques.values():
            self.n_cliques += 1
            self.n_versions += len(versions)
        print(f"{self.n_cliques:>7,} cliques found.")
        print(f"{self.n_versions:>7,} versions found.")

        # Create a list of all versions together with their clique ID
        self.items = []
        for clique_id, versions in self.cliques.items():
            for i in range(len(versions)):
                self.items.append((clique_id, i))

    def __getitem__(self, idx):

        # Get the clique ID
        clique_id, version_idx = self.items[idx]
        version_dict = self.cliques[clique_id][version_idx]

        label = int(clique_id.split("C-")[0])

        # Load the features for the version
        version_yt_id = version_dict["youtube_id"]
        # Get the directory of the features
        feature_dir = self.features_dir / version_yt_id[:2]
        # We store the features as a memmap file and the shape as a separate numpy array
        feature_path = feature_dir / f"{version_yt_id}.mm"
        feature_shape_path = feature_dir / f"{version_yt_id}.npy"
        # Load the entire memmap file
        feature_shape = tuple(np.load(feature_shape_path))
        fp = np.memmap(
            feature_path,
            dtype="float16",
            mode="r",
            shape=feature_shape,
        )  # T,F
        feature = np.array(fp)
        assert feature.ndim == 2, f"Expected 2D feature, got {feature.ndim}D"
        assert feature.shape[0] > 0, "Empty feature"
        assert feature.shape[1] == 84, f"Expected 84 features, got {feature.shape[1]}"

        # Close the memmap file
        fp._mmap.close()

        # feature = feature.T  # from 12xN to Nx12
        feature = feature.astype(np.float32) / (np.max(np.abs(feature)) + 1e-6)
        feature = cut_data_front(feature, self.out_length)  # T,F
        feature = torch.Tensor(feature)
        feature = feature.permute(1, 0)  # .unsqueeze(0)

        return feature, label

    def __len__(self):
        return len(self.items)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("model_path", type=str, help="Path to the model to evaluate.")
    parser.add_argument(
        "test_cliques", type=str, help="Path to the test cliques.json file."
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        default=cqt_dir,
        help="Path to the features directory.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Number of closest embeddings to consider for the evaluation."
        "If None, all embeddings are considered.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers to use in the DataLoader.",
    )
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\033[31mDevice: {device}\033[0m\n")

    if "CQTNet_SpecAugment_x3.pth" in args.model_path:
        from CQTNet.models.CQTNet import CQTNet

        model = CQTNet()
    elif "best.pth" in args.model_path:
        from TPPNet.models.TPPNet import CQTTPPNet

        model = CQTTPPNet()
    model.load(args.model_path)
    model.to(device)

    eval_dataset = CQTNetDataset(
        cliques_json_path=args.test_cliques,
        features_dir=args.features_dir,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    # Create the output directory
    embeddings_dir = os.path.join(args.output_dir, "embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)

    t0 = time.monotonic()

    extract_embeddings(model, eval_loader, embeddings_dir, device=device)
    del model

    embeddings, labels = load_embeddings(embeddings_dir)

    print("Calculating metrics...")
    metrics = calculate_metrics(
        embeddings=embeddings,
        labels=labels,
        similarity_search="MCSS",
        k=args.k,
        memory_efficient=True,
    )

    eval_time = time.monotonic() - t0
    print(
        f"Total time: {eval_time//3600:.0f}H {(eval_time % 3600) // 60:.0f}M {eval_time % 60:.0f}S"
    )

    print("Deleting the embeddings...")
    shutil.rmtree(embeddings_dir)

    # Save the evaluation results next to the model
    with open(os.path.join(args.output_dir, "evaluation_metrics.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for metric, value in metrics.items():
            writer.writerow([metric, value])

    #############
    print("Done!")
