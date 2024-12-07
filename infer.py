"""This script is intended for inference. It creates a csv file following the MIREX 
guidelines provided in https://www.music-ir.org/mirex/wiki/2024:Cover_Song_Identification.

Additional to the specified arguments in the guideline, we added the ability to disable
Automatic Mixed Precision (AMP) for inference. AMP is enabled by default if not specified
in the model configuration file. You can also provide the number of workers to use in the
DataLoader.

The script loads a pre-trained model and computes the pairwise distances between the 
query versions and the candidates in the collection. The output is a tab-separated file 
containing the pairwise distances between the query versions and the candidates.

If you wish to use GPU for inference you should add the CUDA_VISIBLE_DEVICES environment
variable to the command line. For example, to use GPU 0, you should run:

    CUDA_VISIBLE_DEVICES=0 python main.py <collection_list_file> <query_list_file> 
    <output_file> [--disable-amp] [--num-workers]
"""

import os
import csv
import time
import yaml
import argparse
from typing import Tuple

import numpy as np

import torch
from torch.utils.data import DataLoader

from model.nets import CQTNet
from model.dataset import InferenceDataset
from model.utils import load_model
from utilities.utils import format_time
from utilities.tensor_op import pairwise_distance_matrix


@torch.no_grad()
def infer(
    model: CQTNet,
    loader: DataLoader,
    query_list: list[str],
    amp: bool,
    device: torch.device,
) -> Tuple[list[int], np.ndarray]:
    """Perform Version Identification using a pre-trained model. Compute the embeddings
    of all versions and calculate the pairwise distances for given queries. Please refer
    to the argparse arguments for more information.

    Parameters:
    -----------
    model : torch.nn.Module
        Model for inference.
    loader : torch.utils.data.DataLoader
        DataLoader containing the test set cliques
    query_list: list
        List of full path file names for the query versions.
    amp: bool
        Flag to indicate if Automatic Mixed Precision should be used.
    device : torch.device
        Device to use for inference and metric calculation.

    Returns:
    --------
    query_indices: list
        List of indices of the query versions between the candidates.
    pairwise_distance_matrix : np.array
        2D matrix of shape (Q, N) where n is the number of queries and m is the number
        of candidates. Each element (i, j) is the pairwise L2 distance between the i-th query
        and the j-th candidate. Q = len(query_list) and N = len(loader.dataset).
    """

    t0 = time.monotonic()

    model.eval()

    N = len(loader)
    Q = len(query_list)

    emb_dim = model(
        loader.dataset.__getitem__(0).unsqueeze(0).unsqueeze(1).to(device)
    ).shape[1]

    candidate_paths = loader.dataset.file_paths
    assert len(candidate_paths) == N, "Number of candidates does not match the dataset."

    # Preallocate tensors to avoid https://github.com/pytorch/pytorch/issues/13246
    embeddings = torch.zeros((N, emb_dim), dtype=torch.float32, device=device)
    D = torch.zeros((Q, N), dtype=torch.float32, device=device)

    print("Extracting embeddings...")
    for idx, feature in enumerate(loader):
        assert feature.shape[0] == 1, "Batch size must be 1 for inference."
        feature = feature.unsqueeze(1).to(device)  # (1,F,T) -> (1,1,F,T)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            embedding = model(feature)
        embeddings[idx : idx + 1] = embedding
        if (idx + 1) % (len(loader) // 10) == 0 or idx == len(loader) - 1:
            print(f"[{(idx+1):>{len(str(len(loader)))}}/{len(loader)}]")
    print(f"Extraction time: {format_time(time.monotonic() - t0)}")

    print("Calculating pairwise distances...")
    t0 = time.monotonic()
    query_indices = []
    for idx, query_path in enumerate(query_list):
        query_idx = [i for i, p in enumerate(candidate_paths) if p == query_path][0]
        query_indices.append(query_idx)
        q = embeddings[query_idx : query_idx + 1]  # (1, F)
        D[idx : idx + 1] = pairwise_distance_matrix(
            q, embeddings, precision="high"
        )  # (1, N)
    D = D.cpu().numpy()
    print(f"Calculation time: {format_time(time.monotonic() - t0)}")

    return query_indices, D


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="""Path to the configuration file of the trained model. 
        The config should point to the model weigths.""",
    )
    parser.add_argument(
        "collection_list_file",
        type=str,
        help="""Text file containing <number of candidates> full path file names for the 
        <number of candidates> audio files in the collection (including the <number of queries> 
        query documents).""",
    )
    parser.add_argument(
        "query_list_file",
        type=str,
        help="""Text file containing the <number of queries> full path file names for the 
        <number of queries> query documents.""",
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="""Full path to file where submission should output the similarity matrix 
        (<number of candidates> header rows + <number of queries> x <number of candidates> 
        data matrix).""",
    )
    parser.add_argument(
        "--disable-amp",
        action="store_true",
        help="""Flag to disable Automatic Mixed Precision for inference. 
        If not provided, AMP usage will depend on the model config file.""",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=10,
        help="Number of workers to use in the DataLoader.",
    )
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = yaml.safe_load(f)
    print("\033[36m\nExperiment Configuration:\033[0m")
    print(
        "\033[36m" + yaml.dump(config, indent=4, width=120, sort_keys=False) + "\033[0m"
    )

    """This part follows the MIREX2024 submission guidelines."""

    with open(args.collection_list_file, "r") as f:
        collection_list = [p.strip() for p in f.readlines()]
    assert all([os.path.exists(f) for f in collection_list]), "Some files do not exist."

    with open(args.query_list_file, "r") as f:
        query_list = [p.strip() for p in f.readlines()]
    assert set(query_list).issubset(
        set(collection_list)
    ), "Some query files are not in the collection."

    with open(args.output_file, "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "Pairwise Euclidean Distance Matrix",
            ]
        )
        for idx, file_path in enumerate(collection_list):
            writer.writerow([f"{idx+1}", file_path])

    """Ends here"""

    dataset = InferenceDataset(
        collection_list,
        mean_downsample_factor=config["MODEL"]["MEAN_DOWNSAMPLE_FACTOR"],
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\033[31mDevice: {device}\033[0m\n")

    model = load_model(config, device, mode="infer")

    # Disable amp if specified, else keep as the user specified in the config file
    if args.disable_amp:
        config["TRAIN"]["AUTOMATIC_MIXED_PRECISION"] = False

    query_indices, D = infer(
        model, loader, query_list, config["MODEL"]["AUTOMATIC_MIXED_PRECISION"], device
    )

    with open(args.output_file, "a") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["Q/R"] + [f"{idx+1}" for idx in range(len(collection_list))])
        for idx, row in zip(query_indices, D):
            writer.writerow([f"{idx+1}"] + [f"{s}" for s in row])

    print("Done!")
