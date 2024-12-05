import os
import csv
import time
import yaml
import argparse

import torch
from torch.cuda.amp import autocast  # type: ignore
from torch.utils.data import DataLoader

from model.nets import CQTNet
from model.dataset import TestDataset
from utilities.metrics import calculate_metrics
from utilities.utils import load_model, format_time

# My linux is complaining without the following line
# It required for the DataLoader to have the num_workers > 0
torch.multiprocessing.set_sharing_strategy("file_system")


@torch.no_grad()
def evaluate(
    model: CQTNet,
    loader: DataLoader,
    similarity_search: str,
    chunk_size: int,
    noise_works: bool,
    amp: bool,
    device: str,
) -> dict:
    """Evaluate the model by simulating the retrieval task. Compute the embeddings
    of all versions and calculate the pairwise distances. Calculate the mean average
    precision of the retrieval task. Metric calculations are done on the cpu but you
    can choose the device for the model. Since we normalize the embeddings, MCSS is
    equivalent to NNS. Please refer to the argparse arguments for more information.

    Parameters:
    -----------
    model : CQTNet
        Model to evaluate
    loader : torch.utils.data.DataLoader
        DataLoader containing the test set cliques
    similarity_search: str
        Similarity search function. MIPS, NNS, or MCSS.
    chunk_size : int
        Chunk size to use during metrics calculation.
    noise_works : bool
        Flag to indicate if the dataset contains noise works.
    amp : bool
        Flag to indicate if Automatic Mixed Precision should be used.
    device : str
        Device to use for inference and metric calculation.

    Returns:
    --------
    metrics : dict
        Dictionary containing the evaluation metrics. See utilities.metrics.calculate_metrics
    """

    t0 = time.monotonic()

    if amp and device == "cpu":
        raise ValueError("AMP is not supported on CPU.")

    model.eval()

    N = len(loader)

    emb_dim = model(
        loader.dataset.__getitem__(0)[0].unsqueeze(0).unsqueeze(1).to(device)
    ).shape[1]

    # Preallocate tensors to avoid https://github.com/pytorch/pytorch/issues/13246
    embeddings = torch.zeros((N, emb_dim), dtype=torch.float32, device=device)
    labels = torch.zeros(N, dtype=torch.int32, device=device)

    current_index = 0
    print("Extracting embeddings...")
    for idx, (feature, label) in enumerate(loader):
        assert feature.shape[0] == 1, "Batch size must be 1 for inference."
        feature = feature.unsqueeze(1).to(device)  # (1,F,T) -> (1,1,F,T)
        if amp:
            with autocast(dtype=torch.float16):
                embedding = model(feature)
        else:
            embedding = model(feature)
        embeddings[current_index : current_index + 1] = embedding
        labels[current_index : current_index + 1] = label.to(device)
        current_index += 1
        if (idx + 1) % (len(loader) // 10) == 0 or idx == len(loader) - 1:
            print(f"[{(idx+1):>{len(str(len(loader)))}}/{len(loader)}]")
    print(f"Extraction time: {format_time(time.monotonic() - t0)}")

    # If there are no noise works, remove the cliques with single versions
    # this may happen due to the feature extraction process.
    if not noise_works:
        # Count each label's occurrence
        unique_labels, counts = torch.unique(labels, return_counts=True)
        # Filter labels that occur more than once
        valid_labels = unique_labels[counts > 1]
        # Create a mask for indices where labels appear more than once
        keep_mask = torch.isin(labels, valid_labels)
        if keep_mask.sum() < len(labels):
            print("Removing single version cliques...")
            embeddings = embeddings[keep_mask]
            labels = labels[keep_mask]

    print("Calculating metrics...")
    t0 = time.monotonic()
    metrics = calculate_metrics(
        embeddings,
        labels,
        similarity_search=similarity_search,
        noise_works=noise_works,
        chunk_size=chunk_size,
        device=device,
    )
    print(f"Calculation time: {format_time(time.monotonic() - t0)}")

    return metrics


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="""Path to the configuration file of the trained model. 
        The config will be used to find model weigths.""",
    )
    parser.add_argument(
        "test_cliques",
        type=str,
        help="""Path to the test cliques.json file. 
        Can be SHS100K, Da-TACOS or DiscogsVI.""",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--similarity-search",
        "-s",
        type=str,
        default=None,
        choices=["MIPS", "MCSS", "NNS"],
        help="""Similarity search function to use for the evaluation. 
        MIPS: Maximum Inner Product Search, 
        MCSS: Maximum Cosine Similarity Search, 
        NNS: Nearest Neighbour Search.""",
    )
    parser.add_argument(
        "--features-dir",
        "-f",
        type=str,
        default=None,
        help="""Path to the features directory. 
        Optional, by default uses the path in the config file.""",
    )
    parser.add_argument(
        "--chunk-size",
        "-b",
        type=int,
        default=512,
        help="Chunk size to use during metrics calculation.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=10,
        help="Number of workers to use in the DataLoader.",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="""Flag to disable the GPU. If not provided, 
        the GPU will be used if available.""",
    )
    parser.add_argument(
        "--disable-amp",
        action="store_true",
        help="""Flag to disable Automatic Mixed Precision for inference. 
        If not provided, AMP usage will depend on the model config file.""",
    )
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = yaml.safe_load(f)
    print("\033[36m\nExperiment Configuration:\033[0m")
    print(
        "\033[36m" + yaml.dump(config, indent=4, width=120, sort_keys=False) + "\033[0m"
    )

    if args.features_dir is None:
        print("\033[31mFeatures directory NOT provided.\033[0m")
        args.features_dir = config["TRAIN"]["FEATURES_DIR"]
    print(f"\033[31mFeatures directory: {args.features_dir}\033[0m\n")

    # To evaluate the model in an Information Retrieval setting
    eval_dataset = TestDataset(
        args.test_cliques,
        args.features_dir,
        mean_downsample_factor=config["MODEL"]["DOWNSAMPLE_FACTOR"],
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    if args.no_gpu:
        device = "cpu"
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\033[31mDevice: {device}\033[0m\n")

    if args.disable_amp:
        config["TRAIN"]["AUTOMATIC_MIXED_PRECISION"] = False

    model = load_model(config, device, mode="infer")

    if args.output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.output_dir = os.path.join(
            script_dir, "logs", "evaluation", config["MODEL"]["NAME"]
        )
        if "best_epoch" in config["MODEL"]["CHECKPOINT_PATH"]:
            args.output_dir = os.path.join(args.output_dir, "best_epoch")
        elif "last_epoch" in config["MODEL"]["CHECKPOINT_PATH"]:
            args.output_dir = os.path.join(args.output_dir, "last_epoch")

    if eval_dataset.discogs_vi:
        args.output_dir = os.path.join(args.output_dir, "DiscogsVI")
    elif eval_dataset.datacos:
        args.output_dir = os.path.join(args.output_dir, "Da-TACOS")
    elif eval_dataset.shs100k:
        args.output_dir = os.path.join(args.output_dir, "SHS100K")
    else:
        raise ValueError("Dataset not recognized.")
    print(f"\033[31mOutput directory: {args.output_dir}\033[0m\n")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.similarity_search is None:
        args.similarity_search = config["MODEL"]["SIMILARITY_SEARCH"]

    print("Evaluating...")
    t0 = time.monotonic()
    metrics = evaluate(
        model,
        eval_loader,
        similarity_search=args.similarity_search,
        chunk_size=args.chunk_size,
        noise_works=eval_dataset.datacos,
        amp=config["TRAIN"]["AUTOMATIC_MIXED_PRECISION"],
        device=device,
    )
    print(f"Total time: {format_time(time.monotonic() - t0)}")

    eval_path = os.path.join(args.output_dir, "evaluation_metrics.csv")
    print(f"Saving the evaluation results in: {eval_path}")
    with open(eval_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for metric, value in metrics.items():
            writer.writerow([metric, value])

    #############
    print("Done!")
