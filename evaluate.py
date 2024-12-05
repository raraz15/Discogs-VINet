import os
import csv
import time
import yaml
import argparse

import torch
from torch.utils.data import DataLoader

from model.dataset import TestDataset
from utilities.utils import load_model, format_time, evaluate

# My linux is complaining without the following line
# It required for the DataLoader to have the num_workers > 0
torch.multiprocessing.set_sharing_strategy("file_system")


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
