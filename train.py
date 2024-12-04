"""Training script for the model."""

import os
from datetime import datetime
import time
import yaml
import argparse
from typing import Tuple, Union

import numpy as np

import torch
from torch.cuda.amp import autocast  # type: ignore
from torch.utils.data import DataLoader

torch.multiprocessing.set_sharing_strategy("file_system")

from model.nets import CQTNet
from model.dataset import TrainDataset, TestDataset
from model.loss import triplet_loss
from utilities.utils import load_model, save_model, format_time
from utilities.metrics import calculate_metrics

SEED = 27  # License plate code of Gaziantep, gastronomical capital of Türkiye


def train_epoch(
    model: CQTNet,
    loader: DataLoader,
    loss_config: dict,
    optimizer: torch.optim.Optimizer,
    scheduler: Union[torch.optim.lr_scheduler.LRScheduler, None],
    scaler: Union[torch.cuda.amp.GradScaler, None],  # type: ignore
    device: str,
) -> Tuple[float, float, Union[float, None]]:
    """Train the model for one epoch. Return the average loss of the epoch."""

    if scaler is not None and device == "cpu":
        raise ValueError("AMP is not supported on CPU.")

    model.train()
    losses, triplet_stats = [], []
    for i, (anchors, labels) in enumerate(loader):
        anchors = anchors.unsqueeze(1).to(device)  # (B,F,T) -> (B,1,F,T)
        labels = labels.to(device)  # (B,)
        optimizer.zero_grad()  # TODO set_to_none=True?
        if scaler is not None:
            with autocast(dtype=torch.float16):
                embeddings = model(anchors)
                loss, stats = triplet_loss(embeddings, labels, **loss_config)
            scaler.scale(loss).backward()  # type: ignore
            scaler.step(optimizer)
            scaler.update()
        else:
            embeddings = model(anchors)
            loss, stats = triplet_loss(embeddings, labels, **loss_config)
            loss.backward()
            optimizer.step()
        losses.append(loss.detach().item())
        if stats is not None:
            triplet_stats.append(stats)
        if (i + 1) % (len(loader) // 25) == 0 or i == len(loader) - 1:
            print(
                f"[{(i+1):>{len(str(len(loader)))}}/{len(loader)}], Batch Loss: {loss.item():.4f}"
            )

    if scheduler is not None:
        scheduler.step()
        lr = scheduler.optimizer.param_groups[0]["lr"]
    else:
        lr = optimizer.param_groups[0]["lr"]

    # Return the average loss of the epoch
    epoch_loss = np.array(losses).mean().item()

    if triplet_stats:
        triplet_stats = sum(triplet_stats) / len(triplet_stats)
    else:
        triplet_stats = None

    return epoch_loss, lr, triplet_stats


@torch.no_grad()
def validate(
    model: CQTNet,
    loader: DataLoader,
    similarity_search: str,
    chunk_size: int,
    amp: bool,
    device: str,
) -> dict:
    """Evaluate the model by simulating the retrieval task. Compute the embeddings
    of all versions and calculate the pairwise distances. Calculate the mean average
    precision of the retrieval task. Metric calculations are done on the cpu but you
    can choose the device for the model. Since we normalize the embeddings, MCSS is
    equivalent to NNS.

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
    amp : bool
        Use Automatic Mixed Precision.
    device : str
        Device to use for inference.

    Returns:
    --------
    metrics : dict
        Dictionary containing the evaluation metrics. See utilities.metrics.calculate_metrics
    """

    if amp and device == "cpu":
        raise ValueError("AMP is not supported on CPU.")

    t0 = time.monotonic()

    model.eval()
    embedings, labels = [], []
    print("Extracting embeddings...")
    for idx, (feature, label) in enumerate(loader):
        assert feature.shape[0] == 1, "Batch size must be 1 for inference."
        feature = feature.unsqueeze(1).to(device)  # (1,F,T) -> (1,1,F,T)
        if amp:
            with autocast(dtype=torch.float16):
                embedding = model(feature)
        else:
            embedding = model(feature)
        embedings.append(embedding)
        labels.append(label)
        if (idx + 1) % (len(loader) // 10) == 0 or idx == len(loader) - 1:
            print(f"[{(idx+1):>{len(str(len(loader)))}}/{len(loader)}]")
    embedings = torch.cat(embedings, dim=0)
    labels = torch.cat(labels)
    print(f"Extraction time: {format_time(time.monotonic() - t0)}")

    print("Calculating metrics...")
    t0 = time.monotonic()
    metrics = calculate_metrics(
        embedings,
        labels,
        similarity_search=similarity_search,
        chunk_size=chunk_size,
        device=device,
    )
    print(f"Calculation time: {format_time(time.monotonic() - t0)}")

    return metrics


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("config_path", type=str, help="Path to the configuration file.")
    parser.add_argument(
        "--save-frequency", type=int, default=1, help="Save the model every N epochs."
    )
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=1,
        help="Evaluate the model every N epochs.",
    )
    parser.add_argument(
        "--chunk-size",
        "-b",
        type=int,
        default=1024,
        help="Chunk size to use during metrics calculation.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=10,
        help="Number of workers to use in the DataLoader.",
    )
    parser.add_argument(
        "--no-wandb", action="store_true", help="Do not use wandb to log experiments."
    )
    parser.add_argument(
        "--wandb-id", type=str, default=None, help="Wandb id to resume an experiment."
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="VI-after_ismir",
        help="Wandb project name.",
    )
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    print("\n\033[32mArguments:\033[0m")
    for arg in vars(args):
        print(f"\033[32m{arg}: {getattr(args, arg)}\033[0m")

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    print("\n\033[36mExperiment Configuration:\033[0m")
    print(
        "\033[36m" + yaml.dump(config, indent=4, width=120, sort_keys=False) + "\033[0m"
    )

    if not args.no_wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            config=config,
            id=wandb.util.generate_id() if args.wandb_id is None else args.wandb_id,  # type: ignore
            resume="allow",
        )
    else:
        print("\033[31mNot logging the training process.\033[0m")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n\033[34mDevice: {device}\033[0m")

    torch.backends.cudnn.deterministic = config["TRAIN"]["CUDA_DETERMINISTIC"]  # type: ignore
    torch.backends.cudnn.benchmark = config["TRAIN"]["CUDA_BENCHMARK"]  # type: ignore

    # Load or create the model
    model, optimizer, scheduler, scaler, start_epoch, train_loss, best_mAP = load_model(
        config, device
    )

    save_dir = os.path.join(config["MODEL"]["CHECKPOINT_DIR"], config["MODEL"]["NAME"])
    last_save_dir = os.path.join(save_dir, "last_epoch")
    best_save_dir = os.path.join(save_dir, "best_epoch")
    print("Checkpoints will be saved to: ", save_dir)

    date_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    print("Creating the dataset...")
    train_dataset = TrainDataset(
        config["TRAIN"]["TRAIN_CLIQUES"],
        config["TRAIN"]["FEATURES_DIR"],
        context_length=config["TRAIN"]["CONTEXT_LENGTH"],
        mean_downsample_factor=config["MODEL"]["DOWNSAMPLE_FACTOR"],
        versions_per_clique=config["TRAIN"]["VERSIONS_PER_CLIQUE"],
        clique_usage_ratio=config["TRAIN"]["CLIQUE_USAGE_RATIO"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["TRAIN"]["BATCH_SIZE"],
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        drop_last=True,
        num_workers=args.num_workers,
    )

    # To evaluate the model in an Information Retrieval setting
    eval_dataset = TestDataset(
        config["TRAIN"]["VALIDATION_CLIQUES"],
        config["TRAIN"]["FEATURES_DIR"],
        mean_downsample_factor=config["MODEL"]["DOWNSAMPLE_FACTOR"],
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    loss_config = {k.lower(): v for k, v in config["TRAIN"]["LOSS"].items()}

    # Log the initial lr
    if not args.no_wandb:
        if scheduler is not None:
            lr_current = scheduler.optimizer.param_groups[0]["lr"]
        else:
            lr_current = optimizer.param_groups[0]["lr"]
        wandb.log(
            {
                "epoch": start_epoch - 1,
                "lr": lr_current,
            }
        )

    print("Training the model...")
    for epoch in range(start_epoch, config["TRAIN"]["EPOCHS"] + 1):

        t0 = time.monotonic()
        print(f" Epoch: [{epoch}/{config['TRAIN']['EPOCHS']}] ".center(25, "="))
        train_loss, lr_current, triplet_stats = train_epoch(
            model,
            train_loader,
            loss_config,
            optimizer,
            scheduler,
            scaler=scaler,
            device=device,
        )
        t_train = time.monotonic() - t0
        print(f"Average epoch Loss: {train_loss:.6f}, in {format_time(t_train)}")
        if not args.no_wandb:
            wandb.log(
                {
                    "train_loss": train_loss,
                    "train_time": t_train,
                    "epoch": epoch,
                    "lr": lr_current,
                    "difficult_triplets": triplet_stats,
                }
            )

        if epoch % args.save_frequency == 0 or epoch == config["TRAIN"]["EPOCHS"]:
            save_model(
                last_save_dir,
                config=config,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                train_loss=train_loss,
                mAP=best_mAP,
                date_time=date_time,
                epoch=epoch,
            )

        if epoch % args.eval_frequency == 0 or epoch == config["TRAIN"]["EPOCHS"]:
            print("Evaluating the model...")
            t0 = time.monotonic()
            metrics = validate(
                model,
                eval_loader,
                similarity_search=config["MODEL"]["SIMILARITY_SEARCH"],
                chunk_size=args.chunk_size,
                amp=config["TRAIN"]["AUTOMATIC_MIXED_PRECISION"],
                device=device,
            )
            t_eval = time.monotonic() - t0
            print(
                f"MAP: {metrics['MAP']:.3f}, MR1: {metrics['MR1']:.2f} - {format_time(t_eval)}"
            )

            if metrics["MAP"] >= best_mAP:
                best_mAP = metrics["MAP"]
                save_model(
                    best_save_dir,
                    config=config,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    train_loss=train_loss,
                    mAP=best_mAP,
                    date_time=date_time,
                    epoch=epoch,
                )
            if not args.no_wandb:
                wandb.log(
                    {
                        **metrics,
                        "eval_time": t_eval,
                        "epoch": epoch,
                        "best_MAP": best_mAP,
                    }
                )

    print("===Training finished===")
    if not args.no_wandb:
        wandb.finish()

    #############
    print("Done!")
