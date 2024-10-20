"""Training script for the model."""

import os
from datetime import datetime
import time
import yaml
import argparse
from typing import Optional

import numpy as np

import torch
from torch.utils.data import DataLoader

torch.multiprocessing.set_sharing_strategy("file_system")

from model.nets import CQTNet
from model.dataset import TrainDataset, TestDataset
from model.loss import triplet_loss
from utilities.utils import count_model_parameters, save_model, load_model, format_time
from utilities.metrics import calculate_metrics

SEED = 27  # License plate code of Gaziantep, gastronomical capital of Türkiye


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_config: dict,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = "cpu",
) -> float:
    """Train the model for one epoch. Return the average loss of the epoch."""

    model.train()
    losses = []
    for i, (anchors, labels) in enumerate(loader):

        # Add channel dimension and move to device
        anchors = anchors.unsqueeze(1).to(device)  # (B,F,T) -> (B,1,F,T)
        labels = labels.to(device)  # (B,)
        optimizer.zero_grad()
        embeddings = model(anchors)
        loss = triplet_loss(embeddings, labels, **loss_config)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().item())
        # Print the loss
        if (i + 1) % (len(loader) // 10) == 0 or i == len(loader) - 1:
            print(
                f"[{(i+1):>{len(str(len(loader)))}}/{len(loader)}], Batch Loss: {loss.item():.4f}"
            )

    # Step the scheduler if given
    if scheduler is not None:
        scheduler.step()

    # Return the average loss of the epoch
    epoch_loss = np.array(losses).mean().item()

    return epoch_loss


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    similarity_search: str,
    chunk_size: int,
    device: str,
) -> dict:
    """Evaluate the model by simulating the retrieval task. Compute the embeddings
    of all versions and calculate the pairwise distances. Calculate the mean average
    precision of the retrieval task. Metric calculations are done on the cpu but you
    can choose the device for the model. Since we normalize the embeddings, MCSS is
    equivalent to NNS.

    Parameters:
    -----------
    model : torch.nn.Module
        Model to evaluate
    loader : torch.utils.data.DataLoader
        DataLoader containing the test set cliques
    similarity_search: str
        Similarity search function. Options are "NNS" for nearest neighbor search
        and "MCSS" for maximum cosine similarity search.
    chunk_size : int
        Chunk size to use during metrics calculation.
    device : str
        Device to use for inference.

    Returns:
    --------
    metrics : dict
        Dictionary containing the evaluation metrics. See utilities.metrics.calculate_metrics
    """

    t0 = time.monotonic()

    model.eval()  # setting the model to evaluation mode
    embedings, labels = [], []
    print("Extracting embeddings...")
    for idx, (features, batch_labels) in enumerate(loader):
        # Add channel dimension and move to device
        features = features.unsqueeze(1).to(device)  # (B,F,T) -> (B,1,F,T)
        batch_embeddings = model(features)
        embedings.append(batch_embeddings)
        labels.append(batch_labels)
        # Print progress
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
        "--epochs", type=int, default=50, help="Number of epochs to train the model."
    )
    parser.add_argument(
        "--save-frequency", type=int, default=1, help="Save the model every N epochs."
    )
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=5,
        help="Evaluate the model every N epochs.",
    )
    parser.add_argument(
        "--similarity-search",
        "-s",
        type=str,
        default="MCSS",
        choices=["MCSS", "NNS"],
        help="""Similarity search function to use for the evaluation. 
        MCSS: Maximum Cosine Similarity Search, NNS: Nearest Neighbour Search.""",
    )
    parser.add_argument(
        "--chunk-size",
        "-b",
        type=int,
        default=1024,
        help="Chunk size to use during metrics calculation.",
    )
    parser.add_argument(
        "--pre-eval",
        action="store_true",
        help="Evaluate the model before training.",
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
    args = parser.parse_args()

    # Set the seed for reproducibility
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(SEED)

    # Print the arguments
    print("\n\033[32mArguments:\033[0m")
    for arg in vars(args):
        print(f"\033[32m{arg}: {getattr(args, arg)}\033[0m")

    # Load the config from the yaml file
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    print("\n\033[36mExperiment Configuration:\033[0m")
    print(
        "\033[36m" + yaml.dump(config, indent=4, width=120, sort_keys=False) + "\033[0m"
    )
    # Overwrite the epochs in the config file
    config["epochs"] = args.epochs

    # Initialize Wandb if specified
    if not args.no_wandb:
        print("\nUsing Wandb to log the training process.")
        import wandb

        wandb.init(
            project="VersionIdentification",
            config=config,
            id=wandb.util.generate_id() if args.wandb_id is None else args.wandb_id,
            resume="allow",
        )
    else:
        print("\n\033[34mNot logging the training process.\033[0m")

    # Set the device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n\033[31mDevice: {device}\033[0m")

    # Create the save directories
    save_dir = os.path.join(config["MODEL"]["CHECKPOINT_DIR"], config["MODEL"]["NAME"])
    last_save_dir = os.path.join(save_dir, "last_epoch")
    best_save_dir = os.path.join(save_dir, "best_epoch")
    print("\nCheckpoints will be saved to: ", save_dir)

    # Get the current date and time
    date_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    # Initialize the model
    if config["MODEL"]["ARCHITECTURE"].upper() == "CQTNET":
        model = CQTNet(l2_normalize=config["MODEL"]["L2_NORMALIZE"])
    else:
        raise ValueError("Model architecture not recognized.")
    model.to(device)
    grad_params, non_grad_params = count_model_parameters(model)

    # Initialize the optimizer
    if config["TRAIN"]["OPTIMIZER"].upper() == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=config["TRAIN"]["LR"]["INITIAL_RATE"], momentum=0.9
        )
    elif config["TRAIN"]["OPTIMIZER"].upper() == "ADAM":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config["TRAIN"]["LR"]["INITIAL_RATE"]
        )
    elif config["TRAIN"]["OPTIMIZER"].upper() == "ADAMW":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["TRAIN"]["LR"]["INITIAL_RATE"]
        )
    else:
        raise ValueError("Optimizer not recognized.")

    # Schedule the learning rate
    if config["TRAIN"]["LR"]["SCHEDULE"].upper() == "STEP":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=20, gamma=0.1, verbose=True
        )
    elif config["TRAIN"]["LR"]["SCHEDULE"].upper() == "MULTISTEP":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                10,
                20,
                30,
                50,
                70,
                90,
            ],
            gamma=0.5,
            verbose=True,
        )
    elif config["TRAIN"]["LR"]["SCHEDULE"].upper() == "EXPONENTIAL":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    elif config["TRAIN"]["LR"]["SCHEDULE"].upper() == "NONE":
        scheduler = None
    else:
        raise ValueError("Learning rate scheduler not recognized.")

    # Load the loss configuration
    loss_config = {k.lower(): v for k, v in config["TRAIN"]["LOSS_CONFIG"].items()}

    # Load the model and the rest if specified
    if "CHECKPOINT_PATH" in config["MODEL"]:
        model, optimizer, scheduler, start_epoch, train_loss, best_mAP = load_model(
            config["MODEL"]["CHECKPOINT_PATH"], model, optimizer, scheduler
        )
        print(
            f"Model loaded from {config['MODEL']['CHECKPOINT_PATH']}, "
            f"starting from epoch {start_epoch}."
        )
    else:
        start_epoch, best_mAP = 1, 0.0
        print("Starting training from random initialization.")

    # Create the train, validation, and evaluation datasets
    print("Creating the datasets...")
    train_dataset = TrainDataset(
        config["TRAIN"]["TRAIN_CLIQUES"],
        config["TRAIN"]["FEATURES_DIR"],
        context_length=config["TRAIN"]["CONTEXT_LENGTH"],
        mean_downsample_factor=config["MODEL"]["DOWNSAMPLE_FACTOR"],
        data_usage_ratio=config["TRAIN"]["DATA_USAGE_RATIO"],
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

    if args.pre_eval:
        print("Evaluating the model before training...")
        t0 = time.monotonic()
        # Validation set fits in the GPU memory
        metrics = validate(model, eval_loader, device=device)
        t_eval = time.monotonic() - t0
        print(f"mAP: {metrics['mAP']:.6f}, evaluated in {format_time(t_eval)}")
        if not args.no_wandb:
            wandb.log({**metrics, "eval_time": t_eval, "epoch": 0})

    print("Training the model...")
    for epoch in range(start_epoch, args.epochs + 1):

        t0 = time.monotonic()
        print(f" Epoch: [{epoch}/{args.epochs}] ".center(25, "="))
        train_loss = train_epoch(
            model,
            train_loader,
            loss_config,
            optimizer,
            scheduler,
            device=device,
        )
        t_train = time.monotonic() - t0
        print(f"Average epoch Loss: {train_loss:.6f}, in {format_time(t_train)}")
        if not args.no_wandb:
            wandb.log({"train_loss": train_loss, "train_time": t_train, "epoch": epoch})

        # Save the model every N epochs except the last epoch
        if epoch % args.save_frequency == 0 and epoch != args.epochs:
            save_model(
                save_dir=last_save_dir,
                config=config,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loss=train_loss,
                mAP=best_mAP,
                date_time=date_time,
                grad_params=grad_params,
                non_grad_params=non_grad_params,
                epoch=epoch,
            )

        # Evaluate the model on the Validation set every N epochs, and at the last epoch
        if epoch % args.eval_frequency == 0 or epoch == args.epochs:
            print("Evaluating the model...")
            t0 = time.monotonic()
            metrics = validate(
                model,
                eval_loader,
                similarity_search=args.similarity_search,
                chunk_size=args.chunk_size,
                device=device,
            )
            t_eval = time.monotonic() - t0
            print(
                f"mAP: {metrics['mAP']:.5f}, MR1: {metrics['MR']:.2f} - {format_time(t_eval)}"
            )
            if not args.no_wandb:
                wandb.log({**metrics, "eval_time": t_eval, "epoch": epoch})

            # Save the best model so far or the last model
            if metrics["mAP"] > best_mAP or epoch == args.epochs:
                # Save the model
                save_model(
                    save_dir=last_save_dir if epoch == args.epochs else best_save_dir,
                    config=config,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    train_loss=train_loss,
                    mAP=metrics["mAP"],
                    date_time=date_time,
                    grad_params=grad_params,
                    non_grad_params=non_grad_params,
                    epoch=epoch,
                )
                # Update the best mAP
                best_mAP = metrics["mAP"]

    print("===Training finished===")
    if not args.no_wandb:
        wandb.finish()

    #############
    print("Done!")
