import os
import time
import yaml

import torch
from torch.utils.data import DataLoader

from utilities.metrics import calculate_metrics
from model.nets import CQTNet
from model.lr_schedulers import (
    CosineAnnealingWarmupRestarts,
    WarmupPiecewiseConstantScheduler,
)


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

    model.eval()

    N = len(loader)

    emb_dim = model(
        loader.dataset.__getitem__(0)[0].unsqueeze(0).unsqueeze(1).to(device)
    ).shape[1]

    # Preallocate tensors to avoid https://github.com/pytorch/pytorch/issues/13246
    embeddings = torch.zeros((N, emb_dim), dtype=torch.float32, device=device)
    labels = torch.zeros(N, dtype=torch.int32, device=device)

    print("Extracting embeddings...")
    for idx, (feature, label) in enumerate(loader):
        assert feature.shape[0] == 1, "Batch size must be 1 for inference."
        feature = feature.unsqueeze(1).to(device)  # (1,F,T) -> (1,1,F,T)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
            embedding = model(feature)
        embeddings[idx : idx + 1] = embedding
        labels[idx : idx + 1] = label.to(device)
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


def format_time(t_total: float):
    return f"{t_total // 3600:.0f}H {(t_total % 3600) // 60:.0f}M {t_total % 60:.0f}S"


def count_model_parameters(model, verbose: bool = True):
    """Counts the number of parameters in a model."""

    grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_grad_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    if verbose:
        print(f"\nTotal number of\n    trainable parameters: {grad_params:,}")
        print(
            f"non-trainable parameters: {non_grad_params:>{len(str(grad_params))+2},}"
        )

    return grad_params, non_grad_params


def build_model(config: dict, device: str):
    from model.nets import CQTNet

    if config["MODEL"]["ARCHITECTURE"].upper() == "CQTNET":
        model = CQTNet(
            ch_in=config["MODEL"]["CONV_CHANNEL"],
            ch_out=config["MODEL"]["EMBEDDING_SIZE"],
            norm=config["MODEL"]["NORMALIZATION"],
            pool=config["MODEL"]["POOLING"],
            l2_normalize=config["MODEL"]["L2_NORMALIZE"],
            projection=config["MODEL"]["PROJECTION"],
        )
        model.to(device)
    else:
        raise ValueError("Model architecture not recognized.")
    _, _ = count_model_parameters(model)
    return model


def save_model(
    save_dir,
    config,
    model,
    optimizer,
    scheduler,
    scaler,
    train_loss,
    mAP,
    date_time,
    epoch,
):
    os.makedirs(save_dir, exist_ok=True)
    # Add the model path
    config["MODEL"]["CHECKPOINT_PATH"] = os.path.abspath(
        os.path.join(save_dir, "model_checkpoint.pth")
    )
    # Save the config
    with open(os.path.join(save_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    # Save the model and everything else
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler is not None else {}
            ),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else {},
            "train_loss": train_loss,
            "mAP": mAP,
            "date_time": date_time,
        },
        config["MODEL"]["CHECKPOINT_PATH"],
    )
    print(f"Model saved in {save_dir}")


def load_model(config: dict, device: str, mode="train"):

    assert mode in ["train", "infer"], "Mode must be either 'train' or 'infer'"

    model = build_model(config, device)

    if mode == "train":

        if config["TRAIN"]["OPTIMIZER"].upper() == "ADAM":
            optimizer = torch.optim.Adam(
                model.parameters(), lr=config["TRAIN"]["LR"]["LR"]
            )
        elif config["TRAIN"]["OPTIMIZER"].upper() == "ADAMW":
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=config["TRAIN"]["LR"]["LR"]
            )
        else:
            raise ValueError("Optimizer not recognized.")

        if "PARAMS" in config["TRAIN"]["LR"]:
            lr_params = {
                k.lower(): v for k, v in config["TRAIN"]["LR"]["PARAMS"].items()
            }
        if config["TRAIN"]["LR"]["SCHEDULE"].upper() == "STEP":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **lr_params)
        elif config["TRAIN"]["LR"]["SCHEDULE"].upper() == "MULTISTEP":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                **lr_params,
            )
        elif config["TRAIN"]["LR"]["SCHEDULE"].upper() == "EXPONENTIAL":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                **lr_params,
            )
        elif config["TRAIN"]["LR"]["SCHEDULE"].upper() == "NONE":
            scheduler = None
        elif config["TRAIN"]["LR"]["SCHEDULE"].upper() == "COSINE":
            scheduler = CosineAnnealingWarmupRestarts(
                optimizer,
                **lr_params,
            )
        elif config["TRAIN"]["LR"]["SCHEDULE"].upper() == "LIN-WARMUP-PCWS":
            scheduler = WarmupPiecewiseConstantScheduler(
                optimizer,
                min_lr=config["TRAIN"]["LR"]["LR"],
                **lr_params,
            )
        else:
            raise ValueError("Learning rate scheduler not recognized.")

        if config["TRAIN"]["AUTOMATIC_MIXED_PRECISION"]:
            print("\033[32mUsing Automatic Mixed Precision...\033[0m")
            scaler = torch.cuda.amp.GradScaler()
        else:
            print("Using full precision...")
            scaler = None

        start_epoch = 1
        train_loss = 0.0
        mAP = 0.0

        if "CHECKPOINT_PATH" in config["MODEL"]:
            checkpoint_path = config["MODEL"]["CHECKPOINT_PATH"]
            if os.path.isfile(checkpoint_path):
                print(
                    f"\033[32mLoading the model checkpoint from {checkpoint_path}\033[0m"
                )
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])
                print(f"Model loaded from epoch {checkpoint['epoch']}")
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                # TODO Give the user the ability to discard the scheduler or create it from scratch
                if scheduler is not None:
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                if scaler is not None:
                    scaler.load_state_dict(checkpoint["scaler_state_dict"])
                start_epoch = checkpoint["epoch"] + 1  # start_epoch = last_epoch +1
                train_loss = checkpoint["train_loss"]
                mAP = checkpoint["mAP"]
            else:
                print(f"\033[31mNo checkpoint found at {checkpoint_path}\033[0m")
                print("Training from scratch.")
        else:
            print("\033[31mNo checkpoint path provided\033[0m")
            print("Training from scratch.")
        return model, optimizer, scheduler, scaler, start_epoch, train_loss, mAP

    else:
        if "CHECKPOINT_PATH" in config["MODEL"]:
            checkpoint_path = config["MODEL"]["CHECKPOINT_PATH"]
            if os.path.isfile(checkpoint_path):
                print(
                    f"\033[32mLoading the model checkpoint from {checkpoint_path}\033[0m"
                )
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])
                print(f"Model loaded from epoch {checkpoint['epoch']}")
            else:
                raise ValueError(f"No checkpoint found at {checkpoint_path}")
        else:
            raise ValueError("No checkpoint path provided in the config file.")
        return model
