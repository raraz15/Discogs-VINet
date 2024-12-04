import os
import yaml

import torch

from model.lr_schedulers import (
    CosineAnnealingWarmupRestarts,
    WarmupPiecewiseConstantScheduler,
)


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


def load_model(config: dict, device: str):

    model = build_model(config, device)

    if config["TRAIN"]["OPTIMIZER"].upper() == "ADAM":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["TRAIN"]["LR"]["LR"])
    elif config["TRAIN"]["OPTIMIZER"].upper() == "ADAMW":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["TRAIN"]["LR"]["LR"]
        )
    else:
        raise ValueError("Optimizer not recognized.")

    if "PARAMS" in config["TRAIN"]["LR"]:
        lr_params = {k.lower(): v for k, v in config["TRAIN"]["LR"]["PARAMS"].items()}
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
            print(f"\033[32mLoading the model checkpoint from {checkpoint_path}\033[0m")
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
