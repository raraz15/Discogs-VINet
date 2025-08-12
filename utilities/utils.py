import os
import yaml

import torch


def format_time(t_total: float):
    return f"{t_total // 3600:.0f}H {(t_total % 3600) // 60:.0f}M {t_total % 60:.0f}S"


def count_model_parameters(model, verbose=True):
    """Counts the number of parameters in a model."""

    grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_grad_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    if verbose:
        print(f"Total number of\n    trainable parameters: {grad_params:,}")
        print(
            f"non-trainable parameters: {non_grad_params:>{len(str(grad_params))+2},}"
        )

    return grad_params, non_grad_params


def save_model(
    save_dir,
    config,
    model,
    optimizer,
    scheduler,
    train_loss,
    mAP,
    date_time,
    grad_params,
    non_grad_params,
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
            "train_loss": train_loss,
            "mAP": mAP,
            "date_time": date_time,
            "grad_params": grad_params,
            "non_grad_params": non_grad_params,
        },
        config["MODEL"]["CHECKPOINT_PATH"],
    )
    print(f"Model saved in {save_dir}")


def load_model(checkpoint_path, model, optimizer=None, scheduler=None):
    if os.path.isfile(checkpoint_path):
        print(f"\033[32mLoading the model checkpoint from {checkpoint_path}\033[0m")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Model loaded from epoch {checkpoint['epoch']}")
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return (
            model,
            optimizer,
            scheduler,
            checkpoint["epoch"] + 1,  # start_epoch = last_epoch +1
            checkpoint["train_loss"],
            checkpoint["mAP"],
        )
    else:
        print(f"\033[31mNo checkpoint found at {checkpoint_path}\033[0m")
        print("Training from scratch.")
        return model, optimizer, scheduler, 1, 0.0, 0.0
