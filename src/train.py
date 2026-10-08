import argparse
import os
import random

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import CDIDataset
from model import SupportCNN


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Metrics
# ============================================================

def compute_iou_dice(prediction, target, threshold=0.5):
    prediction = (prediction > threshold).float()
    target = (target > 0.5).float()

    intersection = (prediction * target).sum()
    union = prediction.sum() + target.sum() - intersection

    iou = intersection / (union + 1e-8)

    dice = (
        2 * intersection
        / (prediction.sum() + target.sum() + 1e-8)
    )

    return iou.item(), dice.item()


# ============================================================
# Configuration
# ============================================================

def load_config(config_path):
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required. Install it with: pip install pyyaml"
        )

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-8):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        probabilities = torch.sigmoid(logits)

        probabilities = probabilities.flatten(1)
        target = target.flatten(1)

        intersection = (probabilities * target).sum(dim=1)

        dice = (
            2 * intersection + self.smooth
        ) / (
            probabilities.sum(dim=1)
            + target.sum(dim=1)
            + self.smooth
        )

        return (1 - dice).mean()


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, target):
        return (
            self.bce(logits, target)
            + self.dice(logits, target)
        )
        
def create_loss(loss_name):
    if loss_name == "bce":
        return nn.BCEWithLogitsLoss()

    elif loss_name == "dice":
        return DiceLoss()

    elif loss_name == "bce_dice":
        return BCEDiceLoss()

    else:
        raise ValueError(
            f"Unknown loss: {loss_name}"
        )
# ============================================================
# Dataset
# ============================================================

def load_split_indices(split_path):

    with h5py.File(split_path, "r") as f:
        train_idx = f["train_idx"][:]
        val_idx = f["val_idx"][:]
        test_idx = f["test_idx"][:]

    return train_idx, val_idx, test_idx


def create_dataloaders(
    dataset_path,
    train_idx,
    val_idx,
    test_idx,
    normalization,
    batch_size
):

    train_dataset = CDIDataset(
        dataset_path,
        train_idx,
        normalization=normalization
    )

    val_dataset = CDIDataset(
        dataset_path,
        val_idx,
        normalization=normalization
    )

    test_dataset = CDIDataset(
        dataset_path,
        test_idx,
        normalization=normalization
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, test_loader


# ============================================================
# Training
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0

    for diffraction, support in loader:

        diffraction = diffraction.to(device)
        support = support.to(device)

        optimizer.zero_grad()

        logits = model(diffraction)

        loss = criterion(logits, support)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * diffraction.size(0)

    epoch_loss = running_loss / len(loader.dataset)

    return epoch_loss


# ============================================================
# Validation
# ============================================================

def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_samples = 0

    with torch.no_grad():

        for diffraction, support in loader:

            diffraction = diffraction.to(device)
            support = support.to(device)

            logits = model(diffraction)

            loss = criterion(logits, support)

            probabilities = torch.sigmoid(logits)

            iou, dice = compute_iou_dice(
                probabilities,
                support
            )

            batch_size = diffraction.size(0)

            running_loss += loss.item() * batch_size
            total_iou += iou * batch_size
            total_dice += dice * batch_size
            total_samples += batch_size

    val_loss = running_loss / total_samples
    val_iou = total_iou / total_samples
    val_dice = total_dice / total_samples

    return val_loss, val_iou, val_dice


# ============================================================
# Main training function
# ============================================================

def train(config, dataset_path, split_path, checkpoint_dir):

    seed = config["seed"]

    batch_size = config["training"]["batch_size"]
    num_epochs = config["training"]["epochs"]
    learning_rate = config["training"]["learning_rate"]

    normalization = config["dataset"]["normalization"]

    set_seed(seed)

    os.makedirs(checkpoint_dir, exist_ok=True)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_idx, val_idx, test_idx = load_split_indices(
        split_path
    )

    print("Train samples:", len(train_idx))
    print("Validation samples:", len(val_idx))
    print("IID test samples:", len(test_idx))

    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_path=dataset_path,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        normalization=normalization,
        batch_size=batch_size
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = SupportCNN().to(device)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print("Model parameters:", parameter_count)

    # --------------------------------------------------------
    # Loss and optimizer
    # --------------------------------------------------------

    loss_name = config["training"]["loss"]
    criterion = create_loss(loss_name)
    print("Loss:", loss_name)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_iou": [],
        "val_dice": []
    }

    best_val_iou = -1.0
    best_epoch = 0

    checkpoint_name = (
        config["experiment_name"] + "_best.pth"
    )

    checkpoint_path = os.path.join(
        checkpoint_dir,
        checkpoint_name
    )

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(num_epochs):

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_iou, val_dice = validate(
            model,
            val_loader,
            criterion,
            device
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)
        history["val_dice"].append(val_dice)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val IoU: {val_iou:.4f} | "
            f"Val Dice: {val_dice:.4f}"
        )

        # ----------------------------------------------------
        # Save best checkpoint
        # ----------------------------------------------------

        if val_iou > best_val_iou:

            best_val_iou = val_iou
            best_epoch = epoch + 1

            checkpoint = {
                "epoch": best_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_iou": float(val_iou),
                "val_dice": float(val_dice),
                "val_loss": float(val_loss),
                "normalization": normalization,
                "seed": seed,
                "history": history
            }

            torch.save(
                checkpoint,
                checkpoint_path
            )

            print(
                f"  Saved best checkpoint → "
                f"{checkpoint_path}"
            )

    print("\nTraining complete.")
    print("Best epoch:", best_epoch)
    print("Best validation IoU:", best_val_iou)
    print("Checkpoint:", checkpoint_path)


# ============================================================
# Command-line interface
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Train the CDI support estimation CNN."
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to experiment YAML configuration."
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Path to the HDF5 dataset."
    )

    parser.add_argument(
        "--splits",
        required=True,
        help="Path to the HDF5 split file."
    )

    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        help="Directory for saved model checkpoints."
    )

    args = parser.parse_args()

    config = load_config(args.config)

    print("=" * 60)
    print(
        f"Experiment: {config['experiment_name']}"
    )
    print(
        f"Normalization: "
        f"{config['dataset']['normalization']}"
    )
    print("=" * 60)

    train(
        config=config,
        dataset_path=args.data,
        split_path=args.splits,
        checkpoint_dir=args.checkpoint_dir
    )


if __name__ == "__main__":
    main()
