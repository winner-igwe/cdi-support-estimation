import argparse
import os
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import CDIDataset
from model import SupportCNN


def compute_iou_dice(prediction, target, threshold=0.5):
    prediction = (prediction > threshold).float()
    target = (target > 0.5).float()

    intersection = (prediction * target).sum()
    union = prediction.sum() + target.sum() - intersection

    iou = intersection / (union + 1e-8)
    dice = 2 * intersection / (
        prediction.sum() + target.sum() + 1e-8
    )

    return iou.item(), dice.item()


def evaluate(model, loader, device):
    model.eval()

    total_iou = 0.0
    total_dice = 0.0
    total_samples = 0

    with torch.no_grad():
        for diffraction, support in loader:
            diffraction = diffraction.to(device)
            support = support.to(device)

            logits = model(diffraction)
            probabilities = torch.sigmoid(logits)

            iou, dice = compute_iou_dice(
                probabilities,
                support
            )

            batch_size = diffraction.size(0)

            total_iou += iou * batch_size
            total_dice += dice * batch_size
            total_samples += batch_size

    return (
        total_iou / total_samples,
        total_dice / total_samples
    )


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return checkpoint


def load_indices(split_path):
    with h5py.File(split_path, "r") as f:
        return (
            f["train_idx"][:],
            f["val_idx"][:],
            f["test_idx"][:]
        )


def evaluate_dataset(
    model,
    dataset_path,
    indices,
    normalization,
    batch_size,
    device
):
    dataset = CDIDataset(
        dataset_path,
        indices,
        normalization=normalization
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return evaluate(model, loader, device)


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate a trained CDI support estimation model."
    )

    parser.add_argument(
        "--checkpoint",
        required=True
    )

    parser.add_argument(
        "--data",
        required=True
    )

    parser.add_argument(
        "--splits",
        required=True
    )

    parser.add_argument(
        "--ood-geometry",
        required=True
    )

    parser.add_argument(
        "--ood-phase",
        required=True
    )

    parser.add_argument(
        "--ood-combined",
        required=True
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # Load checkpoint first so we use exactly
    # the normalization used during training.
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False
    )

    normalization = checkpoint["normalization"]

    print("Normalization:", normalization)
    print("Best epoch:", checkpoint["epoch"])

    model = SupportCNN().to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    _, _, iid_test_idx = load_indices(args.splits)

    results = {}

    datasets = {
        "IID": (args.data, iid_test_idx),
    }

    # OOD datasets contain all samples directly,
    # so their indices are simply every sample.
    with h5py.File(args.ood_geometry, "r") as f:
        geometry_indices = np.arange(len(f["diffraction"]))

    with h5py.File(args.ood_phase, "r") as f:
        phase_indices = np.arange(len(f["diffraction"]))

    with h5py.File(args.ood_combined, "r") as f:
        combined_indices = np.arange(len(f["diffraction"]))

    datasets["OOD geometry"] = (
        args.ood_geometry,
        geometry_indices
    )

    datasets["OOD phase"] = (
        args.ood_phase,
        phase_indices
    )

    datasets["OOD combined"] = (
        args.ood_combined,
        combined_indices
    )

    for name, (dataset_path, indices) in datasets.items():

        iou, dice = evaluate_dataset(
            model=model,
            dataset_path=dataset_path,
            indices=indices,
            normalization=normalization,
            batch_size=args.batch_size,
            device=device
        )

        results[name] = {
            "IoU": iou,
            "Dice": dice
        }

        print(
            f"{name}: "
            f"IoU={iou:.4f}, "
            f"Dice={dice:.4f}"
        )

    print("\nSummary")
    print("-" * 45)

    for name, metrics in results.items():
        print(
            f"{name:15s} "
            f"IoU: {metrics['IoU']:.4f}   "
            f"Dice: {metrics['Dice']:.4f}"
        )


if __name__ == "__main__":
    main()
