from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


MAE_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+)/(?:\d+)\s*\|\s*Train\s+Loss:\s*(?P<train>[0-9.eE+-]+)\s*\|\s*Val\s+Loss:\s*(?P<val>[0-9.eE+-]+)"
)


def parse_mae_log(path: Path) -> tuple[list[int], list[float], list[float]]:
    text = path.read_text(encoding="utf-8")
    epochs: list[int] = []
    train_losses: list[float] = []
    val_losses: list[float] = []

    for line in text.splitlines():
        match = MAE_RE.search(line)
        if not match:
            continue
        epochs.append(int(match.group("epoch")))
        train_losses.append(float(match.group("train")))
        val_losses.append(float(match.group("val")))

    if not epochs:
        raise ValueError(f"No MAE epochs parsed from: {path}")
    return epochs, train_losses, val_losses


def parse_unet_csv(path: Path) -> tuple[list[int], list[float], list[float]]:
    epochs: list[int] = []
    train_losses: list[float] = []
    val_losses: list[float] = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"epoch", "train_loss", "val_loss"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError("UNet CSV must contain: epoch, train_loss, val_loss")

        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))

    if not epochs:
        raise ValueError(f"No UNet epochs parsed from: {path}")
    return epochs, train_losses, val_losses


def compute_shared_ylim(*loss_series: list[float]) -> tuple[float, float]:
    values = [v for series in loss_series for v in series]
    min_v = min(values)
    max_v = max(values)
    pad = (max_v - min_v) * 0.08 if max_v > min_v else 0.005
    return min_v - pad, max_v + pad


def plot_single(
    epochs: list[int],
    train_losses: list[float],
    val_losses: list[float],
    title: str,
    output_path: Path,
    y_limits: tuple[float, float],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 6))
    plt.plot(epochs, train_losses, label="Train Loss", linewidth=2)
    plt.plot(epochs, val_losses, label="Val Loss", linewidth=2)
    plt.ylim(y_limits)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_compare(
    mae_epochs: list[int],
    mae_train: list[float],
    mae_val: list[float],
    unet_epochs: list[int],
    unet_train: list[float],
    unet_val: list[float],
    output_path: Path,
    y_limits: tuple[float, float],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    axes[0].plot(mae_epochs, mae_train, label="Train", linewidth=2)
    axes[0].plot(mae_epochs, mae_val, label="Val", linewidth=2)
    axes[0].set_title("MAE Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_ylim(y_limits)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(unet_epochs, unet_train, label="Train", linewidth=2)
    axes[1].plot(unet_epochs, unet_val, label="Val", linewidth=2)
    axes[1].set_title("UNet Loss Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(y_limits)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle("MAE vs UNet Loss Comparison (Shared Y-Axis)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MAE and UNet loss curves with same y-axis scale.")
    parser.add_argument("--mae-log", type=Path, required=True)
    parser.add_argument("--unet-csv", type=Path, required=True)
    parser.add_argument("--out-mae", type=Path, required=True)
    parser.add_argument("--out-unet", type=Path, required=True)
    parser.add_argument("--out-compare", type=Path, required=True)
    args = parser.parse_args()

    mae_epochs, mae_train, mae_val = parse_mae_log(args.mae_log)
    unet_epochs, unet_train, unet_val = parse_unet_csv(args.unet_csv)

    y_limits = compute_shared_ylim(mae_train, mae_val, unet_train, unet_val)

    plot_single(mae_epochs, mae_train, mae_val, "MAE Train vs Val Loss", args.out_mae, y_limits)
    plot_single(unet_epochs, unet_train, unet_val, "UNet Train vs Val Loss", args.out_unet, y_limits)
    plot_compare(
        mae_epochs,
        mae_train,
        mae_val,
        unet_epochs,
        unet_train,
        unet_val,
        args.out_compare,
        y_limits,
    )

    print(f"Shared y-limits: {y_limits[0]:.4f} to {y_limits[1]:.4f}")
    print(f"MAE final val: {mae_val[-1]:.4f}")
    print(f"UNet final val: {unet_val[-1]:.4f}")


if __name__ == "__main__":
    main()
