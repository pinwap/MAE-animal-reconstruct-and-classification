from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


EPOCH_LINE_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+)/(?:\d+)\s*\|\s*Train\s+Loss:\s*(?P<train>[0-9.eE+-]+)\s*\|\s*Val\s+Loss:\s*(?P<val>[0-9.eE+-]+)"
)


def parse_losses(log_text: str) -> tuple[list[int], list[float], list[float]]:
    epochs: list[int] = []
    train_losses: list[float] = []
    val_losses: list[float] = []

    for line in log_text.splitlines():
        match = EPOCH_LINE_RE.search(line)
        if not match:
            continue
        epochs.append(int(match.group("epoch")))
        train_losses.append(float(match.group("train")))
        val_losses.append(float(match.group("val")))

    if not epochs:
        raise ValueError("No epoch lines found. Check the log format.")

    return epochs, train_losses, val_losses


def plot_loss(
    epochs: list[int],
    train_losses: list[float],
    val_losses: list[float],
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(11, 6))
    plt.plot(epochs, train_losses, label="Train Loss", linewidth=2)
    plt.plot(epochs, val_losses, label="Val Loss", linewidth=2)

    best_val_idx = min(range(len(val_losses)), key=val_losses.__getitem__)
    best_epoch = epochs[best_val_idx]
    best_val = val_losses[best_val_idx]

    plt.scatter([best_epoch], [best_val], s=80, zorder=3, label=f"Best Val: {best_val:.4f} @ Epoch {best_epoch}")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot train/val loss curves from training log.")
    parser.add_argument("--log", type=Path, required=True, help="Path to training log file.")
    parser.add_argument("--output", type=Path, required=True, help="Output image path (.png).")
    parser.add_argument("--title", type=str, default="Training Loss Curve", help="Plot title.")

    args = parser.parse_args()

    log_text = args.log.read_text(encoding="utf-8")
    epochs, train_losses, val_losses = parse_losses(log_text)
    plot_loss(epochs, train_losses, val_losses, args.output, args.title)

    print(f"Plotted {len(epochs)} epochs to: {args.output}")
    print(f"First Val Loss: {val_losses[0]:.4f} | Last Val Loss: {val_losses[-1]:.4f}")


if __name__ == "__main__":
    main()
