from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from transformers import ViTForImageClassification, ViTMAEForPreTraining

from data.animals10 import DEFAULT_IMAGE_SIZE, build_dataloaders
from mae_core import DEFAULT_MAE_MODEL_NAME
from models.unet import UNet
from training.classification import ViTClassifierTrainer, evaluate_classifier_epoch, load_mae_encoder_weights_into_classifier
from training.common import create_grad_scaler, get_device, load_checkpoint, save_json, set_seed
from training.evaluation import compare_reconstruction_on_batch
from training.mae_trainer import MAETrainer
from training.unet import UNetReconstructionTrainer


"""Command-line entrypoint for the full Animals-10 MAE workflow.

This file wires together all reusable modules (data/model/train/eval), so you can
run each step independently or run the whole pipeline in sequence.
"""


@dataclass(frozen=True)
class AppConfig:
    """User-facing configuration parsed from CLI arguments."""

    step: str
    data_root: str
    output_dir: str
    image_size: int
    batch_size: int
    val_fraction: float
    num_workers: int
    seed: int
    device: str
    model_name: str
    mask_ratio: float
    patch_size: int
    lr: float
    epochs: int
    epochs_mae: int
    epochs_unet: int
    epochs_cls: int
    checkpoint_every: int
    mae_checkpoint: str | None
    unet_checkpoint: str | None
    cls_checkpoint: str | None


@dataclass(frozen=True)
class RuntimeState:
    """Objects created once and shared by every pipeline stage."""

    device: torch.device
    output_dir: Path
    ckpt_dir: Path
    results_dir: Path
    train_loader: Any
    val_loader: Any
    split: Any


def parse_args() -> argparse.Namespace:
    """Define all CLI arguments used by notebook/terminal workflows."""

    parser = argparse.ArgumentParser(description="Animals-10 MAE/U-Net training and evaluation pipeline")
    parser.add_argument(
        "--step",
        choices=["setup-check", "train-mae", "train-unet", "train-cls", "eval-compare", "all"],
        required=True,
        help="Pipeline step to execute.",
    )

    # Data and I/O
    parser.add_argument("--data-root", default="/kaggle/input/animals10/raw-img")
    parser.add_argument("--output-dir", default="/kaggle/working")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)

    # Runtime
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-name", default=DEFAULT_MAE_MODEL_NAME)

    # Optimization
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1, help="Fallback epochs for all tasks")
    parser.add_argument("--epochs-mae", type=int, default=None)
    parser.add_argument("--epochs-unet", type=int, default=None)
    parser.add_argument("--epochs-cls", type=int, default=None)

    # Checkpointing / resume
    parser.add_argument("--mae-checkpoint", default=None)
    parser.add_argument("--unet-checkpoint", default=None)
    parser.add_argument("--cls-checkpoint", default=None)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    return parser.parse_args()


def parse_config(args: argparse.Namespace) -> AppConfig:
    """Convert raw argparse namespace to a typed immutable config."""

    return AppConfig(
        step=args.step,
        data_root=args.data_root,
        output_dir=args.output_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        model_name=args.model_name,
        mask_ratio=args.mask_ratio,
        patch_size=args.patch_size,
        lr=args.lr,
        epochs=args.epochs,
        epochs_mae=args.epochs_mae if args.epochs_mae is not None else args.epochs,
        epochs_unet=args.epochs_unet if args.epochs_unet is not None else args.epochs,
        epochs_cls=args.epochs_cls if args.epochs_cls is not None else args.epochs,
        checkpoint_every=args.checkpoint_every,
        mae_checkpoint=args.mae_checkpoint,
        unet_checkpoint=args.unet_checkpoint,
        cls_checkpoint=args.cls_checkpoint,
    )


class AnimalMAEPipeline:
    """High-level OOP orchestration for all training and evaluation stages."""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.state = self._build_runtime_state()

    def _build_runtime_state(self) -> RuntimeState:
        # Shared initialization lives here so every stage uses the same setup.
        set_seed(self.cfg.seed)
        device = get_device(self.cfg.device)
        output_dir = Path(self.cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        ckpt_dir = output_dir / "checkpoints"
        results_dir = output_dir / "results"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        train_loader, val_loader, split = build_dataloaders(
            root=self.cfg.data_root,
            batch_size=self.cfg.batch_size,
            image_size=self.cfg.image_size,
            val_fraction=self.cfg.val_fraction,
            seed=self.cfg.seed,
            num_workers=self.cfg.num_workers,
        )
        return RuntimeState(
            device=device,
            output_dir=output_dir,
            ckpt_dir=ckpt_dir,
            results_dir=results_dir,
            train_loader=train_loader,
            val_loader=val_loader,
            split=split,
        )

    def _load_model_weights_if_exists(self, model: torch.nn.Module, checkpoint_path: str | Path | None) -> None:
        """Load checkpoint weights when a resume path is provided and valid."""

        if checkpoint_path is None:
            return
        path = Path(checkpoint_path)
        if not path.exists():
            return
        checkpoint = load_checkpoint(path, map_location="cpu")
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)

    @staticmethod
    def _should_save_periodic(epoch: int, checkpoint_every: int) -> bool:
        return checkpoint_every > 0 and epoch % checkpoint_every == 0

    def _run_epoch_loop(
        self,
        *,
        epochs: int,
        train_fn: Callable[[], dict[str, float]],
        val_fn: Callable[[], dict[str, float]],
        is_better: Callable[[dict[str, float], dict[str, float] | None], bool],
        on_best: Callable[[int, dict[str, float], dict[str, float]], None],
        on_periodic: Callable[[int, dict[str, float], dict[str, float]], None],
        log_prefix: str,
    ) -> dict[str, float]:
        """Generic training loop used by MAE/U-Net/CLS to avoid duplicated code."""

        best_val_metrics: dict[str, float] | None = None

        for epoch in range(1, epochs + 1):
            train_metrics = train_fn()
            val_metrics = val_fn()

            train_log = " ".join([f"train_{k}={v:.6f}" for k, v in train_metrics.items()])
            val_log = " ".join([f"val_{k}={v:.6f}" for k, v in val_metrics.items()])
            print(f"[{log_prefix}] Epoch {epoch}: {train_log} {val_log}")

            if is_better(val_metrics, best_val_metrics):
                best_val_metrics = val_metrics
                on_best(epoch, train_metrics, val_metrics)

            if self._should_save_periodic(epoch, self.cfg.checkpoint_every):
                on_periodic(epoch, train_metrics, val_metrics)

        return best_val_metrics or {}

    def setup_check(self) -> None:
        """Quick sanity check for data pipeline, labels, and runtime device."""

        images, labels = next(iter(self.state.train_loader))
        print(f"Device: {self.state.device}")
        print(f"Classes: {len(self.state.split.class_to_idx)}")
        print(f"Train samples: {len(self.state.split.train_samples)}")
        print(f"Val samples: {len(self.state.split.val_samples)}")
        print(f"Batch shape: {tuple(images.shape)}")
        print(f"Batch labels shape: {tuple(labels.shape)}")

    def train_mae(self) -> None:
        """Stage A: continual MAE pretraining on Animals-10."""

        model = ViTMAEForPreTraining.from_pretrained(self.cfg.model_name).to(self.state.device)
        self._load_model_weights_if_exists(model, self.cfg.mae_checkpoint)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.cfg.lr)
        trainer = MAETrainer(
            model=model,
            optimizer=optimizer,
            device=self.state.device,
            mask_ratio=self.cfg.mask_ratio,
            scaler=create_grad_scaler(self.state.device),
        )

        mae_ckpt_dir = self.state.ckpt_dir / "mae"

        def train_fn() -> dict[str, float]:
            return {"mse": trainer.train_epoch(self.state.train_loader)}

        def val_fn() -> dict[str, float]:
            return {"mse": trainer.evaluate_epoch(self.state.val_loader)}

        def is_better(current: dict[str, float], best: dict[str, float] | None) -> bool:
            return best is None or current["mse"] < best["mse"]

        def on_best(epoch: int, train_metrics: dict[str, float], val_metrics: dict[str, float]) -> None:
            trainer.save_checkpoint(
                mae_ckpt_dir / "best.pt",
                epoch=epoch,
                metrics={**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}},
                config={"mask_ratio": self.cfg.mask_ratio, "lr": self.cfg.lr},
            )

        def on_periodic(epoch: int, train_metrics: dict[str, float], val_metrics: dict[str, float]) -> None:
            trainer.save_checkpoint(
                mae_ckpt_dir / f"epoch_{epoch}.pt",
                epoch=epoch,
                metrics={**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}},
                config={"mask_ratio": self.cfg.mask_ratio, "lr": self.cfg.lr},
            )

        best_val = self._run_epoch_loop(
            epochs=self.cfg.epochs_mae,
            train_fn=train_fn,
            val_fn=val_fn,
            is_better=is_better,
            on_best=on_best,
            on_periodic=on_periodic,
            log_prefix="MAE",
        )
        save_json(self.state.results_dir / "mae_metrics.json", {"best_val": best_val})

    def train_unet(self) -> None:
        """Stage B: train U-Net reconstruction baseline with the same masking level."""

        model = UNet().to(self.state.device)
        self._load_model_weights_if_exists(model, self.cfg.unet_checkpoint)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.cfg.lr)
        trainer = UNetReconstructionTrainer(
            model=model,
            optimizer=optimizer,
            device=self.state.device,
            mask_ratio=self.cfg.mask_ratio,
            scaler=create_grad_scaler(self.state.device),
        )

        unet_ckpt_dir = self.state.ckpt_dir / "unet"

        def train_fn() -> dict[str, float]:
            return {"mse": trainer.train_epoch(self.state.train_loader)}

        def val_fn() -> dict[str, float]:
            return {"mse": trainer.evaluate_epoch(self.state.val_loader)}

        def is_better(current: dict[str, float], best: dict[str, float] | None) -> bool:
            return best is None or current["mse"] < best["mse"]

        def on_best(epoch: int, train_metrics: dict[str, float], val_metrics: dict[str, float]) -> None:
            trainer.save_checkpoint(
                unet_ckpt_dir / "best.pt",
                epoch=epoch,
                metrics={**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}},
                config={"mask_ratio": self.cfg.mask_ratio, "patch_size": self.cfg.patch_size, "lr": self.cfg.lr},
            )

        def on_periodic(epoch: int, train_metrics: dict[str, float], val_metrics: dict[str, float]) -> None:
            trainer.save_checkpoint(
                unet_ckpt_dir / f"epoch_{epoch}.pt",
                epoch=epoch,
                metrics={**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}},
                config={"mask_ratio": self.cfg.mask_ratio, "patch_size": self.cfg.patch_size, "lr": self.cfg.lr},
            )

        best_val = self._run_epoch_loop(
            epochs=self.cfg.epochs_unet,
            train_fn=train_fn,
            val_fn=val_fn,
            is_better=is_better,
            on_best=on_best,
            on_periodic=on_periodic,
            log_prefix="U-NET",
        )
        save_json(self.state.results_dir / "unet_metrics.json", {"best_val": best_val})

    def train_cls(self) -> None:
        """Stage C: fine-tune classifier initialized from MAE encoder weights."""

        model = ViTForImageClassification.from_pretrained(
            self.cfg.model_name,
            num_labels=len(self.state.split.class_to_idx),
            ignore_mismatched_sizes=True,
        ).to(self.state.device)
        model = load_mae_encoder_weights_into_classifier(model, self.cfg.mae_checkpoint)
        self._load_model_weights_if_exists(model, self.cfg.cls_checkpoint)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.cfg.lr)
        trainer = ViTClassifierTrainer(
            model=model,
            optimizer=optimizer,
            device=self.state.device,
            scaler=create_grad_scaler(self.state.device),
        )

        cls_ckpt_dir = self.state.ckpt_dir / "cls"

        def train_fn() -> dict[str, float]:
            loss, acc = trainer.train_epoch(self.state.train_loader)
            return {"loss": loss, "acc": acc}

        def val_fn() -> dict[str, float]:
            loss, acc = trainer.evaluate_epoch(self.state.val_loader)
            return {"loss": loss, "acc": acc}

        def is_better(current: dict[str, float], best: dict[str, float] | None) -> bool:
            return best is None or current["acc"] >= best["acc"]

        def on_best(epoch: int, train_metrics: dict[str, float], val_metrics: dict[str, float]) -> None:
            trainer.save_checkpoint(
                cls_ckpt_dir / "best.pt",
                epoch=epoch,
                metrics={**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}},
                config={"lr": self.cfg.lr},
            )

        def on_periodic(epoch: int, train_metrics: dict[str, float], val_metrics: dict[str, float]) -> None:
            trainer.save_checkpoint(
                cls_ckpt_dir / f"epoch_{epoch}.pt",
                epoch=epoch,
                metrics={**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}},
                config={"lr": self.cfg.lr},
            )

        best_val = self._run_epoch_loop(
            epochs=self.cfg.epochs_cls,
            train_fn=train_fn,
            val_fn=val_fn,
            is_better=is_better,
            on_best=on_best,
            on_periodic=on_periodic,
            log_prefix="CLS",
        )
        save_json(self.state.results_dir / "cls_metrics.json", {"best_val": best_val})

    def eval_compare(self) -> None:
        """Run final metrics and export qualitative reconstruction comparison image."""

        mae_model = ViTMAEForPreTraining.from_pretrained(self.cfg.model_name).to(self.state.device)
        self._load_model_weights_if_exists(mae_model, self.cfg.mae_checkpoint or (self.state.ckpt_dir / "mae" / "best.pt"))
        if hasattr(mae_model, "config"):
            mae_model.config.mask_ratio = self.cfg.mask_ratio

        unet_model = UNet().to(self.state.device)
        self._load_model_weights_if_exists(unet_model, self.cfg.unet_checkpoint or (self.state.ckpt_dir / "unet" / "best.pt"))

        cls_model = ViTForImageClassification.from_pretrained(
            self.cfg.model_name,
            num_labels=len(self.state.split.class_to_idx),
            ignore_mismatched_sizes=True,
        ).to(self.state.device)
        self._load_model_weights_if_exists(cls_model, self.cfg.cls_checkpoint or (self.state.ckpt_dir / "cls" / "best.pt"))

        batch = next(iter(self.state.val_loader))
        comparison_path = self.state.results_dir / "comparison" / "sample.png"
        reconstruction_metrics = compare_reconstruction_on_batch(
            mae_model=mae_model,
            unet_model=unet_model,
            batch=batch,
            device=self.state.device,
            mask_ratio=self.cfg.mask_ratio,
            output_path=comparison_path,
        )
        cls_loss, cls_acc = evaluate_classifier_epoch(cls_model, self.state.val_loader, self.state.device)

        payload = {
            **reconstruction_metrics,
            "classification_loss": cls_loss,
            "classification_accuracy": cls_acc,
            "comparison_image": str(comparison_path),
        }
        save_json(self.state.results_dir / "final_metrics.json", payload)
        print(payload)

    def run(self) -> None:
        """Dispatch the selected CLI step (or all steps in order)."""

        if self.cfg.step == "setup-check":
            self.setup_check()
            return
        if self.cfg.step == "train-mae":
            self.train_mae()
            return
        if self.cfg.step == "train-unet":
            self.train_unet()
            return
        if self.cfg.step == "train-cls":
            self.train_cls()
            return
        if self.cfg.step == "eval-compare":
            self.eval_compare()
            return

        # "all" mode executes the whole pipeline in sequence.
        self.setup_check()
        self.train_mae()
        self.train_unet()
        self.train_cls()
        self.eval_compare()


def main() -> None:
    """Script entrypoint used by `python main.py ...`."""

    config = parse_config(parse_args())
    pipeline = AnimalMAEPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
