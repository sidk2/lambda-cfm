"""Training script for CosmoFlow flow-matching compression model."""

import argparse
import os
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from cosmo_compression.data import data
from cosmo_compression.model import represent

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 5"

@dataclass
class TrainConfig:
    """Configuration for flow-matching model training."""
    output_dir: str
    root: str
    run_name: str
    camels_suite: str
    camels_data: str
    learning_rate: float
    grad_clip: float
    batch_size: int
    accumulate_gradients: int | None
    num_workers: int
    save_every: int
    eval_every: int
    latent_dim: int
    latent_img_channels: int
    train_size: int
    val_size: int
    max_steps: int
    max_epochs: int
    profile: bool
    gpus: int
    use_wandb: bool
    no_temporal_masking: bool
    conditioned_attention: bool
    seed: int


torch.set_float32_matmul_precision("medium")


def get_camels_dataloaders(
    batch_size: int,
    num_workers: int,
    idx_train: range,
    idx_val: range,
    map_type: str,
    parameters: list[str],
    suite: str,
    dataset: str,
    root: str,
) -> tuple[DataLoader, DataLoader, int, int]:
    """Build training and validation DataLoaders for CAMELS data."""
    print(f"Using {len(idx_train)} training points and {len(idx_val)} validation points.")
    train_data = data.CAMELS(
        root=root,
        idx_list=idx_train,
        map_type=map_type,
        parameters=parameters,
        suite=suite,
        dataset=dataset,
    )
    val_data = data.CAMELS(
        root=root,
        idx_list=idx_val,
        map_type=map_type,
        parameters=parameters,
        suite=suite,
        dataset=dataset,
    )
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, len(train_data), len(val_data)


def train(args: TrainConfig) -> None:
    """Main training loop."""
    seed_everything(args.seed, workers=True)

    # Logger
    logger = None
    if args.use_wandb:
        logger = WandbLogger(
            project="hierarchical_representations",
            name=args.run_name,
            log_model=False,
        )
        logger.log_hyperparams(vars(args))
    else:
        print("🔸 Running without Weights & Biases logger.")

    # Data splits
    train_end = args.train_size
    val_end = args.train_size + args.val_size
    idx_train = range(train_end)
    idx_val = range(train_end, val_end)

    train_loader, val_loader, n_train, n_val = get_camels_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        idx_train=idx_train,
        idx_val=idx_val,
        map_type="Mcdm",
        parameters=["Omega_m", "sigma_8"],
        suite=args.camels_suite,
        dataset=args.camels_data,
        root=args.root,
    )

    print(
        f"Using {n_train} training samples and {n_val} validation samples "
        f"(suite={args.camels_suite}, data={args.camels_data})."
    )

    fm = represent.CosmoFlow(
        log_wandb=args.use_wandb,
        latent_img_channels=args.latent_img_channels,
        use_temporal_masking=not args.no_temporal_masking,
        conditioned_attention=args.conditioned_attention,
    )

    def init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in", nonlinearity="relu")
            m.bias.data.fill_(0.01)

    fm.apply(init_weights)

    # Checkpoints
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(args.output_dir) / f"{args.run_name}",
        filename="step={step}-val_loss={val_loss:.3f}",
        save_top_k=1,
        monitor="val_loss",
        save_last=True,
        every_n_train_steps=args.save_every,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Trainer
    trainer = Trainer(
        max_steps=args.max_steps,
        gradient_clip_val=args.grad_clip,
        logger=logger,
        log_every_n_steps=50,
        accumulate_grad_batches=args.accumulate_gradients or 1,
        callbacks=[checkpoint_callback, lr_monitor],
        devices=args.gpus,
        val_check_interval=args.eval_every,
        max_epochs=args.max_epochs,
        profiler="simple" if args.profile else None,
        strategy="ddp_find_unused_parameters_true",
        accelerator="gpu",
    )

    trainer.fit(
        model=fm,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )


def parse_args() -> TrainConfig:
    """Parse command line arguments into TrainConfig."""
    parser = ArgumentParser(description="Train CosmoFlow compression model.")

    # Output + logging
    out_grp = parser.add_argument_group("Output & Logging")
    out_grp.add_argument("--output_dir", default="", help="Output directory for checkpoints.")
    out_grp.add_argument("--root", type=str, required=True, help="Dataset directory.")
    out_grp.add_argument("--run_name", default="finetuning", type=str, help="WandB run name.")
    out_grp.add_argument("--use_wandb", action="store_true", default=False)
    
    # CAMELS-specific
    data_grp = parser.add_argument_group("Data Settings")
    data_grp.add_argument("--camels_suite", type=str, default="Astrid", help="CAMELS suite (e.g. IllustrisTNG).")
    data_grp.add_argument("--camels_data", type=str, default="LH", help="CAMELS data/map type (e.g. WDM).")
    data_grp.add_argument("--train_size", type=int, required=True, help="Number of training samples.")
    data_grp.add_argument("--val_size", type=int, required=True, help="Number of validation samples.")

    # Optimisation
    opt_grp = parser.add_argument_group("Optimization")
    opt_grp.add_argument("--learning_rate", default=5e-5, type=float)
    opt_grp.add_argument("--grad_clip", default=1.0, type=float)
    opt_grp.add_argument("--batch_size", default=16, type=int)
    opt_grp.add_argument("--accumulate_gradients", default=None, type=int)
    opt_grp.add_argument("--num_workers", default=4, type=int)
    opt_grp.add_argument("--latent_dim", default=256, type=int)
    opt_grp.add_argument("--latent_img_channels", type=int, default=8)

    # Trainer settings
    train_grp = parser.add_argument_group("Trainer Config")
    train_grp.add_argument("--max_steps", default=2_000_000, type=int)
    train_grp.add_argument("--max_epochs", default=100, type=int)
    train_grp.add_argument("--save_every", default=50, type=int)
    train_grp.add_argument("--eval_every", default=50, type=int)
    train_grp.add_argument("--profile", action="store_true", default=False)
    train_grp.add_argument("--gpus", type=int, default=3, help="How many GPUs to use.")
    train_grp.add_argument(
        "--no-temporal-masking",
        action="store_true",
        default=False,
        help="Disable time-dependent channel masking on encoder output.",
    )
    train_grp.add_argument(
        "--conditioned-attention",
        action="store_true",
        default=False,
        help="Use AdaLN-Zero timestep-conditioned self-attention instead of vanilla.",
    )
    train_grp.add_argument(
        "--seed",
        type=int,
        default=12,
        help="Global random seed passed to seed_everything().",
    )

    args = parser.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    config = parse_args()
    train(config)

