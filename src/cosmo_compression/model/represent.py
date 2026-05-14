from typing import Any, Dict, Tuple
from io import BytesIO

import torch
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
import wandb
from lightning.pytorch import utilities
from lightning import LightningModule

from cosmo_compression.model import flow_matching as fm
from cosmo_compression.model import resnet, unet


def log_matplotlib_figure(figure_label: str) -> None:
    """Log a matplotlib figure to WandB without converting to Plotly."""
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=300)
    buf.seek(0)
    image = Image.open(buf)
    wandb.log({figure_label: wandb.Image(image)})
    buf.close()


class CosmoFlow(LightningModule):
    """LightningModule for flow‑matching based cosmological field compression."""

    def __init__(
        self,
        unconditional: bool = False,
        log_wandb: bool = True,
        reverse: bool = False,
        latent_img_channels: int = 64,
        use_temporal_masking: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.unconditional = unconditional
        self.log_wandb = log_wandb
        self.latent_img_channels = latent_img_channels
        self.use_temporal_masking = use_temporal_masking

        self.encoder = self._init_encoder(in_channels=1)
        velocity_model = self._init_velocity()
        self.decoder = fm.FlowMatching(velocity_model, reverse=reverse)

        self.validation_step_outputs: list[Dict[str, Any]] = []

    def _init_encoder(self, in_channels: int) -> nn.Module:
        return resnet.ResNetEncoder(
            in_channels=in_channels,
            latent_img_channels=self.latent_img_channels,
        )

    def _init_velocity(self) -> nn.Module:
        return unet.UNet(
            n_channels=1,
            time_dim=256,
            latent_img_channels=self.latent_img_channels,
            use_temporal_masking=self.use_temporal_masking,
        )

    def get_loss(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        y, _ = batch
        t = torch.rand((y.shape[0],), device=y.device)
        h = self.encoder(y) if not self.unconditional else None
        x0 = torch.randn_like(y)
        return self.decoder.compute_loss(x0=x0, x1=y, h=h, t=t)

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        *args,
    ) -> torch.Tensor:
        loss = self.get_loss(batch)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        loss = self.get_loss(batch)
        self.validation_step_outputs.append({"val_loss": loss, "batch": batch})

    def on_validation_epoch_end(self) -> None:
        losses = [out["val_loss"] for out in self.validation_step_outputs]
        mean_val_loss = torch.stack(losses).mean()
        self.log("val_loss", mean_val_loss, prog_bar=True, sync_dist=True)

        batch = self.validation_step_outputs[0]["batch"]
        if self.log_wandb:
            self._log_figures(batch)

        self.validation_step_outputs.clear()

    @utilities.rank_zero_only
    def _log_figures(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        y, _ = batch
        y = y[0:1]
        h = self.encoder(y)

        x0 = torch.randn_like(y)
        pred = self.decoder.predict(x0, h=h, n_sampling_steps=30)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].imshow(y[0].permute(1, 2, 0).cpu().numpy())
        axes[1].imshow(pred[0].permute(1, 2, 0).cpu().numpy())
        axes[0].set_title("x")
        axes[1].set_title("Reconstructed x")
        plt.tight_layout()

        log_matplotlib_figure("field_reconstruction")
        plt.close(fig)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=10, min_lr=1e-8
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
