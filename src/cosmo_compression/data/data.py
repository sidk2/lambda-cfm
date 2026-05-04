"""CAMELS Multifield Dataset loader."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchio.transforms import Resize
from torchvision.datasets import VisionDataset

logging.basicConfig(level=logging.WARNING)

# Normalisation constants: map_type → {resolution → {mean, std, min}}
NORM_DICT: dict[str, dict[int, dict[str, float]]] = {
    "T":     {256: {"mean": 4.63,  "std": 1.1,  "min": 224.6}},
    "Mcdm":  {256: {"mean": 11.00, "std": 0.5,  "min": 2655544000.0}},
    "HI":    {256: {"mean": 4.5,   "std": 1.60, "min": 0.0}},
    "Mstar": {256: {"mean": -5.7,  "std": 2.16, "min": 0.0}},
    "P":     {256: {"mean": 3.8,   "std": 1.5,  "min": 0.3}},
    "Z":     {256: {"mean": -4.5,  "std": 1.3,  "min": 0.0}},
    "Mtot":  {256: {"mean": 11.12, "std": 0.48, "min": 4665044500.0}},
}


class CAMELS(VisionDataset):
    """PyTorch dataset for the CAMELS Multifield Dataset.

    Loads 2-D projected maps and their associated cosmological/astrophysical
    parameters from the CAMELS suite.
    """

    def __init__(
        self,
        root: str,
        redshift: float = 0.0,
        transform: Callable | None = None,
        parameters: Sequence[str] | None = None,
        suite: str = "Astrid",
        resolution: int = 256,
        original_resolution: int = 256,
        idx_list: Sequence[int] | None = None,
        dataset: str | list[str] = "LH",
        map_type: str = "Mcdm",
    ):
        super().__init__(root, transform=transform)
        if parameters is None:
            parameters = ["Omega_m", "sigma_8"]

        self.root = Path(self.root)
        self.redshift = redshift
        self.idx_list = idx_list
        self.resolution = resolution
        self.suite = suite
        self.dataset = dataset
        self.map_type = map_type

        if resolution != original_resolution:
            self.resize: Resize | None = Resize((1, resolution, resolution))
        else:
            self.resize = None

        # Load all requested dataset splits and concatenate
        datasets = [dataset] if isinstance(dataset, str) else dataset
        imgs: list[np.ndarray] = []
        params: list[np.ndarray] = []

        for ds in datasets:
            imgs.append(self._load_images(suite=suite, dataset=ds, map_type=map_type))
            params.append(self._load_parameters(suite=suite, dataset=ds, parameters=list(parameters)))

        self.y = np.concatenate(imgs, axis=0)
        self.x = np.concatenate(params, axis=0)

        self.mean = float(np.mean(self.y))
        self.std = float(np.std(self.y))

    def __len__(self) -> int:
        return min(len(self.x), len(self.y))

    def _load_images(self, suite: str, dataset: str, map_type: str = "Mtot") -> np.ndarray:
        """Load and normalise map images from .npy files."""
        raw = np.load(self.root / f"Maps_{map_type}_{suite}_{dataset}_z={self.redshift:.2f}.npy")

        if self.idx_list is not None:
            if max(self.idx_list) >= len(raw):
                idx_list = [idx for idx in self.idx_list if idx < len(raw)]
                logging.warning(
                    "Index list contains out-of-bounds indices for %s. "
                    "Truncating to %d samples.", suite, len(idx_list),
                )
            else:
                idx_list = list(self.idx_list)
            raw = raw[idx_list]

        min_val = NORM_DICT[map_type][self.resolution]["min"]
        if min_val == 0.0:
            raw = raw + 1.0e-6
        elif min_val < 0.0:
            raw = raw - 1.05 * min_val

        return np.log10(raw.astype(np.float32))[:, None]

    def _load_parameters(
        self,
        suite: str,
        dataset: str,
        parameters: list[str],
    ) -> np.ndarray:
        """Load cosmological parameters from the CAMELS parameter files."""
        params = pd.read_csv(
            self.root / f"params_{dataset}_{suite}.txt",
            sep=" ",
            names=parameters,
            usecols=range(len(parameters)),
            header=None,
        )[parameters]
        params = params.loc[params.index.repeat(15)].reset_index(drop=True)

        if self.idx_list is not None:
            if max(self.idx_list) >= len(params):
                logging.warning(
                    "Index list contains out-of-bounds indices for %s. Truncating.", suite,
                )
                idx_list = [idx for idx in self.idx_list if idx < len(params)]
            else:
                idx_list = list(self.idx_list)
            param_values = params.iloc[idx_list].values
        else:
            param_values = params.values

        return param_values.astype(np.float32)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, np.ndarray]:
        """Return a standardised map and its parameter vector."""
        x, y = self.x[index], self.y[index]
        if self.transform is not None:
            y = self.transform(y)
        if self.resize is not None:
            y = self.resize(y)
        y = (y - self.mean) / self.std
        return y, x