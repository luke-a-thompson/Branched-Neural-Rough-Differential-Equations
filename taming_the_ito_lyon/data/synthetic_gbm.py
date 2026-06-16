from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import jax
import numpy as np
from cyreal.datasets.dataset_protocol import DatasetProtocol
from cyreal.datasets.time_utils import make_sequence_disk_source
from cyreal.datasets.utils import to_host_jax_array
from cyreal.sources import ArraySource, DiskSource

from taming_the_ito_lyon.config import Config

# Synthetic GBM benchmark settings. Edit these constants directly when you want
# to benchmark a different regime without adding config plumbing.
GBM_DIM = 8
NUM_EXAMPLES = 4096
NUM_TIMESTEPS = 256
MU = 0.05
SIGMA = 0.2
X0 = 1.0


@dataclass
class SyntheticGBMDataset(DatasetProtocol):
    config: Config
    split: Literal["train", "val", "test"]
    ordering: Literal["sequential", "shuffle"] = field(init=False)

    def __post_init__(self) -> None:
        self.ordering = "shuffle" if self.split == "train" else "sequential"

        driver, solution = _generate_synthetic_gbm(
            seed=int(self.config.experiment_config.seed),
            dim=_resolve_gbm_dim(self.config),
        )

        train_fraction = float(self.config.experiment_config.train_fraction)
        val_fraction = float(self.config.experiment_config.val_fraction)
        driver_split = _select_example_split(
            driver,
            split=self.split,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
        )
        solution_split = _select_example_split(
            solution,
            split=self.split,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
        )

        self._driver = to_host_jax_array(driver_split)
        self._solution = to_host_jax_array(solution_split)

    def __len__(self) -> int:
        return int(self._driver.shape[0])

    def __getitem__(self, index: int) -> dict[str, jax.Array]:
        return {
            "driver": self._driver[index],
            "solution": self._solution[index],
        }

    def as_array_dict(self) -> dict[str, jax.Array]:
        return {"driver": self._driver, "solution": self._solution}

    def make_array_source(self) -> ArraySource:
        dataset = SyntheticGBMDataset(config=self.config, split=self.split)
        array_source = ArraySource(dataset.as_array_dict(), ordering=self.ordering)
        return array_source

    def make_disk_source(self) -> DiskSource:
        dataset = SyntheticGBMDataset(config=self.config, split=self.split)
        disk_source = make_sequence_disk_source(
            contexts=np.asarray(dataset._driver),
            targets=np.asarray(dataset._solution),
            ordering=self.ordering,
            prefetch_size=128,
        )
        return disk_source


def _resolve_gbm_dim(config: Config) -> int:
    dim = config.experiment_config.synthetic_gbm_dim
    return int(GBM_DIM if dim is None else dim)


def _generate_synthetic_gbm(*, seed: int, dim: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    if NUM_EXAMPLES <= 0:
        raise ValueError("NUM_EXAMPLES must be positive.")
    if NUM_TIMESTEPS < 2:
        raise ValueError("NUM_TIMESTEPS must be at least 2.")
    if dim <= 0:
        raise ValueError("GBM dimension must be positive.")
    if X0 <= 0.0:
        raise ValueError("X0 must be positive for GBM.")

    dt = 1.0 / float(NUM_TIMESTEPS - 1)
    sqrt_dt = np.sqrt(dt, dtype=np.float64)

    increments = sqrt_dt * rng.standard_normal(
        size=(NUM_EXAMPLES, NUM_TIMESTEPS - 1, dim)
    )
    w = np.concatenate(
        [
            np.zeros((NUM_EXAMPLES, 1, dim), dtype=np.float64),
            np.cumsum(increments, axis=1, dtype=np.float64),
        ],
        axis=1,
    )
    t = np.linspace(0.0, 1.0, NUM_TIMESTEPS, dtype=np.float64)[None, :, None]
    drift = (float(MU) - 0.5 * float(SIGMA) ** 2) * t
    solution = float(X0) * np.exp(drift + float(SIGMA) * w)

    return w.astype(np.float32), solution.astype(np.float32)


def _select_example_split(
    array: np.ndarray,
    *,
    split: Literal["train", "val", "test"],
    train_fraction: float,
    val_fraction: float = 0.0,
) -> np.ndarray:
    n = int(len(array))
    if n <= 0:
        raise ValueError("Array must be non-empty.")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1).")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1).")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be < 1.")

    train_end = min(max(int(n * train_fraction), 1), n)
    if val_fraction > 0.0:
        val_end = min(max(int(n * (train_fraction + val_fraction)), train_end + 1), n)
    else:
        val_end = train_end

    if split == "train":
        return array[:train_end]
    if split == "val":
        if val_fraction == 0.0:
            raise ValueError("val_fraction must be > 0 when split='val'.")
        return array[train_end:val_end]
    return array[val_end:]
