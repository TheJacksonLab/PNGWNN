"""Shared committee-model inference functions."""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from data import XYZGraphDataset
from model.readout import load_model_checkpoint


CHECKPOINT_PATTERN = re.compile(r"best_center_mae_(\d+)_(\d+)\.pth$")


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def discover_checkpoints(model_dir: str | Path, count: int = 3) -> list[Path]:
    """Find committee checkpoints and sort by the two filename indices."""

    model_dir = Path(model_dir)
    found = []
    for path in model_dir.glob("*.pth"):
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        if match:
            found.append((int(match[1]), int(match[2]), path))
    if not found and (model_dir / "best_center_mae.pth").is_file():
        found.append((0, 0, model_dir / "best_center_mae.pth"))
    if not found:
        raise FileNotFoundError(
            f"No PNGWNN checkpoints found in {model_dir}. "
            "Extract the three .pth.zip files into models/checkpoints first."
        )
    if count < 1:
        raise ValueError("The committee size must be at least one")
    if len(found) < count:
        raise ValueError(
            f"Requested {count} models but found {len(found)} in {model_dir}. "
            "Use --num-models 1 for a single newly trained checkpoint."
        )
    return [item[2] for item in sorted(found)[:count]]


@torch.inference_mode()
def predict_one_model(
    model,
    paths: Sequence[str | Path],
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 0,
) -> np.ndarray:
    dataset = XYZGraphDataset(paths)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    predictions = []
    for batch in loader:
        center_output, _, _ = model(batch.to(device))
        predictions.append(center_output.squeeze(-1).cpu().numpy())
    if not predictions:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(predictions)


def predict_committee(
    paths: Sequence[str | Path],
    checkpoints: Sequence[str | Path],
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    if not paths:
        raise ValueError("No XYZ files were supplied")
    if device.type == "cpu":
        warnings.warn(
            "The supplied CUDA-trained PNGWNN checkpoints are not scientifically "
            "validated on CPU. PyG radius_graph truncates dense neighborhoods at "
            "32 neighbors and its CPU selection differs from CUDA; CPU predictions "
            "are compatibility smoke-test outputs only.",
            RuntimeWarning,
            stacklevel=2,
        )
    members = []
    for checkpoint in checkpoints:
        model = load_model_checkpoint(checkpoint, device)
        members.append(
            predict_one_model(model, paths, device, batch_size, num_workers)
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    stacked = np.vstack(members)
    return stacked.mean(axis=0), stacked.std(axis=0, ddof=0)
