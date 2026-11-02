"""Dataset and XYZ helpers for PNGWNN."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from ase.io import read
from torch.utils.data import Dataset
from torch_geometric.data import Data


TRAINING_FOLDERS = {
    "DataSet1": ("300_C1", "300_C2", "300_C3"),
    "DataSet2": ("300_C1", "300_C2"),
    "DataSet3": ("300_C0", "300_C2", "300_C3", "300_CO"),
    "DataSet1s": ("300_C1", "300_C2", "300_C3"),
    "DataSet2s": ("300_C1", "300_C2"),
    "DataSet3s": ("300_C0", "300_C2", "300_C3", "300_CO"),
}


def xyz_to_graph(path: str | Path) -> Data:
    """Read an XYZ local environment; atom zero is the prediction center."""

    atoms = read(str(path))
    if len(atoms) == 0:
        raise ValueError(f"XYZ file contains no atoms: {path}")
    graph = Data(
        z=torch.as_tensor(atoms.get_atomic_numbers(), dtype=torch.long),
        pos=torch.as_tensor(atoms.get_positions(), dtype=torch.float32),
    )
    graph.center_mask = torch.zeros(len(atoms), dtype=torch.bool)
    graph.center_mask[0] = True
    return graph


def _read_nonempty_lines(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def listed_xyz_paths(folder: str | Path) -> list[Path]:
    """Read ``xyzlist`` when present, otherwise discover ``*.xyz`` files."""

    folder = Path(folder)
    list_path = folder / "xyzlist"
    if list_path.is_file():
        paths = []
        for item in _read_nonempty_lines(list_path):
            candidate = Path(item)
            paths.append(candidate if candidate.is_absolute() else folder / candidate)
        return paths
    return sorted(folder.glob("*.xyz"))


def load_label_table(folder: str | Path) -> tuple[list[Path], list[str], dict]:
    """Load aligned XYZ paths, DFT keys, and NMR label mapping."""

    folder = Path(folder)
    paths = listed_xyz_paths(folder)
    keys = _read_nonempty_lines(folder / "outlist")
    if len(paths) != len(keys):
        raise ValueError(
            f"xyzlist/outlist length mismatch in {folder}: "
            f"{len(paths)} versus {len(keys)}"
        )
    labels = np.load(folder / "nmr.npy", allow_pickle=True).item()
    return paths, keys, labels


def shielding_value(labels: dict, key: str) -> float:
    """Extract an isotropic shielding while tolerating legacy newline keys."""

    record = labels.get(key)
    if record is None:
        record = labels.get(key + "\n")
    if record is None:
        raise KeyError(f"No NMR label found for {key!r}")
    return float(record["Chemical shielding, Isotropic"])


class XYZGraphDataset(Dataset):
    def __init__(self, paths: Sequence[str | Path]) -> None:
        self.paths = [Path(path) for path in paths]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Data:
        return xyz_to_graph(self.paths[index])


class LabeledXYZDataset(Dataset):
    """Labeled local environments collected from one or more folders."""

    def __init__(self, folders: Iterable[str | Path]) -> None:
        self.samples: list[tuple[Path, float, int, int]] = []
        self.folder_names: list[str] = []
        self.dataset_names: list[str] = []
        dataset_ids: dict[str, int] = {}

        for folder_id, folder_value in enumerate(folders):
            folder = Path(folder_value)
            paths, keys, labels = load_label_table(folder)
            self.folder_names.append(folder.name)
            dataset_name = folder.parent.name
            dataset_id = dataset_ids.setdefault(dataset_name, len(dataset_ids))
            for path, key in zip(paths, keys):
                self.samples.append(
                    (path, shielding_value(labels, key), folder_id, dataset_id)
                )
        self.dataset_names = [None] * len(dataset_ids)
        for name, index in dataset_ids.items():
            self.dataset_names[index] = name

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Data:
        path, target, folder_id, dataset_id = self.samples[index]
        graph = xyz_to_graph(path)
        graph.y = torch.tensor([target], dtype=torch.float32)
        graph.folder_id = torch.tensor(folder_id, dtype=torch.long)
        graph.dataset_id = torch.tensor(dataset_id, dtype=torch.long)
        return graph


def training_data_folders(root: str | Path) -> list[Path]:
    root = Path(root)
    return [
        root / dataset / folder
        for dataset, folders in TRAINING_FOLDERS.items()
        for folder in folders
    ]
