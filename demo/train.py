"""Training entry point for the central-atom PNGWNN model."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from data import LabeledXYZDataset, training_data_folders
from inference import resolve_device
from model.readout import DEFAULT_MODEL_CONFIG, CenterReadoutComENet


def split_dataset(dataset, validation_fraction: float, seed: int):
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("--val-frac must be between zero and one")
    if len(dataset) < 2:
        raise ValueError("Training requires at least two structures")
    groups = {}
    for index in range(len(dataset)):
        if isinstance(dataset, Subset):
            sample_index = dataset.indices[index]
            folder_id = dataset.dataset.samples[sample_index][2]
        else:
            folder_id = dataset.samples[index][2]
        groups.setdefault(folder_id, []).append(index)

    generator = torch.Generator().manual_seed(seed)
    training_indices = []
    validation_indices = []
    for indices in groups.values():
        order = torch.randperm(len(indices), generator=generator).tolist()
        shuffled = [indices[position] for position in order]
        validation_size = (
            min(len(indices) - 1, max(1, round(len(indices) * validation_fraction)))
            if len(indices) > 1
            else 0
        )
        validation_indices.extend(shuffled[:validation_size])
        training_indices.extend(shuffled[validation_size:])
    if not validation_indices:
        validation_indices.append(training_indices.pop())
    return Subset(dataset, training_indices), Subset(dataset, validation_indices)


def run_epoch(model, loader, device, loss_function, optimizer=None) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in loader:
            batch = batch.to(device)
            prediction, _, _ = model(batch)
            loss = loss_function(prediction.squeeze(-1), batch.y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * prediction.shape[0]
            count += prediction.shape[0]
    return total / max(count, 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train PNGWNN on the included DFT-labeled environments."
    )
    parser.add_argument("--data-root", type=Path, default=Path("dataset/train"))
    parser.add_argument("--output-dir", type=Path, default=Path("demo/outputs/training"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--patience", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size-train", type=int, default=24)
    parser.add_argument("--batch-size-eval", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=7e-5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--cutoff", type=float, default=5.5)
    parser.add_argument("--num-layers", type=int, default=7)
    parser.add_argument("--hidden-channels", type=int, default=512)
    parser.add_argument("--middle-channels", type=int, default=192)
    parser.add_argument("--out-channels", type=int, default=1)
    parser.add_argument("--num-radial", type=int, default=10)
    parser.add_argument("--num-spherical", type=int, default=2)
    parser.add_argument("--num-output-layers", type=int, default=3)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.epochs < 1:
        raise ValueError("--epochs must be at least one")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    dataset = LabeledXYZDataset(training_data_folders(args.data_root))
    if args.max_samples:
        if args.max_samples < 2:
            raise ValueError("--max-samples must be zero or at least two")
        generator = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(len(dataset), generator=generator)[: args.max_samples]
        dataset = Subset(dataset, indices.tolist())

    training_set, validation_set = split_dataset(
        dataset, args.val_frac, args.seed
    )
    training_loader = DataLoader(
        training_set,
        batch_size=args.batch_size_train,
        shuffle=True,
        num_workers=args.num_workers,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=args.batch_size_eval,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model_config = {
        key: getattr(args, key) for key in DEFAULT_MODEL_CONFIG
    }
    model = CenterReadoutComENet(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_function = nn.L1Loss()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_center_mae.pth"
    history_path = args.output_dir / "history.csv"

    print(
        f"device={device} samples={len(dataset)} "
        f"train={len(training_set)} validation={len(validation_set)}"
    )
    best = float("inf")
    stale_epochs = 0
    history = []
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_mae = run_epoch(
            model, training_loader, device, loss_function, optimizer
        )
        validation_mae = run_epoch(
            model, validation_loader, device, loss_function
        )
        elapsed = time.perf_counter() - epoch_start
        history.append((epoch, train_mae, validation_mae, elapsed))
        print(
            f"epoch={epoch:03d} train_mae={train_mae:.6f} "
            f"val_mae={validation_mae:.6f} seconds={elapsed:.1f}",
            flush=True,
        )
        if validation_mae < best:
            best = validation_mae
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "hparams": model_config,
                    "training": {
                        "epoch": epoch,
                        "validation_mae": validation_mae,
                        "seed": args.seed,
                    },
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early_stop epoch={epoch}")
                break

    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("epoch", "train_mae", "validation_mae", "seconds"))
        writer.writerows(history)
    total = time.perf_counter() - start
    print(
        f"best_validation_mae={best:.6f} total_seconds={total:.1f} "
        f"checkpoint={checkpoint_path} history={history_path}"
    )


if __name__ == "__main__":
    main()
