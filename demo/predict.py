"""Predict a shielding distribution from MD-derived local environments."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from data import listed_xyz_paths
from inference import discover_checkpoints, predict_committee, resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict a PNGWNN committee distribution for XYZ environments."
    )
    parser.add_argument(
        "--input-dir", type=Path, default=Path("demo/Dataset_Spectrum/aPP_amor/C2")
    )
    parser.add_argument("--model-dir", type=Path, default=Path("models/checkpoints"))
    parser.add_argument(
        "--output", type=Path, default=Path("demo/outputs/aPP_amor_C2.npz")
    )
    parser.add_argument("--num-models", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    paths = listed_xyz_paths(args.input_dir)
    if args.max_samples:
        paths = paths[: args.max_samples]
    device = resolve_device(args.device)
    checkpoints = discover_checkpoints(args.model_dir, args.num_models)
    start = time.perf_counter()
    mean, standard_deviation = predict_committee(
        paths,
        checkpoints,
        device,
        args.batch_size,
        args.num_workers,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        predicted_shielding=mean,
        committee_std=standard_deviation,
        xyz=np.asarray([str(path) for path in paths]),
    )
    elapsed = time.perf_counter() - start
    print(
        f"device={device} committee={len(checkpoints)} samples={len(paths)} "
        f"seconds={elapsed:.1f} output={args.output}"
    )


if __name__ == "__main__":
    main()
