"""Central-atom readout used for polymer NMR prediction."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn
from torch_scatter import scatter

from .comenet import ComENet


DEFAULT_MODEL_CONFIG = {
    "cutoff": 5.5,
    "num_layers": 7,
    "hidden_channels": 512,
    "middle_channels": 192,
    "out_channels": 1,
    "num_radial": 10,
    "num_spherical": 2,
    "num_output_layers": 3,
}


class CenterReadoutComENet(nn.Module):
    """Apply the ComENet node output to the designated central atom.

    The vendored ComENet backbone retains its original graph-level output.
    A forward hook captures the last per-node output before graph pooling, and
    the mask selects the central atom (the first atom in each PNGWNN XYZ file).
    """

    def __init__(self, **config: Any) -> None:
        super().__init__()
        unknown = set(config) - set(DEFAULT_MODEL_CONFIG)
        if unknown:
            raise TypeError(f"Unknown model configuration keys: {sorted(unknown)}")
        self.config = {**DEFAULT_MODEL_CONFIG, **config}
        self.base = ComENet(**self.config)
        self._node_output: torch.Tensor | None = None
        self.base.lin_out.register_forward_hook(self._capture_node_output)

    def _capture_node_output(self, _module, _inputs, output) -> None:
        self._node_output = output

    @staticmethod
    def _center_pool(node_output, center_mask, batch):
        weights = center_mask.to(node_output.dtype)
        counts = scatter(weights, batch, dim=0, reduce="sum").clamp(min=1.0)
        values = scatter(
            weights.unsqueeze(-1) * node_output, batch, dim=0, reduce="sum"
        )
        return values / counts.unsqueeze(-1)

    def forward(self, data):
        graph_output = self.base(data)
        if self._node_output is None:
            raise RuntimeError("ComENet node-output hook did not run")
        node_output = self._node_output
        if node_output.ndim == 1:
            node_output = node_output.unsqueeze(-1)
        center_output = self._center_pool(
            node_output, data.center_mask, data.batch
        )
        return center_output, node_output, graph_output


def model_config_from_checkpoint(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Return only architecture parameters stored in a checkpoint."""

    saved = checkpoint.get("hparams", checkpoint.get("args", {}))
    return {key: saved[key] for key in DEFAULT_MODEL_CONFIG if key in saved}


def load_model_checkpoint(path, device: torch.device) -> CenterReadoutComENet:
    """Load a portable ``state_dict + config`` PNGWNN checkpoint."""

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, Mapping) or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"{path} is not a PNGWNN state-dict checkpoint; expected "
            "'model_state_dict' and 'hparams' (or 'args')."
        )
    model = CenterReadoutComENet(
        **model_config_from_checkpoint(checkpoint)
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
