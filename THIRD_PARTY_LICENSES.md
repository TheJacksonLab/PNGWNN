# Third-party source and modifications

`demo/model/comenet.py` and `demo/model/features.py` derive from the ComENet
implementation distributed with DIG (Dive into Graphs), under GNU GPL v3.
The accompanying source is distributed under GPL-3.0-only; see `LICENSE`.
ComENet is described by Wang et al., "ComENet: Towards Complete and Efficient
Message Passing for 3D Molecular Graphs," NeurIPS 2022. Original GemNet basis
utility attribution links are retained in `features.py`.

Local source changes:

1. `torch.cross()` calls explicitly use `dim=-1`.
2. `np.math.factorial()` was replaced with `math.factorial()`.
3. `associated_legendre_polynomials()` returns early for `L <= 1`.
4. Basis-function imports are package-relative.
5. A separate wrapper reads out the central-carbon node prediction.

The 2026-10-29 distribution places shared data/inference utilities and the
training/prediction entry points directly under `demo/`. Imports and default
paths were adapted, with CUDA as the default device and reduced sample counts
for the two demos. Checkpoint discovery validates committee size and accepts
a single newly trained checkpoint. Numerical backbone operations are unchanged.

Python dependencies are installed separately and retain their own licenses.
LAMMPS (29 August 2024) and ORCA (6.0.0) were used upstream; representative text
inputs are provided in `demo/example_inputs/`. Their executables are not
redistributed. Obtain them separately and follow their license/citation terms.
