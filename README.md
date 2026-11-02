# PNGWNN

Data, pretrained models, and demonstration code accompanying:

**Machine-Learned Solid-State ¹³C Chemical Shifts for Structure-Resolved
Analysis of Polymer Simulations**

Hang Zhang and Nicholas E. Jackson  
Department of Chemistry, University of Illinois at Urbana-Champaign,
Urbana, Illinois 61801, United States. Nicholas E. Jackson is also affiliated
with the Beckman Institute for Advanced Science and Technology.
Correspondence: jacksonn@illinois.edu.

## Contents

- `dataset/train.zip`: 16,800 DFT-labeled PP/PVA/PMMA local environments.
- `dataset/test.zip`: 144 PE/aPP DFT-labeled evaluation examples.
- `models/`: three pretrained committee checkpoints, each compressed as ZIP.
- `demo/train.py`, `demo/predict.py`: the main training and prediction code.
- `demo/data.py`, `demo/inference.py`, `demo/model/`: required shared code.
- `demo/Dataset_Spectrum.zip`: 30,000 unlabeled aPP C2 local environments.
- `demo/example_inputs/`: representative LAMMPS and ORCA inputs.

After extraction, each XYZ contains one local environment in angstrom, with the target carbon
first. Each labeled subset has aligned `xyzlist`, `outlist`, and `nmr.npy`
files. `nmr.npy` is the original NumPy object dictionary; the loader uses the
`Chemical shielding, Isotropic` field and `allow_pickle=True`.
`outlist` entries identify original DFT calculations; the output files are
not bundled. Folder names are preserved from the source data.

## Run the demos

Run all commands from the repository root. Use Python 3.12 and an NVIDIA CUDA
GPU for scientific predictions. Install the pinned dependencies:

```bash
conda env create -f demo/environment.yml
conda activate pngwnn
```

Extract the data and model archives using the commands below. Data archives
already contain their top-level directory (`train`, `test`, or
`Dataset_Spectrum`); extract them into the indicated destination without adding
another folder named after the ZIP file.

```bash
python -m zipfile -e dataset/train.zip dataset
python -m zipfile -e dataset/test.zip dataset
python -m zipfile -e demo/Dataset_Spectrum.zip demo
python -m zipfile -e models/best_center_mae_1_3.pth.zip models/checkpoints
python -m zipfile -e models/best_center_mae_2_248.pth.zip models/checkpoints
python -m zipfile -e models/best_center_mae_3_359.pth.zip models/checkpoints
```

The resulting paths should be `dataset/train/`, `dataset/test/`,
`demo/Dataset_Spectrum/aPP_amor/C2/`, and `models/checkpoints/*.pth`.
Training requires the training data; pretrained prediction requires the
spectrum data and model checkpoints. The test archive supplies the accompanying
DFT-labeled evaluation examples. Unpacked data, checkpoints, and generated
outputs are excluded from Git by `.gitignore`.

Train a new example model (32 structures, one epoch):

```bash
python demo/train.py
```

Training reads the labels, constructs atomic graphs, makes a folder-stratified
80/20 split, optimizes central-carbon MAE, and saves the best checkpoint and
`history.csv` under `demo/outputs/training/`.
The default seed is 42. Use `--max-samples 0 --epochs 200` to train on all
16,800 structures in the training dataset.
This short example does not recreate the pretrained committee, whose saved
seeds are 1999, 2087, and 2122.

Predict with the three pretrained models (first 16 spectrum structures):

```bash
python demo/predict.py
```

Use `--max-samples 0` for all 30,000 aPP C2 structures. The output
`demo/outputs/aPP_amor_C2.npz` contains `predicted_shielding`, `committee_std`,
and corresponding `xyz` paths. Predictions are chemical shieldings in ppm;
committee standard deviation uses `ddof=0`. Under the original referencing
convention, `shift_ppm = 183.6555 - shielding_ppm`.

For your own XYZ directory, or a single newly trained checkpoint:

```bash
python demo/predict.py --input-dir path/to/xyz --max-samples 0 --output predictions.npz
python demo/predict.py --model-dir demo/outputs/training --num-models 1
```

The one-epoch example model is not expected to produce useful scientific
predictions. CPU execution (`--device cpu`) is a compatibility check only:
the 32-neighbor-capped CPU graph differs from the CUDA graph used by the
pretrained models.

## Simulation input examples

- `demo/example_inputs/lammps/Equal.in` expects a separate `PP.data` structure,
  topology, and force-field file, which is not included.
- `demo/example_inputs/orca/nmr.inp` uses TPSS/PCSSEG-1 and expects a separate
  `file.xyz` (charge 0, multiplicity 1; target carbon first). The template
  requests eight processes and `maxcore 4000`.

These inputs document the upstream workflow; the Python demos use the supplied
processed geometries and labels.

## License

Source code: GPL-3.0-only. See [LICENSE](LICENSE) and
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for source attribution and
modifications.
