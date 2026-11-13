# Reproducing ReScene on synthetic Multi-State

This package reproduces **ReScene’s original versioned simple60 benchmark method**. Baseline training code is not included. The configuration retains the original regularisers used to generate the reported benchmark results; it is not a photometric-plus-preservation-only variant. The internal dataset directory remains `multistate_hard`.

## Protocol

| Setting | Value |
|---|---|
| Scenes | Robot Transfer (12 states); Teddy Coupled, Plant Motion, Cloth Fold (10 each) |
| Initialization | Standard 3DGS, all 52 period-0 training images |
| Initial support | Supplied synthetic training-depth point cloud, filtered to ≥2 initial training tracks |
| Initial training | 30,000 nominal iterations / 29,999 optimizer steps, upstream final-save behavior retained |
| Later observations | 24 current training views, deterministic farthest-camera-center selection |
| Later support | MASt3R depth from current images only; no later ground-truth depth, masks, or meshes |
| Update training | 5,000 steps per visit; seed 0; Adam learning rate 0.001 |
| Births | At most 30,000 initial proposals; splitting up to 60,000 new Gaussians per update |
| Splitting | Eight opportunities: 1000, 1500, …, 4500 |
| Evaluation | 8 official held-out views per state; 336 total; all seen states rerendered after each update |
| History | Initial state plus immutable update records; growing storage |

The cameras are exact synthetic poses. Initialization uses oracle training-depth support, so this is **not image-only initialization**. RGB capture is isolated by training-stage access guards. Preparation and evaluation may read held-out images; optimization may not.

## 1. Environment and pinned dependencies

Linux, an NVIDIA GPU, a compatible driver, a CUDA 12.4 development toolkit (`nvcc`), Git, a C++ compiler, `patch`, and Conda are required. The reference runtime uses Python 3.10 and PyTorch 2.5.1 + CUDA 12.4. Actual peak GPU memory depends on the scene and MASt3R matching stage. Keep at least **100 GiB free** throughout the run; the runner enforces this reserve, which is not an estimate of total storage consumption.

```bash
conda create -n chronogs python=3.10 pip -y
conda activate chronogs
export CUDA_HOME=/usr/local/cuda-12.4  # adjust to your toolkit
bash examples/setup.sh
```

`rescene/reproduction/upstream-lock.json` pins LTGS's 3DGS fork, MASt3R/DUSt3R/CroCo, COLMAP-format readers, and rasterization dependencies. Setup applies the recorded dataset-reader integration patch and builds a separate visibility-safe rasterizer. Upstream licenses remain in those checkouts. The reference environment uses the PyTorch RoPE fallback; setup does not switch to the optional CUDA RoPE kernel.

MASt3R (`naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric`) and LPIPS/VGG weights are fetched through their libraries on first use. Obtain these weights before setting `HF_HUB_OFFLINE=1`. Pretrained weights are not committed here. `pyproject.toml` records the relevant versions from the reference environment; `rescene/reproduction/tested-packages.json` records the observed versions, including PyTorch.

## 2. Obtain the dataset

The Multi-State Hard dataset is distributed separately; this repository does not contain it and no public download URL is asserted. Use the supplied release with this structure:

```text
/path/to/multistate_hard/
  manifest.json
  SHA256SUMS
  dataset/
    robot_transfer/t000/{images,sparse/0,...}
    robot_transfer/t001/...
    teddy_coupled/...
    plant_motion/...
    cloth_fold/...
```

The original dataset release includes 60 images per state (52 train / 8 test). Do not change the official split or substitute rendered predictions for input images. The preparation command creates an isolated adapter and records input hashes; it never modifies the source dataset.

## 3. Prepare and run

From the repository root, inside `chronogs`:

```bash
python examples/synthetic.py prepare \
  --dataset-root /path/to/multistate_hard --gpus 0

# Initial fit, then two 3-step update smoke tests per scene.
# The initial fit is still the full 30k nominal fit, so this is not a quick CPU test.
python examples/synthetic.py smoke

# Reuses validated initial fits/smokes, then completes the full histories.
python examples/synthetic.py run

# Read-only validation and reporting after all scenes finish.
python examples/synthetic.py report
```

The runner is serial: `--gpus 0,1` supplies candidate devices to lease, not two simultaneous workers. Physical GPU IDs are selected during `prepare`; keep visibility consistent across subsequent commands. Use `--scenes robot_transfer` during preparation for an isolated single-scene reproduction. Prepare a fresh checkout/workspace to change a frozen cohort or configuration.

Completed stage markers are reused. Failed stages are **not retried unchanged**. Inspect their logs and failure markers; preserve evidence and document any repair. Do not relax history, isolation, or finite-value assertions to obtain a result.

## 4. Outputs and evaluation

- `artifacts/multistate_hard-simple60-v1/protocol.json`: frozen settings, sources, dataset hashes and camera selection.
- `outputs/multistate_hard-simple60-v1/<scene>/initial_model/`: initial 3DGS fit and isolation audit.
- `.../<scene>/smoke/`: separate technical tests, never reported as benchmark results.
- `.../<scene>/full/initial.pt` and `period_XXXX.pt`: deployment history.
- `.../<scene>/full/renders/`: held-out images, verified unchanged when historical states are revisited.
- `artifacts/multistate_hard-simple60-v1/results.json` and `report.md`: per-view and per-period PSNR/SSIM/LPIPS and storage.
- `outputs/multistate_hard-simple60-v1/comparison/index.html`: GT-versus-ours qualitative gallery.

PSNR is evaluated on quantized full-resolution RGB, SSIM and VGG-LPIPS use the same common held-outs. Scene summaries average periods equally; cross-scene summaries should average scenes equally. Because our historical renders are checked for exact equality, final recall equals arrival for each state. This is achieved by storing historical updates, not by proving that one mutable model never forgets.

The original four-scene mean arrival PSNR was **32.71 dB** (including period 0); Robot/Teddy/Plant/Cloth means were **36.14 / 32.45 / 30.87 / 31.39 dB**. These are reference observations, not acceptance thresholds. CUDA execution can vary across reruns and hardware; do not tune to these scores. Separate ground-truth geometry evaluation is not part of this reproduction entry point.

## Implementation map

- `rescene/model/`: original model, retirement, refinement, evidence, splitting, serialization and optimization.
- `rescene/pipeline/prepare_simple60.py`: current-view evidence and birth proposal preparation.
- `rescene/pipeline/multistate_hard_adapter.py`: supplied synthetic-data/COLMAP adapter.
- `rescene/pipeline/multistate_hard_depth.py`: fixed-camera MASt3R depth, including resize/crop intrinsics and confidence remapping.
- `rescene/pipeline/mast3r_depth_support.py`: the called LTGS reconstruction helper/class copied verbatim; unrelated localization imports are omitted.
- `rescene/pipeline/multistate_hard_initial_entry.py`: period-0-only training access guard.
- `rescene/pipeline/run_multistate_hard.py`: smoke gates, full updates and historical rerender checks.

The worker's WAT-specific evaluation paths are replaced at preparation with this dataset's common-GT directory; its training body is unchanged. `rescene/reproduction/portability-audit.json` documents copied source hashes and packaging changes.

## Validation limits

See `rescene/reproduction/validation.json` for the checks actually run on this package. A short smoke test does not constitute a new full benchmark or a clean-machine CUDA build. The reported complete benchmark was produced before this packaging step. Dataset, model and cache storage remain excluded from the Git repository.
