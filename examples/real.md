# Reproducing ReScene on real Multi-State

This pipeline includes ReScene only. It uses the accepted Can, Tissue, Tomato and Gimbal sequences, each with four visits. Shirt and Hulkbuster are excluded. Use the environment and pinned dependencies in [the synthetic reproduction guide](synthetic.md).

## Inputs and calibration

Obtain the original `wei_real_multistate_20261030` release separately. The extracted directory must contain `SHA256SUMS` and `dataset/<scene>/t000/` through `t003/`, with images, COLMAP `sparse/0` binaries, `train.txt` and `test.txt`. Datasets and models are not included in this repository.

`reproduction/real-alignment/` contains the accepted transforms into the initial coordinate frame and their reserved-correspondence reprojection checks. Can, Tissue and Tomato use alignment v2; Gimbal uses v1. The adapter verifies the camera, point and split file hashes before applying these transforms, checks camera projection invariance and assigns distinct camera IDs across visits. Do not apply these transforms to an already aligned or corrected dataset.

The supplied SfM used all images, including held-out views. Alignment used training images from multiple visits and persistent background structure. This is offline registration, not causal pose estimation. Scale is arbitrary and there is no ground-truth real geometry. These limitations apply to the reported results.

## Frozen benchmark settings

- All initial training images, with initial SfM points observed in at least two initial training views. Later SfM points do not seed the initial model.
- Standard 3DGS initial fitting for 30,000 nominal iterations (29,999 optimiser steps under the retained upstream loop).
- Up to 24 current training images at each later visit, chosen by a lexical first camera followed by farthest-camera-centre selection. Official held-out splits are preserved.
- MASt3R depth from current images with fixed registered cameras, crop-aware intrinsics and invalidation outside the crop.
- 5,000 update steps, seed 0, original simple60 configuration, 30,000 initial birth proposals, 60,000 maximum births and eight splitting rounds.
- The original benchmark objective includes its regularisation terms. It must not be described as reproducing a photometric-plus-preservation-only variant. No experimental unbounded gate, mask replacement or birth-limit sweep is substituted for the benchmark method.

## Commands

Run from this repository in the `chronogs` environment. Choose the scene list and GPU IDs during preparation and retain them for subsequent commands. No training starts during preparation.

```bash
python examples/real.py prepare \
  --dataset-root /path/to/wei_real_multistate_20261030 --gpus 0
python examples/real.py smoke --gpus 0
python examples/real.py run --gpus 0
python examples/real.py report
```

The smoke action performs the full initial fit, then two three-step updates per scene. It is not a quick CPU test. The run action reuses validated stages and continues the full histories. A failed stage blocks unchanged retries. Up to two GPU workers use shared leases; at least 104 GiB of free storage is required. Run preparation once per frozen configuration. Source changes require a documented migration or a fresh checkout/output root.

Outputs are written to `outputs/rescene-real-reproduction/rescene/`. Protocols, source snapshots and reports are in `artifacts/rescene-real-reproduction/`. The run evaluates the initial model and re-renders all previous visits after each update, checking saved history. The report validates held-out identities, original RGB access, budgets, historical hashes, finite metrics and recomputed PSNR. PSNR, SSIM and LPIPS are available by visit and averaged equally across visits and scenes; initial-included and later-only results are separate. History size includes the initial model once and every update. Arrival and final archival recall coincide only when exact-history checks pass.

This portable integration was checked against the validated research implementation without launching another training experiment. See `reproduction/real-portability-audit.json` and `reproduction/release-validation.json` for the precise checks. It is not a claim of a fresh end-to-end reproduction.
