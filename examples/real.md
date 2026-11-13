# Real Multi-State

The real subset contains Can, Tissue, Tomato and Gimbal, each with four visits.

## Data preparation

Use the original release with this layout:

```text
wei_real_multistate_20261030/
  SHA256SUMS
  dataset/
    can/t000/{images,sparse/0,train.txt,test.txt}
    can/t001/...
    tissue/...
    tomato/...
    gimbal/...
```

The supplied transforms align each visit to the initial coordinate frame using persistent background structure. Preparation verifies the camera and split files before applying these transforms. Use the original camera files rather than an already transformed dataset.

The supplied SfM reconstruction used all images, including held-out views. Background alignment uses training images across visits. Camera registration is therefore performed offline. The scenes have arbitrary scale and no ground-truth geometry.

## Training

After [installation](../README.md#installation), run:

```bash
python examples/real.py prepare --dataset-root /path/to/wei_real_multistate_20261030 --gpus 0
python examples/real.py run --gpus 0
python eval/multistate.py --subset real
```

Add `--scenes can` to each command to run one scene. Keep at least 104 GiB free during training. Preparation fixes the selected cameras and configuration. Use a fresh workspace when changing these settings.

The initial reconstruction uses all initial training images and SfM points observed in at least two of those images. Initial fitting uses 30,000 nominal iterations, corresponding to 29,999 optimiser steps in the upstream loop.

Each later visit uses up to 24 training images selected by farthest-camera-centre sampling, with 5,000 Adam steps at a learning rate of 0.001 and seed 0. MASt3R estimates depth with the registered cameras fixed. The photometric and image-preservation losses, birth limits and splitting schedule are the same as in the [synthetic configuration](synthetic.md#training).

## Outputs

- `outputs/rescene-real-photo-v1/rescene/<scene>/initial_model/` contains the initial fit.
- `outputs/rescene-real-photo-v1/rescene/<scene>/full/` contains the initial state, visit updates and held-out renders.
- `artifacts/rescene-real-photo-v1/` contains the evaluation reports and run configuration.

The four scenes contain 214 held-out images. Evaluation reports arrival and final recall separately and checks that saved reconstructions remain unchanged. See [evaluation](../eval/README.md) and [rendering a visit](../demo/README.md).
