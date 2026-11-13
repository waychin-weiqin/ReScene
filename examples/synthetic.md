# Synthetic Multi-State

The synthetic subset contains Robot, Teddy, Plant and Cloth. Robot has 12 visits and the other scenes have 10 each. Each visit provides 52 training images and eight held-out images.

## Data preparation

Extract the release into this layout:

```text
multistate_synthetic/
  manifest.json
  dataset/
    robot_transfer/t000/{images,sparse/0,...}
    robot_transfer/t001/...
    teddy_coupled/...
    plant_motion/...
    cloth_fold/...
```

Preparation uses the supplied camera poses and splits. The initial point cloud comes from ground-truth depth of the initial training views. Later visits use MASt3R depth estimated from their training images.

## Training

After [installation](../README.md#installation), run:

```bash
python examples/synthetic.py prepare --dataset-root /path/to/multistate_synthetic --gpus 0
python examples/synthetic.py run --gpus 0
python eval/multistate.py --subset synthetic
```

Add `--scenes robot_transfer` to each command to run one scene. Preparation records the selected cameras and configuration. Use a fresh workspace when changing these settings.

The initial 3DGS fit uses all 52 initial training images and 30,000 nominal iterations. The upstream loop performs 29,999 optimiser steps. Later visits use 24 images selected by farthest-camera-centre sampling and 5,000 Adam steps at a learning rate of 0.001, with seed 0.

The update loss contains photometric supervision in changed, stable and refinement regions, together with image preservation outside refinement regions. Photometric supervision combines L1 and SSIM dissimilarity with weights 0.8 and 0.2. Each region is normalised by its own mask weight. Image preservation penalises the L1 difference from the preceding reconstruction under the same appearance correction.

Up to 30,000 new Gaussians are seeded per visit. Splitting occurs every 500 steps from step 1,000 to 4,500, up to 60,000 births. Birth positions and scales remain bounded around their initial values.

## Outputs

- `outputs/rescene-synthetic-photo-v1/<scene>/initial_model/` contains the initial fit.
- `outputs/rescene-synthetic-photo-v1/<scene>/full/` contains the initial state, visit updates and held-out renders.
- `artifacts/rescene-synthetic-photo-v1/results.json` and `report.md` contain the evaluation results.

The stored initial state and visit updates recover each reconstruction without optimising it again. See [rendering a visit](../demo/README.md).
