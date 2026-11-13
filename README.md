# ReScene: Incremental Scene Chronology from Sparse Revisits

ReScene reconstructs changing scenes from sparse revisits. Per-Gaussian change and stability evidence guides suppression, refinement and the addition of new surfaces. Each visit is stored as an update, allowing earlier reconstructions to be recovered.

![ReScene methodology](assets/workflow.png)

## Installation

Use Linux with an NVIDIA GPU and CUDA 12.4. The environment uses Python 3.10 and PyTorch 2.5.1.

```bash
git clone https://github.com/waychin-weiqin/ReScene.git
cd ReScene
conda create -n chronogs python=3.10 pip -y
conda activate chronogs
export CUDA_HOME=/usr/local/cuda-12.4
bash examples/setup.sh
```

The setup script installs the dependencies and builds the renderer. MASt3R and LPIPS weights are downloaded on first use.

## Dataset

Extract the Multi-State release and retain its original images, camera files and training/test splits.

| Subset | Scenes | Dataset directory |
|---|---|---|
| Synthetic | Robot, Teddy, Plant, Cloth | `multistate_hard` |
| Real | Can, Tissue, Tomato, Gimbal | `wei_real_multistate_20261030` |

See the [synthetic](examples/synthetic.md) and [real](examples/real.md) guides for the directory layouts and camera preparation.

## Quick start

Prepare the dataset, then train and evaluate the sequence.

### Synthetic Multi-State

```bash
python examples/synthetic.py prepare --dataset-root /path/to/multistate_hard --gpus 0
python examples/synthetic.py run --gpus 0
```

### Real Multi-State

```bash
python examples/real.py prepare --dataset-root /path/to/wei_real_multistate_20261030 --gpus 0
python examples/real.py run --gpus 0
```

To train a single scene, add `--scenes robot_transfer` or `--scenes can` to both commands. Run commands from the repository root. Allow at least 100 GiB of free disk space for synthetic runs and 104 GiB for real runs.

The initial model uses all training images of the first visit. Each later update uses up to 24 images and 5,000 optimisation steps. The update objective combines masked photometric losses with image preservation. New Gaussians are initialised from MASt3R depth and split up to a limit of 60,000 per visit.

## Evaluation

Training renders the held-out images and checks that earlier visits can be recovered. Generate the results summary with:

```bash
python eval/multistate.py --subset synthetic
python eval/multistate.py --subset real
```

Reports include PSNR, SSIM, LPIPS and history storage. If training a scene subset, pass the same `--scenes` list to evaluation. See [evaluation](eval/README.md) for output locations and [rendering](demo/README.md) to recover a stored visit.

## Licence

See [licensing information](LICENSE) and [third-party notices](LICENSES/README.md).
