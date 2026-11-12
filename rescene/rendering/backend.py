"""Isolated LTGS renderer and WAT adapter. Import before renderer/utils modules."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
LTGS = ROOT / 'external/LTGS'
RASTERIZER = ROOT / 'external/periodic-rasterizer-install'
if not (RASTERIZER / 'diff_gaussian_rasterization').is_dir():
    raise RuntimeError('Run bash examples/setup.sh for the isolated visibility-safe renderer')
sys.path[:0] = [str(RASTERIZER), str(LTGS / 'gaussian-splatting'), str(LTGS)]
import numpy as np
from PIL import Image
import torch
from gaussian_renderer import GaussianModel, render
from scene.cameras import Camera
from utils.graphics_utils import focal2fov

spec = importlib.util.spec_from_file_location('periodic_colmap', LTGS / 'submodules/Hierarchical-Localization/hloc/utils/read_write_model.py')
rw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rw)
PIPE = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=False)


class WAT:
    def __init__(self, source, protocol):
        self.source, self.protocol = Path(source), protocol
        self.cameras = rw.read_cameras_binary(self.source / 'sparse/0/cameras.bin')
        self.images = {im.name: im for im in rw.read_images_binary(self.source / 'sparse/0/images.bin').values()}
        if set(protocol['update_names']) & set(protocol['test_names']):
            raise ValueError('Train/test overlap')

    def names(self, period, split):
        if split not in ('train', 'test'):
            raise ValueError(split)
        names = self.protocol['update_names' if split == 'train' else 'test_names']
        result = sorted(n for n in names if n.startswith(f'IMG_{period:04d}/'))
        if not result:
            raise ValueError(f'No {split} images for period {period}')
        return result

    def camera(self, name):
        im = self.images[name]
        c = self.cameras[im.camera_id]
        if c.model != 'PINHOLE':
            raise ValueError(c.model)
        fx, fy, cx, cy = c.params
        if abs(cx - c.width / 2) > 1e-3 or abs(cy - c.height / 2) > 1e-3:
            raise ValueError('Renderer requires centered principal point')
        scale = max(1., c.width / 1600.)
        resolution = (int(c.width / scale), int(c.height / scale))
        return Camera(resolution, im.id, im.qvec2rotmat().T, im.tvec,
                      focal2fov(fx, c.width), focal2fov(fy, c.height), None,
                      Image.new('RGB', (c.width, c.height)), None, name, im.id, data_device='cpu')

    def rgb(self, name, cam):
        with Image.open(self.source / 'images' / name) as im:
            im = im.convert('RGB').resize((cam.image_width, cam.image_height), Image.Resampling.BICUBIC)
            return torch.from_numpy(np.array(im).copy()).permute(2, 0, 1).float().div(255).cuda()


def draw(cam, view):
    return render(cam, view, PIPE, torch.zeros(3, device='cuda'), separate_sh=True,
                  use_trained_exp=False)['render']
