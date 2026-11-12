#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Activate the chronogs environment first. CUDA_HOME must point to a CUDA 12.4 toolkit.
python rescene/pipeline/fetch_dependencies.py
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
export MAX_JOBS="${MAX_JOBS:-4}"
for module in diff-gaussian-rasterization simple-knn fused-ssim flashsplat-rasterization; do
 python -m pip install --no-build-isolation --no-deps "external/LTGS/gaussian-splatting/submodules/$module"
done
python - <<'PY'
from pathlib import Path
import shutil,subprocess
src=Path('external/LTGS/gaussian-splatting/submodules/diff-gaussian-rasterization');dst=Path('external/periodic-rasterizer-src')
if not dst.exists():shutil.copytree(src,dst,ignore=shutil.ignore_patterns('.git','build','*.egg-info','*.so','__pycache__'))
patch=Path('patches/periodic-gs/empty-visibility-rasterizer.patch').resolve()
with patch.open('rb') as f:done=subprocess.run(['patch','--dry-run','--reverse','-p1','-d',str(dst)],stdin=f,capture_output=True).returncode==0
if not done:
 with patch.open('rb') as f:subprocess.run(['patch','-p1','-d',str(dst)],stdin=f,check=True)
PY
python -m pip install --upgrade --no-build-isolation --no-deps --target external/periodic-rasterizer-install external/periodic-rasterizer-src
# The validated environment uses the PyTorch RoPE fallback; do not enable a different kernel here.
python rescene/pipeline/check_reproduction.py --runtime
