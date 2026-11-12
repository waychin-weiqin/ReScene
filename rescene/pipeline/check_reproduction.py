"""CPU configuration checks; dependency import checks when --runtime is supplied."""
from pathlib import Path
import argparse,ast,json,sys
R=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--runtime',action='store_true');a=p.parse_args()
for folder in ['rescene','examples','eval','demo']:
 for f in (R/folder).rglob('*.py'):ast.parse(f.read_text(),filename=str(f))
c=json.loads((R/'configs/multistate_hard.json').read_text());assert c['iterations']==5000 and c['max_births']==60000 and c['split_steps']==list(range(1000,4501,500))
sys.path.insert(0,str(R/'rescene/pipeline'));from multistate_hard_depth import test;test()
if a.runtime:
 sys.path.insert(0,str(R/'rescene/model'));from common import GaussianModel,render
 import torch,lpips
 sys.path[:0]=[str(R/'external/LTGS/submodules/mast3r')]
 from mast3r.model import AsymmetricMASt3R
 from mast3r_depth_support import get_reconstructed_scene
 print('Runtime imports passed; CUDA available:',torch.cuda.is_available())
print('Configuration, Python syntax and crop/intrinsics checks passed.')
