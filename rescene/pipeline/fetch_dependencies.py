"""Fetch pinned sources without recursive cloning unrelated baseline components."""
from pathlib import Path
import json,subprocess
R=Path(__file__).resolve().parents[2]
def run(*cmd,**kw):subprocess.run(list(map(str,cmd)),check=True,**kw)
for item in json.loads((R/'reproduction/upstream-lock.json').read_text()):
 dst=R/item['path'];dst.parent.mkdir(parents=True,exist_ok=True)
 if not (dst/'.git').exists():
  if dst.exists() and any(dst.iterdir()):raise RuntimeError(f'Refusing to replace nonempty directory: {dst}')
  run('git','init',dst);run('git','-C',dst,'remote','add','origin',item['url']);run('git','-C',dst,'fetch','--depth','1','origin',item['commit']);run('git','-C',dst,'checkout','--detach','FETCH_HEAD')
 actual=subprocess.check_output(['git','-C',str(dst),'rev-parse','HEAD'],text=True).strip();assert actual==item['commit'],dst
repo=R/'external/LTGS/gaussian-splatting';patch=R/'patches/gaussian-dataset-reader.patch'
if patch.stat().st_size:
 reverse=subprocess.run(['git','-C',str(repo),'apply','--reverse','--check',str(patch)],capture_output=True)
 if reverse.returncode:run('git','-C',repo,'apply','--check',patch);run('git','-C',repo,'apply',patch)
print('Pinned runtime dependencies ready; upstream licenses remain in each checkout.')
