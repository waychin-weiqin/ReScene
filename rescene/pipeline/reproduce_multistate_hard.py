"""Portable entry point; original worker numerics and benchmark budgets retained."""
import argparse,os,subprocess,sys,json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','smoke','run','report']);p.add_argument('--dataset-root',type=Path);p.add_argument('--scenes',nargs='+',default=['robot_transfer','teddy_coupled','plant_motion','cloth_fold'],choices=['robot_transfer','teddy_coupled','plant_motion','cloth_fold']);p.add_argument('--gpus',default='0');a=p.parse_args()
env={**os.environ,'GSVER_SCENES':','.join(a.scenes),'GSVER_GPUS':a.gpus}
if a.action=='prepare':
 if a.dataset_root is None:p.error('prepare requires --dataset-root (directory containing dataset/, manifest.json and SHA256SUMS)')
 src=a.dataset_root.expanduser().resolve();assert all((src/f).exists() for f in ['dataset','manifest.json','SHA256SUMS']),src
 expected=json.loads((R/'rescene/reproduction/dataset-release.json').read_text())
 for f,key in [('manifest.json','manifest_sha256'),('SHA256SUMS','sha256sums_sha256')]:assert hashlib.sha256((src/f).read_bytes()).hexdigest()==expected[key],f'Dataset release differs: {f}'
 link=R/'data/multistate_synthetic';link.parent.mkdir(exist_ok=True)
 if link.exists():assert link.resolve()==src,'Refusing to replace dataset link'
 else:link.symlink_to(src,target_is_directory=True)
 script='prepare_multistate_hard.py'
else:
 assert (R/'artifacts/rescene-synthetic-photo-v1/protocol.json').exists(),'Run prepare first'
 script='report_multistate_hard.py' if a.action=='report' else 'run_multistate_hard.py'
 if a.action=='smoke':env['GSVER_SMOKE_ONLY']='1'
subprocess.run([sys.executable,str(R/'rescene/pipeline'/script)],cwd=R,env=env,check=True)
