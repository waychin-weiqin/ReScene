"""Portable ReScene entry point for the four accepted real Multi-State scenes."""
import argparse,os,subprocess,sys,json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('action',choices=['prepare','smoke','run','report'])
p.add_argument('--dataset-root',type=Path,help='Extracted wei_real_multistate_20261030 directory containing dataset/ and SHA256SUMS')
p.add_argument('--scenes',nargs='+',choices=['can','tissue','tomato','gimbal'],default=['can','tissue','tomato','gimbal'])
p.add_argument('--gpus',default='0')
a=p.parse_args()
try:gpus=[int(x) for x in a.gpus.split(',')];assert gpus and min(gpus)>=0 and len(set(gpus))==len(gpus)
except (ValueError,AssertionError):p.error('--gpus must contain distinct nonnegative GPU IDs')
env={**os.environ,'RESCENE_SCENES':','.join(a.scenes),'RESCENE_GPUS':a.gpus}
if a.action=='prepare':
 if a.dataset_root is None:p.error('prepare requires --dataset-root')
 source=a.dataset_root.expanduser().resolve()
 if not (source/'dataset').is_dir() or not (source/'SHA256SUMS').is_file():p.error('Expected dataset/ and SHA256SUMS')
 link=R/'data/real-multistate/wei_real_multistate_20261030';link.parent.mkdir(parents=True,exist_ok=True)
 if link.exists():assert link.resolve()==source,'Refusing to replace the existing dataset link'
 else:link.symlink_to(source,target_is_directory=True)
 script='prepare_real_multistate.py'
else:
 protocol=R/'artifacts/rescene-real-photo-v1/protocol.json'
 assert protocol.exists(),'Run prepare first'
 frozen=json.loads(protocol.read_text());assert frozen['scenes']==a.scenes,'Use the scene selection frozen at prepare'
 if a.action!='report':assert frozen['gpus']==gpus,'Use the GPU selection frozen at prepare'
 script='report_real_multistate.py' if a.action=='report' else 'run_real_multistate.py'
 if a.action=='smoke':env['RESCENE_SMOKE_ONLY']='1'
subprocess.run([sys.executable,str(R/'rescene/pipeline'/script)],cwd=R,env=env,check=True)
