"""Render and evaluate an existing ReScene stored visit at its held-out cameras."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from rescene.cli import ROOT
if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subset',choices=['synthetic','real'],required=True)
    p.add_argument('--run',type=Path,required=True,help='Prepared scene/full output directory')
    p.add_argument('--visit',type=int,required=True,help='Zero-based stored visit index')
    p.add_argument('--gpu',type=int,default=0)
    a=p.parse_args();run=a.run.resolve()
    cfg=json.loads((run/'config.json').read_text());assert a.visit in cfg['periods']
    tag='rescene-synthetic-photo-v1' if a.subset=='synthetic' else 'rescene-real-photo-v1'
    protocol=json.loads((ROOT/'artifacts'/tag/'protocol.json').read_text())
    import hashlib
    for name,digest in protocol['sources'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    worker=ROOT/'artifacts'/tag/'implementation/worker.py'
    assert worker.exists(),'Run dataset preparation before rendering'
    subprocess.run([sys.executable,str(worker),'evaluate','--run',str(run),'--period',str(a.visit)],cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(a.gpu)},check=True)
