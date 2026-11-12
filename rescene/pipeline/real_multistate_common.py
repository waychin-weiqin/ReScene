from pathlib import Path
import json,hashlib,os
R=Path(__file__).resolve().parents[2];TAG='rescene-real-reproduction';A=R/'artifacts'/TAG;O=R/'outputs'/TAG/'rescene';D=R/'data'/TAG
ALL_SCENES=['can','tissue','tomato','gimbal']
SCENES=os.environ.get('RESCENE_SCENES',','.join(ALL_SCENES)).split(',')
assert SCENES and set(SCENES)<=set(ALL_SCENES)
STATE_COUNTS={s:4 for s in SCENES}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False));tmp.replace(p)
def freeze():
 p=read(A/'protocol.json')
 for f,h in p['sources'].items():assert sha(f)==h,f
 return p
