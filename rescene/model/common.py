import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'rescene/rendering'))
from backend import WAT,GaussianModel,draw,PIPE,render
import torch
torch.backends.cuda.matmul.allow_tf32=True
from types import SimpleNamespace
KEYS=('xyz','features_dc','features_rest','scaling','rotation','opacity')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def save(p,x):
 p=Path(p);assert not p.exists(),p;tmp=p.with_suffix('.tmp');torch.save(x,tmp);tmp.replace(p)
def cpu(x):return {k:v.detach().cpu() for k,v in x.items()}
def view(x):
 return SimpleNamespace(active_sh_degree=3,max_sh_degree=3,get_xyz=x['xyz'],get_features_dc=x['features_dc'],get_features_rest=x['features_rest'],get_scaling=x['scaling'].exp(),get_rotation=torch.nn.functional.normalize(x['rotation'],dim=-1),get_opacity=x['opacity'])
def state(run,t):
 x=torch.load(run/'initial.pt',map_location='cuda',weights_only=True)
 for j in range(1,t+1):
  s=torch.load(run/f'period_{j:04d}.pt',map_location='cuda',weights_only=True)
  assert s['period']==j and len(s['old_opacity'])==len(x['xyz'])
  
  if 'refined' in s:
   for k,v in s['refined'].items():x[k]=x[k].index_copy(0,s['refine_ids'].to(x[k].device),v)
  x['opacity']=s['old_opacity'];x={k:torch.cat((x[k],s['birth'][k]),0) for k in KEYS}
 return x
def guard(source,names):
 allowed={(source/'images'/n).resolve() for n in names};accessed=set()
 def audit(event,args):
  if event=='open' and isinstance(args[0],(str,bytes)):
   p=Path(args[0].decode() if isinstance(args[0],bytes) else args[0]).resolve()
   if p.suffix.lower() in ('.png','.jpg','.jpeg'):
    if p not in allowed:raise PermissionError(f'Training RGB isolation: {p}')
    accessed.add(str(p))
 sys.addaudithook(audit);return accessed

