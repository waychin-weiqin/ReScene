import sys,json,random
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'rescene/model'))
from common import *
from change_evidence import prepare
from soft_eligibility import build
from refinement import select
import numpy as np
run=Path(sys.argv[1]);t=int(sys.argv[2]);cfg=json.loads((run/'config.json').read_text());cfg['max_births']=30000
torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0);random.seed(0);src=Path(cfg['source']);data=WAT(src,json.loads((src/'wat_protocol.json').read_text()));names=data.names(t,'train');seen=guard(src,names);old=state(run,t-1);cams={n:data.camera(n) for n in names};targets={n:data.rgb(n,cams[n]) for n in names};depth=Path(cfg['depth_root'])/f'depth_{t:04d}.npz'
seeds,baseids,e,center,extent,stats,refs,base_masks,bs=prepare(old,names,cams,targets,depth,cfg);ids,e,masks,soft=build(old,names,cams,targets,refs,cfg);original,rm,rs=select(old,names,cams,depth,base_masks,baseids,cfg);supported=(e[:,1]>1e-6)|(e[:,2]>1e-6);supported[original]=True;ri=supported.nonzero().flatten();rs.update(expanded_selected=len(ri))
cache=dict(seeds=cpu(seeds),ids=ids.cpu(),evidence=e.cpu(),center=center.cpu(),extent=extent.cpu(),stats=stats,refs=cpu(refs),masks={n:tuple(v.cpu() for v in pair) for n,pair in masks.items()},refine_masks=cpu(rm),refine_ids=ri.cpu(),refine_stats=rs,birth_support=bs.cpu());assert set(seen)=={str((src/'images'/n).resolve()) for n in names};save(run/f'prepared_{t:04d}.pt',cache);dump(run/f'prepare_audit_{t:04d}.json',dict(names=names,accessed=sorted(seen),cache_sha=sha(run/f'prepared_{t:04d}.pt'),previous_sha=sha(run/('initial.pt' if t==1 else f'period_{t-1:04d}.pt')),initial_births=len(seeds['xyz'])))
