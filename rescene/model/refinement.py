"""Current-view, depth-supported soft refinement; no held-out evidence."""
import numpy as np
import torch
import torch.nn.functional as F
from common import view,render,PIPE
from change_evidence import project

@torch.no_grad()
def select(old,names,cams,depthfile,masks,retirement_ids,cfg):
 d=np.load(depthfile);assert d['names'].tolist()==names
 device=old['xyz'].device
 dep=torch.as_tensor(d['depth'],device=device);conf=torch.as_tensor(d['conf'],device=device)
 poses=torch.as_tensor(d['poses'],device=device,dtype=torch.float32);Ks=torch.as_tensor(d['intrinsics'],device=device,dtype=torch.float32)
 h,w=dep.shape[-2:];votes=torch.zeros(len(old['xyz']),device=device,dtype=torch.int32);changed=torch.zeros_like(votes,dtype=torch.bool);valid_maps={}
 for i,n in enumerate(names):
  out=render(cams[n],view(old),PIPE,torch.zeros(3,device=device),separate_sh=True)
  alpha=render(cams[n],view(old),PIPE,torch.zeros(3,device=device),separate_sh=False,override_color=torch.ones_like(old['xyz']))['render'][0]
  od=F.interpolate((alpha/out['depth'].squeeze().clamp_min(1e-8))[None,None],(h,w),mode='nearest')[0,0]
  change=F.interpolate(masks[n][0][None,None],(h,w),mode='nearest')[0,0]>.5
  good=(conf[i]>cfg['depth_confidence'])&(dep[i]>0)&~change&((od-dep[i]).abs()<=cfg['multiview_tolerance']*dep[i])
  z,u,v,ok=project(old['xyz'],poses[i],Ks[i],h,w)
  visible=ok&((z-od[v,u]).abs()<=cfg['multiview_tolerance']*od[v,u])&(old['opacity'][:,0]>.005)
  votes+=visible&good[v,u];changed|=visible&change[v,u]
  valid_maps[n]=F.interpolate(good.float()[None,None],masks[n][0].shape,mode='bilinear',align_corners=False)[0,0]
 keep=(votes>=cfg['minimum_views'])&~changed;keep[retirement_ids]=False;ids=keep.nonzero().flatten()
 color=keep.float()[:,None].expand(-1,3).contiguous();refine={}
 for n in names:
  coverage=render(cams[n],view(old),PIPE,torch.zeros(3,device=device),separate_sh=False,override_color=color)['render'][0].clamp(0,1)
  refine[n]=(coverage*valid_maps[n]).detach()
 return ids,refine,dict(selected=int(len(ids)),total=len(keep),multi_view_depth_supported=int((votes>=cfg['minimum_views']).sum()),change_veto=int(changed.sum()),mask_means={n:float(m.mean()) for n,m in refine.items()})
