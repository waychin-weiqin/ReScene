"""Detached compositing support from current training RGB only; no ROI/depth gate."""
import torch,numpy as np
from scipy.ndimage import median_filter,label
from common import view,render,PIPE

def build(old,names,cams,targets,refs,cfg):
 C=torch.zeros(len(old['xyz']),device='cuda');S=torch.zeros_like(C);masks={}
 for n in names:
  with torch.no_grad():
   diff=targets[n]-refs[n];r=diff.abs().mean(0);low=r<=r.median();offset=diff[:,low].median(1).values.clamp(-.1,.1)
   r=(diff-offset[:,None,None]).abs().mean(0)
   binary=median_filter(r.cpu().numpy(),size=3)>cfg['residual_threshold'];cc,num=label(binary);sizes=np.bincount(cc.ravel());keep=sizes>=cfg['minimum_component'];keep[0]=False
   change=torch.as_tensor(keep[cc],device='cuda');stable=(r<cfg['stable_threshold'])&~change
   masks[n]=(change.float(),stable.float())
  colors=torch.ones_like(old['xyz'],requires_grad=True)
  pred=render(cams[n],view(old),PIPE,torch.zeros(3,device='cuda'),separate_sh=False,override_color=colors)['render'][0]
  C+=torch.autograd.grad((pred*change).sum(),colors,retain_graph=True)[0][:,0].detach()/len(names)
  S+=torch.autograd.grad((pred*stable).sum(),colors)[0][:,0].detach()/len(names)
 w=C/(C+S+1e-8);ids=torch.where(C>1e-6)[0]
 evidence=torch.stack([w,C/(C+1),S/(S+1)],1)
 return ids,evidence,masks,dict(eligible=len(ids),numerical_support_cutoff=1e-6,weight_median=float(w[ids].median()),weight_mean=float(w[ids].mean()),weight_above_09=int((w[ids]>.9).sum()),rule='effective gate=1-w*(1-sigmoid); C/(C+S) compositing support; low weights cap suppression',no_heldout_or_roi=True)
