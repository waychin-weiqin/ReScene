"""Per-visit implicit activation field plus explicitly seeded birth geometry."""
import torch,math
from torch import nn
from common import view
def encode(x):
 return torch.cat([x,*[f(x*(2**i)*math.pi) for i in range(4) for f in (torch.sin,torch.cos)]],-1)
class Field(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Linear(35,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU(),nn.Linear(32,2));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
 def forward(self,x,rgb,evidence,period,birth=False):
  # xyz encoding27 + RGB3 + evidence3 + visit1 + proposal type1 =35.
  return self.net(torch.cat([encode(x),rgb,evidence,x.new_full((len(x),1),period/4),x.new_full((len(x),1),float(birth))],-1))
class Update(nn.Module):
 def __init__(self,old,seeds,eligible,evidence,center,extent,period,refine_ids,appearance_mode="global"):
  super().__init__();self.old=old;self.seeds=seeds;self.ids=eligible;self.evidence=evidence;self.center=center;self.extent=extent;self.period=period;self.field=Field().cuda()
  self.refine_ids=refine_ids
  confidence=(1-evidence[refine_ids,0])*evidence[refine_ids,2]
  self.register_buffer('refinement_weights',(1+9*(1-confidence.clamp(0,1))).detach())
  m=len(refine_ids);self.old_xyz_delta=nn.Parameter(torch.zeros(m,3,device='cuda'));self.old_scale_delta=nn.Parameter(torch.zeros(m,3,device='cuda'));self.appearance_mode=appearance_mode
  if appearance_mode=='local':self.old_rgb_delta=nn.Parameter(torch.zeros(m,1,3,device='cuda'))
  else:self.register_buffer('old_rgb_delta',torch.zeros(m,1,3,device='cuda'))
  self.log_gain=nn.Parameter(torch.zeros(3,device='cuda'));self.bias=nn.Parameter(torch.zeros(3,device='cuda'))
  if appearance_mode=='local' and m>1:
   from scipy.spatial import cKDTree
   import numpy as np
   pts=old['xyz'][refine_ids].detach().cpu().numpy();near=cKDTree(pts).query(pts,k=2,workers=4)[1][:,1];self.register_buffer('neighbor',torch.as_tensor(near,device='cuda',dtype=torch.long))
  else:self.register_buffer('neighbor',torch.arange(m,device='cuda'))
  n=len(seeds['xyz']);self.xyz_delta=nn.Parameter(torch.zeros(n,3,device='cuda'));self.scale_delta=nn.Parameter(torch.zeros(n,3,device='cuda'));self.rgb_delta=nn.Parameter(torch.zeros(n,1,3,device='cuda'));self.rot_delta=nn.Parameter(torch.zeros(n,4,device='cuda'));self.birth_logits=nn.Parameter(torch.full((n,1),-2.1972246,device='cuda'))
 def values(self):
  old=self.old;ids=self.ids;rgb=(old['features_dc'][ids,0]*.28209479177387814+.5).clamp(0,1)
  raw=self.field((old['xyz'][ids]-self.center)/self.extent,rgb,self.evidence[ids],self.period)[:,0:1]
  gate=1-self.evidence[ids,0:1]*(1-torch.sigmoid(raw+math.log(19.)));alpha=old['opacity'].clone();alpha=alpha.index_copy(0,ids,old['opacity'][ids]*gate)
  s=self.seeds;birth={k:v for k,v in s.items()};xyz=s['xyz']+2*s['scaling'].exp()*self.xyz_delta.tanh();birth['xyz']=xyz
  rgb=(s['features_dc'][:,0]*.28209479177387814+.5).clamp(0,1)
  braw=self.field((xyz-self.center)/self.extent,rgb,xyz.new_zeros((len(xyz),3)),self.period,birth=True)[:,1:2]
  birth['opacity']=(self.birth_logits+braw).sigmoid();birth['features_dc']=s['features_dc']+self.rgb_delta
  birth['scaling']=s['scaling']+self.scale_delta.clamp(-.693147,.693147);birth['rotation']=torch.nn.functional.normalize(s['rotation']+self.rot_delta,dim=-1)
  return alpha,birth,gate
 def refined(self):
  ids=self.refine_ids;old=self.old
  return dict(xyz=old['xyz'][ids]+old['scaling'][ids].exp()*self.old_xyz_delta,
              scaling=old['scaling'][ids]+self.old_scale_delta,
              features_dc=old['features_dc'][ids]+self.old_rgb_delta)
 def refinement_penalty(self):
  return sum((v.square()*self.refinement_weights[:,None]).sum()/max(1,v.numel()) for v in (self.old_xyz_delta,self.old_scale_delta))
 def forward(self):
  alpha,birth,gate=self.values();old=dict(self.old)
  for k,v in self.refined().items():old[k]=old[k].index_copy(0,self.refine_ids,v)
  x={k:torch.cat((alpha if k=='opacity' else old[k],birth[k]),0) for k in old}
  return view(x),gate,birth

 def correct(self,rgb):
  return self.log_gain.exp()[:,None,None]*rgb+self.bias[:,None,None]
 def appearance_penalty(self):
  global_reg=self.log_gain.square().mean()+self.bias.square().mean()
  local=self.old_rgb_delta*.28209479177387814
  local_reg=(local.square()*self.refinement_weights[:,None,None]).sum()/max(1,local.numel())
  smooth=(local-local[self.neighbor]).square().sum()/max(1,local.numel())
  return global_reg,local_reg,smooth
