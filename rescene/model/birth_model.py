"""Isolated soft-retirement and adaptive-birth ablations. Old parameters unchanged."""
import torch,math
from torch import nn
from base_model import Update as Base
class Update(Base):
 arm='control'
 def values(self):
  alpha,birth,gate=super().values()
  if self.arm=='soft':
   ids=self.ids;rgb=(self.old['features_dc'][ids,0]*.28209479177387814+.5).clamp(0,1)
   raw=self.field((self.old['xyz'][ids]-self.center)/self.extent,rgb,self.evidence[ids],self.period)[:,0:1]
   start=1-.05*self.evidence[ids,0:1];gate=torch.sigmoid(raw+torch.logit(start.clamp(1e-7,1-1e-7)))
   alpha=self.old['opacity'].index_copy(0,ids,self.old['opacity'][ids]*gate)
  return alpha,birth,gate
 def preservation(self,gate):
  return ((1-self.evidence[self.ids,0:1])*(1-gate).square()).mean() if self.arm=='soft' else gate.new_zeros(())
 @torch.no_grad()
 def split_births(self,opt,importance,support,cap=30000):
  _,birth,_=self.values();n=len(birth['xyz']);cand=((birth['opacity'][:,0]>.05)&(importance>0)).nonzero().flatten();count=min(cap-n,max(1,n//10),len(cand))
  if count<=0:return support,dict(split=0,count=n)
  selected=cand[torch.argsort(importance[cand],descending=True)[:count]];keep=torch.ones(n,device=selected.device,dtype=torch.bool);keep[selected]=False;kept=keep.nonzero().flatten();parents=selected.repeat_interleave(2);newseeds={k:torch.cat((self.seeds[k][kept],birth[k][parents].clone()),0) for k in self.seeds}
  scales=birth['scaling'][parents].exp();axis=scales.argmax(-1);q=torch.nn.functional.normalize(birth['rotation'][parents],dim=-1);vec=torch.nn.functional.one_hot(axis,3).to(scales)*scales*.5
  # Rotate principal-axis displacement by quaternion (w,x,y,z).
  uv=torch.cross(q[:,1:],vec,dim=-1);offset=vec+2*(q[:,:1]*uv+torch.cross(q[:,1:],uv,dim=-1));sign=torch.tensor([-1.,1.],device=vec.device).repeat(count)[:,None]
  newseeds['xyz'][len(kept):]+=sign*offset;newseeds['scaling'][len(kept):]-=math.log(1.6)
  alpha=(1-torch.sqrt(1-birth['opacity'][parents].clamp(max=1-1e-6))).clamp(1e-6,1-1e-6)
  xyz=newseeds['xyz'][len(kept):];rgb=(newseeds['features_dc'][len(kept):,0]*.28209479177387814+.5).clamp(0,1)
  braw=self.field((xyz-self.center)/self.extent,rgb,xyz.new_zeros(len(xyz),3),self.period,birth=True)[:,1:2]
  for name in ['xyz_delta','scale_delta','rgb_delta','rot_delta','birth_logits']:
   old=getattr(self,name);extra=torch.zeros((2*count,*old.shape[1:]),device=old.device,dtype=old.dtype)
   if name=='birth_logits':extra=torch.logit(alpha)-braw
   new=nn.Parameter(torch.cat((old.detach()[kept],extra),0));setattr(self,name,new)
   for group in opt.param_groups:group['params']=[new if p is old else p for p in group['params']]
   st=opt.state.pop(old,{})
   opt.state[new]={k:(torch.cat((v[kept],torch.zeros((2*count,*v.shape[1:]),device=v.device,dtype=v.dtype)),0) if torch.is_tensor(v) and v.shape==old.shape else v) for k,v in st.items()}
  self.seeds=newseeds
  return torch.cat((support[kept],support[parents]),0),dict(split=count,before=n,after=n+count,rule='top10pct accumulated birth XYZ-parameter gradient, opacity>.05; two principal-axis children, scale/1.6, transmittance-matched opacity; kept Adam moments retained, child moments zero')
