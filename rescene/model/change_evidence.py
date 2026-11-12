"""Detached current-view evidence; all thresholds are declared in run config."""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import median_filter, label
from common import view, render, PIPE
from utils.loss_utils import create_window

def retirement_disagreement(old_z,current_z,tolerance):
 # Bidirectional depth discrepancy; does not establish physical disappearance.
 return (old_z-current_z).abs()>tolerance*current_z

def project(x, pose, K, h, w):
 q=(x-pose[:3,3])@pose[:3,:3];z=q[:,2]
 u=(q[:,0]/z.clamp_min(1e-6)*K[0,0]+K[0,2]).round().long()
 v=(q[:,1]/z.clamp_min(1e-6)*K[1,1]+K[1,2]).round().long()
 ok=(z>0)&(u>=0)&(u<w)&(v>=0)&(v<h)
 return z,u.clamp(0,w-1),v.clamp(0,h-1),ok

def weighted(x,w):
 return (x*w).sum()/w.sum().clamp_min(1.)

def objective(pred,target,reference,change,stable):
 # Identical Gaussian11x11 SSIM definition as common evaluator, retaining map.
 win=create_window(11,3).to(pred.device)
 x=pred[None];y=target[None]
 mx=F.conv2d(x,win,padding=5,groups=3);my=F.conv2d(y,win,padding=5,groups=3)
 vx=F.conv2d(x*x,win,padding=5,groups=3)-mx*mx
 vy=F.conv2d(y*y,win,padding=5,groups=3)-my*my
 cov=F.conv2d(x*y,win,padding=5,groups=3)-mx*my
 ss=((2*mx*my+.01**2)*(2*cov+.03**2))/((mx*mx+my*my+.01**2)*(vx+vy+.03**2))
 photo=weighted(.8*(pred-target).abs().mean(0)+.2*(1-ss.mean((0,1))),change)
 preserve=weighted((pred-reference).abs().mean(0),stable)
 return photo,preserve

@torch.no_grad()
def prepare(old,names,cams,targets,depthfile,cfg):
 d=np.load(depthfile);assert d['names'].tolist()==names
 device=old['xyz'].device
 dep=torch.as_tensor(d['depth'],device=device);conf=torch.as_tensor(d['conf'],device=device)
 poses=torch.as_tensor(d['poses'],device=device,dtype=torch.float32);Ks=torch.as_tensor(d['intrinsics'],device=device,dtype=torch.float32)
 h,w=dep.shape[-2:];preds=[];rgb=[];raw=[];stable=[];olddepth=[];refs={};offsets=[]
 for i,n in enumerate(names):
  out=render(cams[n],view(old),PIPE,torch.zeros(3,device=device),separate_sh=True)
  alpha=render(cams[n],view(old),PIPE,torch.zeros(3,device=device),separate_sh=False,override_color=torch.ones_like(old['xyz']))['render'][0]
  od=alpha/out['depth'].squeeze().clamp_min(1e-8)
  olddepth.append(F.interpolate(od[None,None],(h,w),mode='nearest')[0,0])
  refs[n]=out['render'].detach();p=F.interpolate(refs[n][None],(h,w),mode='bilinear',align_corners=False)[0]
  t=F.interpolate(targets[n][None],(h,w),mode='bilinear',align_corners=False)[0]
  diff=t-p;r=diff.abs().mean(0);low=r<=r.median()
  offset=diff[:,low].median(1).values.clamp(-.1,.1);offsets.append(offset.tolist())
  r=(diff-offset[:,None,None]).abs().mean(0)
  filtered=median_filter(r.cpu().numpy(),size=3)
  binary=filtered>cfg['residual_threshold'];cc,num=label(binary);sizes=np.bincount(cc.ravel());keep=sizes>=cfg['minimum_component'];keep[0]=False
  cleaned=torch.as_tensor(keep[cc],device=device)
  good=(conf[i]>cfg['depth_confidence'])&(dep[i]>0)
  raw.append(cleaned*good);stable.append((r<cfg['stable_threshold'])&good);rgb.append(t)
 # Require depth-consistent changed evidence in at least two current views.
 yy,xx=torch.meshgrid(torch.arange(h,device=device),torch.arange(w,device=device),indexing='ij');world=[];changes=[];support=[]
 for i in range(len(names)):
  z=dep[i];K=Ks[i];x=torch.stack(((xx-K[0,2])*z/K[0,0],(yy-K[1,2])*z/K[1,1],z),-1).reshape(-1,3)@poses[i,:3,:3].T+poses[i,:3,3];world.append(x)
  count=torch.zeros(h*w,device=device)
  for j in range(len(names)):
   zz,u,v,ok=project(x,poses[j],Ks[j],h,w)
   agree=(zz-dep[j,v,u]).abs()<=cfg['multiview_tolerance']*dep[j,v,u]
   count+=ok&agree&raw[j][v,u]
  support.append(count.reshape(h,w));changes.append(raw[i]&(support[-1]>=cfg['minimum_views']))
 evidence=torch.zeros(len(old['xyz']),3,device=device);veto=torch.zeros(len(old['xyz']),device=device,dtype=torch.bool)
 points=[];colors=[];scales=[];birthsupport=[];stats=[];masks={}
 center=old['xyz'].mean(0);extent=(old['xyz']-center).norm(dim=-1).quantile(.95).clamp_min(1e-6)
 for i,n in enumerate(names):
  # Births require supported change and depth disagreement with previous surface.
  missing=(olddepth[i]-dep[i]).abs()>cfg['multiview_tolerance']*dep[i]
  valid=changes[i]&missing&(xx%2==0)&(yy%2==0)
  x=world[i][valid.flatten()];good=torch.isfinite(x).all(1)&((x-center).norm(dim=1)<3*extent)
  points.append(x[good]);colors.append(rgb[i].permute(1,2,0)[valid][good]);scales.append((dep[i][valid]/Ks[i,0,0]*1.2)[good].clamp(extent*1e-5,extent*.01));birthsupport.append(support[i][valid][good]/len(names))
  z,u,v,ok=project(old['xyz'],poses[i],Ks[i],h,w)
  visible=ok&((z-olddepth[i][v,u]).abs()<=cfg['multiview_tolerance']*olddepth[i][v,u])&(old['opacity'][:,0]>.005)
  # Absolute discrepancy admits nearer and farther observed surfaces, including occlusion.
  contradiction=visible&changes[i][v,u]&retirement_disagreement(z,dep[i,v,u],cfg['multiview_tolerance'])
  evidence[:,0]+=contradiction;evidence[:,1]+=contradiction*support[i][v,u]/len(names)
  evidence[:,2]+=contradiction*((dep[i,v,u]-z)/dep[i,v,u].clamp_min(1e-6)).clamp(0,1)
  veto|=visible&stable[i][v,u]
  shape=targets[n].shape[-2:]
  up=lambda a:F.interpolate(a.float()[None,None],shape,mode='bilinear',align_corners=False)[0,0]
  masks[n]=(up(changes[i]),up(stable[i]&~changes[i]))
  stats.append(dict(name=n,raw_change_pixels=int(raw[i].sum()),changed_pixels=int(changes[i].sum()),stable_pixels=int(stable[i].sum()),birth_candidates=int(good.sum()),retirement_votes=int(contradiction.sum()),exposure_offset=offsets[i]))
 ids=torch.where((evidence[:,0]>=cfg['minimum_views'])&~veto)[0]
 count=evidence[:,0].clone();evidence[:,1:]/=count.clamp_min(1)[:,None];evidence[:,0]/=len(names)
 xyz=torch.cat(points);color=torch.cat(colors);scale=torch.cat(scales);bs=torch.cat(birthsupport)
 if len(xyz)>cfg['max_births']:
  ix=torch.randperm(len(xyz),device=device)[:cfg['max_births']];xyz,color,scale,bs=xyz[ix],color[ix],scale[ix],bs[ix]
 n=len(xyz);rot=xyz.new_zeros(n,4);rot[:,0]=1
 seeds=dict(xyz=xyz,features_dc=((color-.5)/.28209479177387814)[:,None,:],features_rest=xyz.new_zeros(n,15,3),scaling=scale[:,None].expand(-1,3).log().clone(),rotation=rot,opacity=xyz.new_full((n,1),.1))
 return seeds,ids,evidence,center,extent,stats,refs,masks,bs
