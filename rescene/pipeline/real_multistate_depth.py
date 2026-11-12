"""MASt3R fixed-SfM depth with explicit resize/crop accounting."""
import sys,argparse,random
from pathlib import Path
import numpy as np
from real_multistate_common import R,A,read,dump,sha

def matrices(width,height,params,out_w,out_h):
 nw,nh=[round(v*512/max(width,height)) for v in (width,height)]
 cw,ch=((2*(nw//2))//16)*16,((2*(nh//2))//16)*16
 if nw==nh:ch=int(3*cw/4)
 ox,oy=nw//2-cw/2,nh//2-ch/2
 fx,fy,cx,cy=params
 crop=np.array([[fx*nw/width,0,cx*nw/width-ox],[0,fy*nh/height,cy*nh/height-oy],[0,0,1.]])
 full=np.array([[fx*out_w/width,0,cx*out_w/width],[0,fy*out_h/height,cy*out_h/height],[0,0,1.]])
 return crop,full,(ch,cw),(nw,nh,ox,oy)

def test():
 for width,height in [(800,800),(1920,1080),(2000,1500),(2000,1498),(2000,1499)]:
  k,f,shape,(nw,nh,x,y)=matrices(width,height,[1000,999,width/2,height/2],512,384)
  for u,v in [(0,0),(width/2,height/2),(width-1,height-1)]:
   a=np.linalg.inv(k)@np.array([u*nw/width-x,v*nh/height-y,1]);b=np.linalg.inv(f)@np.array([u*512/width,v*384/height,1]);assert np.allclose(a,b,atol=1e-12)
 print('PASS: crop/full rays agree for 16:9 and variable-height 4:3 views.')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--test',action='store_true');ap.add_argument('--run',type=Path);ap.add_argument('--period',type=int);a=ap.parse_args()
 if a.test:test();return
 sys.path.insert(0,str(R/'rescene/model'));from common import WAT,guard
 import torch,cv2
 from PIL import Image
 cfg=read(a.run/'config.json');src=Path(cfg['source']);data=WAT(src,read(src/'wat_protocol.json'));names=data.names(a.period,'train');assert len(names)==cfg['training_views'][str(a.period)];seen=guard(src,names)
 sys.path[:0]=[str(R/'external/LTGS'),str(R/'external/LTGS/submodules/mast3r')]
 from mast3r.model import AsymmetricMASt3R
 from mast3r_depth_support import get_reconstructed_scene
 from dust3r.utils.device import to_numpy
 from dust3r.utils.image import load_images
 torch.set_num_threads(4);torch.manual_seed(0);random.seed(0);np.random.seed(0)
 loaded=load_images([str(src/'images'/n) for n in names],size=512,verbose=False)
 first=data.cameras[data.images[names[0]].camera_id];ow=512;oh=round(first.height*ow/first.width);poses=[];ks=[];fullks=[];shapes=[]
 for n,im in zip(names,loaded):
  cam=data.cameras[data.images[n].camera_id]
  with Image.open(src/'images'/n) as pil:assert pil.size==(cam.width,cam.height) and pil.getexif().get(274,1)==1
  k,f,shape,_=matrices(cam.width,cam.height,cam.params,ow,oh);assert tuple(im['true_shape'][0])==shape,(n,im['true_shape'],shape)
  pose=np.eye(4);pose[:3,:3]=data.images[n].qvec2rotmat();pose[:3,3]=data.images[n].tvec;poses.append(np.linalg.inv(pose));ks.append(k);fullks.append(f);shapes.append(shape)
 model=AsymmetricMASt3R.from_pretrained('naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric').cuda();cache=a.run/f'depth-cache-{a.period:04d}'
 recon=get_reconstructed_scene(str(cache),model,[str(src/'images'/n) for n in names],cam2w=torch.tensor(np.array(poses),device='cuda',dtype=torch.float32),K=torch.tensor(np.array(ks),device='cuda',dtype=torch.float32),optim_level='refine',matching_conf_thr=.9,silent=False)
 _,depth,conf=to_numpy(recon.sparse_ga.get_dense_pts3d(clean_depth=True));ds=[];cs=[];coverage=[]
 yy,xx=np.mgrid[:oh,:ow]
 for k,f,shape,z,c in zip(ks,fullks,shapes,depth,conf):
  z=np.asarray(z).reshape(shape).astype(np.float32);c=np.asarray(c).reshape(shape).astype(np.float32)
  mx=((xx-f[0,2])/f[0,0]*k[0,0]+k[0,2]).astype(np.float32);my=((yy-f[1,2])/f[1,1]*k[1,1]+k[1,2]).astype(np.float32)
  valid=(mx>=0)&(mx<=shape[1]-1)&(my>=0)&(my<=shape[0]-1)
  d=cv2.remap(z,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0);c=cv2.remap(c,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
  d[~valid]=0;c[~valid]=0;ds.append(d);cs.append(c);coverage.append(float(valid.mean()))
 ds=np.stack(ds);cs=np.stack(cs);assert np.isfinite(ds).all() and np.isfinite(cs).all() and (ds>0).any();assert seen=={str((src/'images'/n).resolve()) for n in names}
 np.savez_compressed(a.run/f'depth_{a.period:04d}.npz',depth=ds,conf=cs,poses=np.array(poses),intrinsics=np.array(fullks),names=np.array(names))
 dump(a.run/f'depth_{a.period:04d}.json',{'names':names,'input_sha256':{n:sha(src/'images'/n) for n in names},'accessed_real_rgb':sorted(seen),'crop_K':np.array(ks).tolist(),'full_K':np.array(fullks).tolist(),'crop_shapes':shapes,'valid_fraction':coverage,'cache':str(cache),'mapping':'Crop-aware MASt3R input; bilinear remap to full-frame normalized grid; outside-crop confidence=0; supplied fixed global SfM'})
if __name__=='__main__':main()
