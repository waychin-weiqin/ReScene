"""Freeze reviewed Sim3 cameras, initial-only tracked SfM support and official splits."""
import importlib.util,numpy as np
from real_multistate_common import *
def prepare_raw():
 spec=importlib.util.spec_from_file_location('rw',R/'external/LTGS/submodules/Hierarchical-Localization/hloc/utils/read_write_model.py');rw=importlib.util.module_from_spec(spec);spec.loader.exec_module(rw)
 for s in SCENES:
  src=R/'data/multistate_real/dataset'/s;dst=D/'raw-adapter'/s
  alignpath=R/'rescene/reproduction/real-alignment'/s/'alignment.json';al=read(alignpath);review=read(alignpath.parent/'reprojection-review.json');assert len(review)==3 and all(x['pass'] for x in review.values())
  cameras={};images={};points={};split=[];tests=[];audit={};idmaps={}
  for t in range(STATE_COUNTS[s]):
   state=src/f't{t:03d}'
   expected=read(R/'rescene/reproduction/real-alignment'/s/'input-hashes.json')[f't{t:03d}']
   for rel,digest in expected.items():assert sha(state/rel)==digest,f'Dataset/calibration revision mismatch: {state/rel}'
   cams,ims,pts=rw.read_model(state/'sparse/0',ext='.bin');test=set((state/'test.txt').read_text().splitlines());train=set((state/'train.txt').read_text().splitlines());assert not test&train and test|train=={im.name for im in ims.values()}
   matrix=np.array(al['transforms_to_initial'][f't{t:03d}']);scale=np.cbrt(np.linalg.det(matrix[:3,:3]));rot=matrix[:3,:3]/scale;offset=matrix[:3,3];assert np.allclose(rot.T@rot,np.eye(3),atol=1e-8)
   for k,c in cams.items():cameras[t*1000+k]=c._replace(id=t*1000+k)
   if t==0:
    trainids={i for i,im in ims.items() if im.name in train}
    for pid,p in pts.items():
     keep=np.isin(p.image_ids,list(trainids))
     if keep.sum()>=2:points[pid]=p._replace(image_ids=p.image_ids[keep],point2D_idxs=p.point2D_idxs[keep])
   for i,im in ims.items():
    name=f't{t:03d}/{im.name}';newid=t*100000+i;ids=im.point3D_ids.copy();ids[~np.isin(ids,list(points))]=-1
    if t or im.name in test:ids[:]=-1
    oldrot=im.qvec2rotmat();newrot=oldrot@rot.T;newtrans=scale*im.tvec-newrot@offset
    # Sim3 transformed cameras preserve all source projections exactly.
    sample=np.array([[.1,.2,3],[1,2,5],[-2,1,4]])
    before=sample@oldrot.T+im.tvec;after=(scale*sample@rot.T+offset)@newrot.T+newtrans;assert np.allclose(after,scale*before,atol=1e-8)
    images[newid]=im._replace(id=newid,name=name,camera_id=t*1000+im.camera_id,qvec=rw.rotmat2qvec(newrot),tvec=newtrans,point3D_ids=ids)
    inp=state/'images'/im.name;out=dst/'images'/name;out.parent.mkdir(parents=True,exist_ok=True)
    if not out.exists():out.symlink_to(inp.resolve())
    split.append(f'{name} {t}')
    if im.name in test:tests.append(name)
   audit[str(t)]={'cameras_sha':sha(state/'sparse/0/cameras.bin'),'images_sha':sha(state/'sparse/0/images.bin'),'test_sha':sha(state/'test.txt'),'train_sha':sha(state/'train.txt'),'transform':matrix.tolist(),'projection_invariance':True}
  (dst/'sparse/0').mkdir(parents=True,exist_ok=True);rw.write_model(cameras,images,points,dst/'sparse/0',ext='.bin');(dst/'sparse/split.txt').write_text('\n'.join(split)+'\n');(dst/'sparse/0/test.txt').write_text('\n'.join(tests)+'\n')
  dump(dst/'adapter-audit.json',{'states':audit,'alignment_sha':sha(alignpath),'initial_points':len(points),'calibration':'Supplied all-view SfM; offline TRAIN-only background alignment; arbitrary scale, no GT depth','later_geometry_used_for_initialization':False})
