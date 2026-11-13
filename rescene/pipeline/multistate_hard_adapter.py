"""COLMAP adapter: retain only t0 training-point support, never later GT depth."""
import importlib.util
import numpy as np
from PIL import Image
from multistate_hard_common import *
def prepare_raw():
 spec=importlib.util.spec_from_file_location('rw',R/'external/LTGS/submodules/Hierarchical-Localization/hloc/utils/read_write_model.py');rw=importlib.util.module_from_spec(spec);spec.loader.exec_module(rw)
 for s in SCENES:
  src=R/'data/multistate_synthetic/dataset'/s;dst=D/'raw-adapter'/s
  cams0=None;images={};points={};split=[];tests=[];audit={}
  for t in range(STATE_COUNTS[s]):
   state=src/f't{t:03d}';cams=rw.read_cameras_binary(state/'sparse/0/cameras.bin');ims=rw.read_images_binary(state/'sparse/0/images.bin');pts=rw.read_points3D_binary(state/'sparse/0/points3D.bin') if t==0 else {};test=set((state/'sparse/0/test.txt').read_text().splitlines());assert len(test)==8 and len(ims)==60
   assert test=={f'view_{i:03d}.png' for i in range(0,60,8)}
   if cams0 is None:cams0=cams
   assert set(cams)==set(cams0)
   for k,c in cams.items():
    o=cams0[k];assert c.model==o.model=='PINHOLE' and c.width==c.height==800 and np.array_equal(c.params,o.params)
   if t==0:
    trainids={i for i,im in ims.items() if im.name not in test}
    for pid,p in pts.items():
     keep=np.isin(p.image_ids,list(trainids))
     if keep.sum()>=2:points[pid]=p._replace(image_ids=p.image_ids[keep],point2D_idxs=p.point2D_idxs[keep])
    assert points
   for i,im in ims.items():
    name=f't{t:03d}/{im.name}';newid=t*1000+i;ids=im.point3D_ids.copy()
    if t or im.name in test:ids[:]=-1
    else:ids[~np.isin(ids,list(points))]=-1
    images[newid]=im._replace(id=newid,name=name,point3D_ids=ids)
    inp=state/'images'/im.name;out=dst/'images'/name;out.parent.mkdir(parents=True,exist_ok=True)
    if not out.exists():out.symlink_to(inp.resolve())
    with Image.open(inp) as rgb:assert rgb.size==(800,800) and rgb.mode=='RGB'
    split.append(f'{name} {t}')
    if im.name in test:tests.append(name)
   audit[str(t)]={'cameras_sha':sha(state/'sparse/0/cameras.bin'),'images_sha':sha(state/'sparse/0/images.bin'),'test_sha':sha(state/'sparse/0/test.txt')}
  (dst/'sparse/0').mkdir(parents=True,exist_ok=True);rw.write_model(cams0,images,points,dst/'sparse/0',ext='.bin')
  (dst/'sparse/split.txt').write_text('\n'.join(split)+'\n');(dst/'sparse/0/test.txt').write_text('\n'.join(tests)+'\n')
  dump(dst/'adapter-audit.json',{'source':str(src),'states':audit,'initial_points':len(points),'initial_points_sha':sha(src/'t000/sparse/0/points3D.bin'),'initialization':'Supplied synthetic training-depth oracle PCD; >=2 t0 train tracks only','later_gt_depth_used':False,'unique_images':len(images),'heldout':len(tests)})
