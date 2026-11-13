"""Preserve official test identities; select current cameras without test feedback."""
import importlib.util,sys,shutil,subprocess
import numpy as np
from PIL import Image
from multistate_hard_common import *
def main():
 A.mkdir(parents=True,exist_ok=True);O.mkdir(parents=True,exist_ok=True)
 from multistate_hard_adapter import prepare_raw
 prepare_raw()
 spec=importlib.util.spec_from_file_location('rw',R/'external/LTGS/submodules/Hierarchical-Localization/hloc/utils/read_write_model.py');rw=importlib.util.module_from_spec(spec);spec.loader.exec_module(rw)
 jobs={};inputs={str(R/'data/multistate_hard/manifest.json'):sha(R/'data/multistate_hard/manifest.json'),str(R/'data/multistate_hard/SHA256SUMS'):sha(R/'data/multistate_hard/SHA256SUMS')}
 for s in SCENES:
  src=D/'raw-adapter'/s;cams,ims,pts=rw.read_model(src/'sparse/0',ext='.bin');period={n:int(t) for n,t in (ln.rsplit(maxsplit=1) for ln in (src/'sparse/split.txt').read_text().splitlines() if ln.strip())};tests=set((src/'sparse/0/test.txt').read_text().splitlines());assert tests<=set(period)
  assert set(im.name for im in ims.values())==set(period)
  initial_ids={i for i,im in ims.items() if period[im.name]==0 and im.name not in tests}
  pp={}
  for i,pt in pts.items():
   keep=np.isin(pt.image_ids,list(initial_ids))
   if keep.sum()>=2:pp[i]=pt._replace(image_ids=pt.image_ids[keep],point2D_idxs=pt.point2D_idxs[keep])
  assert pp,(s,'No initial-only track support')
  renamed={};mapping={};oldby={im.name:im for im in ims.values()}
  for k,(i,im) in enumerate(sorted(ims.items(),key=lambda x:(period[x[1].name],x[1].name))):
   name=f'IMG_{period[im.name]:04d}/frame_{k:06d}{Path(im.name).suffix.lower()}';mapping[im.name]=name;ids=im.point3D_ids.copy();ids[~np.isin(ids,list(pp))]=-1
   if i not in initial_ids:ids[:]=-1
   renamed[i]=im._replace(name=name,point3D_ids=ids)
  orders={}
  for t in sorted(set(period.values())-{0}):
   pool=sorted(n for n in period if period[n]==t and n not in tests);centers={n:-oldby[n].qvec2rotmat().T@oldby[n].tvec for n in pool};pick=[pool[0]]
   while len(pick)<min(24,len(pool)):pick.append(max((n for n in pool if n not in pick),key=lambda n:min(np.linalg.norm(centers[n]-centers[b]) for b in pick)))
   orders[str(t)]=pick
  initial=sorted(mapping[ims[i].name] for i in initial_ids);updates=sorted(mapping[n] for ns in orders.values() for n in ns);tn=sorted(mapping[n] for n in tests);allowed=set(initial+updates+tn)
  q={'scene':s,'source':str(src),'periods':sorted(set(period.values())),'initial_names':initial,'update_names':updates,'test_names':tn,'name_map':mapping,'selected_original_names':orders,'selection':'Lexical-first then greedy farthest camera centers, current training views only','initial_points':len(pp),'source_points':len(pts),'initialization':'Oracle synthetic training-depth points with >=2 initial training tracks','image_sha256':{mapping[n]:sha(src/'images'/n) for n in mapping if mapping[n] in allowed},'source_split_sha':sha(src/'sparse/split.txt'),'source_test_sha':sha(src/'sparse/0/test.txt')}
  dest=D/s
  for kind,names in [('current',allowed),('initial',set(initial))]:
   ds=dest/kind;(ds/'images').mkdir(parents=True,exist_ok=True);(ds/'sparse/0').mkdir(parents=True,exist_ok=True)
   for orig,name in mapping.items():
    if name not in names:continue
    to=ds/'images'/name;to.parent.mkdir(parents=True,exist_ok=True)
    if not to.exists():to.symlink_to((src/'images'/orig).resolve())
   imsout={i:im for i,im in renamed.items() if im.name in names};rw.write_model(cams,imsout,pp,ds/'sparse/0',ext='.bin')
   (ds/'sparse/0/test.txt').write_text('\n'.join(tn if kind=='current' else [])+'\n');(ds/'images/changes.txt').write_text('\n'.join(updates if kind=='current' else [])+'\n');dump(ds/'wat_protocol.json',q if kind=='current' else {**q,'test_names':[],'update_names':[]})
   for n in ['cameras.bin','images.bin','points3D.bin']:inputs[str(ds/'sparse/0'/n)]=sha(ds/'sparse/0'/n)
   inputs[str(ds/'wat_protocol.json')]=sha(ds/'wat_protocol.json')
  # Independent common GT cache from unchanged held-out RGB, same renderer dimensions.
  gt=O/s/'common-gt';by={im.name:im for im in renamed.values()}
  gtinfo={}
  for t in q['periods']:
   for k,n in enumerate(sorted(n for n in tn if n.startswith(f'IMG_{t:04d}/'))):
    cam=cams[by[n].camera_id];assert cam.model=='PINHOLE';assert abs(cam.params[2]-cam.width/2)<1e-3 and abs(cam.params[3]-cam.height/2)<1e-3
    sc=max(1,cam.width/1600);size=(int(cam.width/sc),int(cam.height/sc));out=gt/str(t)/f'{k:05d}.png';out.parent.mkdir(parents=True,exist_ok=True)
    with Image.open(dest/'current/images'/n) as im:im.convert('RGB').resize(size,Image.Resampling.BICUBIC).save(out)
    gtinfo[n]={'path':str(out),'sha':sha(out),'size':size}
  dump(O/s/'gt-manifest.json',gtinfo);inputs[str(O/s/'gt-manifest.json')]=sha(O/s/'gt-manifest.json')
  jobs[s]={'source':str(dest/'current'),'initial_source':str(dest/'initial'),'periods':q['periods'],'counts':{t:len(ns) for t,ns in orders.items()},'initial_train':len(initial),'test_views':len(tn),'initial_points':len(pp),'common_gt':str(gt),'initial_ply':str(O/s/'initial_model/point_cloud/iteration_30000/point_cloud.ply')}
  print(s,jobs[s],flush=True)
 # Copy worker and remove only WAT-specific evaluation path assertions. Training body unchanged.
 orig=(R/'rescene/model/worker.py').read_text();code='import sys\nfrom pathlib import Path\nsys.path.insert(0,"'+str(R/'rescene/model')+'")\n'+orig
 code=code.replace("ROOT/'outputs/wat-20261007'/src.name/'test'/f'ours_40000_{a.period}'/'gt'/f'{i:05d}.png'","Path(cfg['common_gt'])/str(a.period)/f'{i:05d}.png'")
 a=code.index('   if a.period==0:\n');b=code.index('   if dest.exists():',a);code=code[:a]+code[b:]
 impl=A/'implementation';impl.mkdir(exist_ok=True);(impl/'worker.py').write_text(code)
 sources={str(f):sha(f) for f in list((R/'rescene/pipeline').glob('*.py'))+list((R/'rescene/model').glob('*.py'))+[R/'rescene/rendering/backend.py',R/'external/LTGS/gaussian-splatting/train.py',R/'external/LTGS/gaussian-splatting/scene/dataset_readers.py',R/'rescene/configs/multistate_hard.json',impl/'worker.py']}
 protocol={'objective':'photometric_plus_preservation','tag':TAG,'scenes':SCENES,'jobs':jobs,'sources':sources,'inputs':inputs,'gpus':[int(v) for v in __import__('os').environ.get('GSVER_GPUS','0').split(',')],'reserve_gib':100,'seed':0,'initial_nominal_steps':30000,'update_steps':5000,'base_config':str(R/'rescene/configs/multistate_hard.json'),'scope':'Four complex-motion synthetic sequences, 42 states and38updates','camera_protocol':'Exact synthetic camera poses; no causal localization. Supplied held-out identities; RGB BICUBIC maxwidth1600 with per-camera aspect. MASt3R crop-aware K and depth remap to full-frame normalized grid.','initialization':'Standard3DGS period0 train-only fit, oracle synthetic train-depth points with filtered tracks; upstream nominal30k/29999optimizersteps and final-save behavior retained from WATinitial baseline','architecture':'ReScene photometric-plus-preservation objective:30kseeds,60kceiling,8splitrounds,5000steps/lr.001 currentRGB only, versioned history','evaluation':'All official heldouts, same common metrics; rerender history aftereach update; frozen initial-only reference','repository_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()}
 if (A/'protocol.json').exists():assert read(A/'protocol.json')==protocol,'Frozen protocol exists; explicit migration required'
 else:dump(A/'protocol.json',protocol)
 for f,h in sources.items():
  out=A/'source-snapshot'/Path(f).relative_to(R);out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,out)
 dump(A/'integration.json',{'worker_changes':'Only common-GT path and WAT-specific initial-reference assertion removed. Uses the release photometric-plus-preservation training body without parameter regularisers.','depth_change':'Explicit crop-aware intrinsics; depth/confidence remapped to normalized full-frame grid; out-of-crop pixels invalid (confidence0), not extrapolated.'})
if __name__=='__main__':main()
