"""Two isolated full histories; technical gates and deduplicated stages."""
import os,sys,time,fcntl,subprocess,threading,concurrent.futures,shutil
from pathlib import Path
import numpy as np
from real_multistate_common import *
from gpu_schedule import acquire_gpu
from runtime_policy import require_space as base_require_space,wait_stage
def require_space():
 base_require_space()
 if shutil.disk_usage(R).free<104*1024**3:raise RuntimeError("Real dataset104GiB storage guard")

S={};M=threading.RLock()
def save():
 with M:S['heartbeat']=time.time();dump(A/'pipeline.json',S)
def main():
 lock=(A/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p=freeze()
 for f,h in p['inputs'].items():assert sha(f)==h,f
 prev=read(A/'pipeline.json') if (A/'pipeline.json').exists() else {};S.update(pid=os.getpid(),status='running',started=time.time(),jobs=prev.get('jobs',{}));save()
 def job(scene):
  j=p['jobs'][scene];base=O/scene;base.mkdir(parents=True,exist_ok=True);lease=None
  if S['jobs'].get(scene,{}).get('status')=='complete' and (base/'full'/f'eval-{max(j["periods"])}-after-{max(j["periods"])}.done.json').exists():return
  while lease is None:
   require_space()
   for gpu in p['gpus']:
    lease=acquire_gpu(gpu,lambda g:int(subprocess.check_output(['nvidia-smi','-i',str(g),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))<500)
    if lease:break
   if lease is None:time.sleep(10)
  try:
   env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONHASHSEED':'0','OMP_NUM_THREADS':'4','PYTHONUNBUFFERED':'1','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True','MPLBACKEND':'Agg','HF_HUB_OFFLINE':os.environ.get('HF_HUB_OFFLINE','0')}
   def stage(run,tag,args):
    freeze();marker=run/f'{tag}.done.json';fail=run/f'{tag}.failed.json'
    if marker.exists():return
    if fail.exists():raise RuntimeError(f'Unchanged retry prohibited:{fail}')
    require_space();log=run/f'{tag}-{time.time_ns()}.log';cmd=[sys.executable,*map(str,args)];start=time.time()
    with log.open('w') as f:
     proc=subprocess.Popen(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
     with M:S['jobs'][scene]=dict(status='running',stage=f'{run.name}/{tag}',pid=proc.pid,gpu=gpu,log=str(log),started=start);save()
     rc=wait_stage(proc,f)
    if rc:dump(fail,dict(command=cmd,code=rc,log=str(log)));raise RuntimeError(f'{scene}/{tag} exited{rc}:{log}')
    dump(marker,dict(command=cmd,seconds=time.time()-start,log=str(log),protocol_sha=sha(A/'protocol.json')))
   stage(base,'initial-smoke',[R/'rescene/pipeline/real_multistate_initial_entry.py',scene,'--smoke'])
   stage(base,'initial-train',[R/'rescene/pipeline/real_multistate_initial_entry.py',scene])
   initial=Path(j['initial_ply']);assert sha(initial)==read(base/'initial_model/isolation.json')['initial_ply_sha']
   q=read(Path(j['source'])/'wat_protocol.json')
   for smoke in ([True] if os.environ.get('RESCENE_SMOKE_ONLY')=='1' else [True,False]):
    run=base/('smoke' if smoke else 'full');run.mkdir(exist_ok=True);cfg=read(p['base_config']);cfg.update(source=j['source'],protocol_sha256=sha(Path(j['source'])/'wat_protocol.json'),initial_ply=str(initial),initial_sha256=sha(initial),training_views=j['counts'],periods=j['periods'],common_gt=j['common_gt'],depth_root=str(run),iterations=3 if smoke else 5000,split_steps=[2] if smoke else list(range(1000,4501,500)),checkpoints=[],label='Real capture ReScene current24',limitations=['Supplied all-view SfM; offline TRAIN alignment; real initial SfM points; estimated update depth','Heuristic evidence','Versioned memory grows','60k new-birth ceiling/update','Single seed; no heldout tuning'])
    if (run/'config.json').exists():assert read(run/'config.json')==cfg
    else:dump(run/'config.json',cfg)
    if not smoke and not (run/'initial.pt').exists():(run/'initial.pt').symlink_to(base/'smoke/initial.pt')
    if not (run/'initial.pt').exists():stage(run,'init',[A/'implementation/worker.py','init','--run',run])
    stage(run,'eval-initial',[A/'implementation/worker.py','evaluate','--run',run,'--period',0])
    for t in j['periods'][1:3] if smoke else j['periods'][1:]:
     depth=run/f'depth_{t:04d}.npz';sm=base/'smoke'/depth.name
     if not smoke and sm.exists() and not depth.exists():
      depth.symlink_to(sm);depth.with_suffix('.json').symlink_to(sm.with_suffix('.json'));dump(run/f'depth-{t}.done.json',dict(reused=str(sm),sha=sha(sm)))
     if not depth.exists():stage(run,f'depth-{t}',[R/'rescene/pipeline/real_multistate_depth.py','--run',run,'--period',t])
     meta=read(depth.with_suffix('.json'));names=sorted(n for n in q['update_names'] if n.startswith(f'IMG_{t:04d}/'));assert meta['names']==names and meta['input_sha256']=={n:sha(Path(j['source'])/'images'/n) for n in names}
     assert set(meta['accessed_real_rgb'])=={str((Path(j['source'])/'images'/n).resolve()) for n in names}
     with np.load(depth) as d:assert d['names'].tolist()==names and np.isfinite(d['depth']).all()
     stage(run,f'prepare-{t}',[R/'rescene/pipeline/prepare_simple60.py',run,t]);assert read(run/f'prepare_audit_{t:04d}.json')['cache_sha']==sha(run/f'prepared_{t:04d}.pt')
     stage(run,f'train-{t}',[A/'implementation/worker.py','train','--run',run,'--period',t]);tr=read(run/f'train_{t:04d}.json');assert tr['steps']==cfg['iterations'] and tr['names']==names and tr['births_kept']<=60000
     assert [e['step'] for e in tr['topology']]==cfg['split_steps']
     if smoke:assert any(e['split']>0 for e in tr['topology']),'Smoke did not exercise splitting'
     for old in range(t+1):stage(run,f'eval-{old}-after-{t}',[A/'implementation/worker.py','evaluate','--run',run,'--period',old])
     cache=run/f'depth-cache-{t:04d}'
     if cache.exists():
      assert not cache.is_symlink() and cache.resolve().is_relative_to(run.resolve());shutil.rmtree(cache);dump(run/f'cache-cleanup-{t}.json',dict(path=str(cache),depth_sha=sha(depth),validated_history=True))
    if smoke:dump(base/'smoke-validation.json',dict(periods=j['periods'][1:3],positive_splits=True,history_pixel_exact=True,steps=3,not_benchmark=True))
   with M:S['jobs'][scene]=dict(status='complete');save()
  except BaseException as e:
   with M:S['jobs'][scene]=dict(status='failed',error=repr(e));save()
   raise
  finally:lease.close()
 errors=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=min(2,len(p['gpus']))) as ex:
  fs={ex.submit(job,s):s for s in p['scenes']}
  for f in concurrent.futures.as_completed(fs):
   try:f.result()
   except Exception as e:errors.append(repr(e))
 try:
  if os.environ.get('RESCENE_SMOKE_ONLY')!='1':subprocess.run([sys.executable,str(R/'rescene/pipeline/report_real_multistate.py')],cwd=R,check=True)
 except Exception as e:errors.append(repr(e))
 S.update(status='partial-complete' if errors else 'complete',errors=errors,finished=time.time());save()
if __name__=='__main__':main()
