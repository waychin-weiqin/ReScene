"""Train standard initial3DGS on physically isolated period0 training RGB only."""
import sys,os,runpy,random
from pathlib import Path
from multistate_hard_common import R,O,A,read,dump,sha,freeze
import numpy as np,torch
scene=sys.argv[1];p=freeze();j=p['jobs'][scene];src=Path(j['initial_source']);q=read(src/'wat_protocol.json');allowed={str((src/'images'/n).resolve()) for n in q['initial_names']};accessed=set()
def guard(event,args):
 if event=='open' and isinstance(args[0],(str,bytes)):
  f=Path(os.fsdecode(args[0])).resolve()
  if f.suffix.lower() in ('.png','.jpg','.jpeg'):
   assert str(f) in allowed,f'Forbidden initial RGB:{f}';accessed.add(str(f))
sys.addaudithook(guard);random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.set_num_threads(4)
lt=R/'external/LTGS';sys.path[:0]=[str(lt/'gaussian-splatting'),str(lt)]
model=O/scene/'initial_model';model.mkdir(parents=True,exist_ok=True)
sys.argv=[str(lt/'gaussian-splatting/train.py'),'-s',str(src),'-m',str(model),'--eval','--single_timestep','--data_device','cpu','--disable_viewer','--iterations','30000','--test_iterations','30001','--save_iterations','30000','--checkpoint_iterations','30000']
runpy.run_path(sys.argv[0],run_name='__main__');assert accessed==allowed
out=Path(j['initial_ply']);assert out.exists();dump(model/'isolation.json',{'accessed':sorted(accessed),'initial_ply_sha':sha(out),'nominal_iterations':30000,'optimizer_steps':29999,'heldout_or_future_rgb':False,'protocol_sha':sha(A/'protocol.json')})
