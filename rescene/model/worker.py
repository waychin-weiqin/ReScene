import argparse,json,random,time
from pathlib import Path
from common import ROOT,WAT,GaussianModel,draw,state,view,save,cpu,dump,sha,guard
import torch,numpy as np
import torch.nn.functional as F
from birth_model import Update
from utils.loss_utils import ssim

from change_evidence import prepare, objective
from soft_eligibility import build
from refinement import select

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['init','train','evaluate']);p.add_argument('--run',type=Path,required=True);p.add_argument('--period',type=int,default=0);a=p.parse_args();cfg=json.loads((a.run/'config.json').read_text())
 torch.set_num_threads(4);random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
 src=Path(cfg['source']);protocol=json.loads((src/'wat_protocol.json').read_text());assert sha(src/'wat_protocol.json')==cfg['protocol_sha256'];data=WAT(src,protocol)
 if a.action=='init':
  assert sha(cfg['initial_ply'])==cfg['initial_sha256'];g=GaussianModel(3);g.load_ply(cfg['initial_ply']);x={k:getattr(g,'_'+k).detach() for k in ('xyz','features_dc','features_rest','scaling','rotation','opacity')};x['opacity']=x['opacity'].sigmoid();save(a.run/'initial.pt',cpu(x));dump(a.run/'initial.json',dict(gaussians=len(x['xyz']),sha256=sha(a.run/'initial.pt')));return
 if a.action=='train':
  assert a.period>0;names=data.names(a.period,'train');assert len(names)==cfg['training_views'][str(a.period)];accessed=guard(src,names);old=state(a.run,a.period-1)
  history={str(p):sha(p) for p in [a.run/'initial.pt',*[a.run/f'period_{j:04d}.pt' for j in range(1,a.period)]]}
  cams={n:data.camera(n) for n in names};targets={n:data.rgb(n,cams[n]) for n in names};torch.cuda.reset_peak_memory_stats();start=time.monotonic()
  cache=torch.load(a.run/f'prepared_{a.period:04d}.pt',map_location='cuda',weights_only=True)
  seeds,ids,evidence,center,extent,stats,refs,masks,birth_support=[cache[k] for k in ['seeds','ids','evidence','center','extent','stats','refs','masks','birth_support']]
  refine_ids,refine_masks,refine_stats=[cache[k] for k in ['refine_ids','refine_masks','refine_stats']]
  model=Update(old,seeds,ids,evidence,center,extent,a.period,refine_ids,cfg['appearance_mode']).cuda();model.arm=cfg['arm'];opt=torch.optim.Adam(model.parameters(),lr=cfg['lr']);steps=cfg['iterations'];logs=[];topology=[];importance=torch.zeros(len(seeds['xyz']),device='cuda')
  dump(a.run/f'proposals_{a.period:04d}.json',dict(births=len(seeds['xyz']),old_eligible=len(ids),old_count=len(old['xyz']),views=stats))
  for step in range(1,steps+1):
   n=names[random.randrange(len(names))];opt.zero_grad(set_to_none=True);v,gate,birth=model();pred=model.correct(draw(cams[n],v));change,stable=masks[n];rm=refine_masks[n]
   photo,preserve=objective(pred,targets[n],model.correct(refs[n]),change,stable*(1-rm))
   refine_photo,_=objective(pred,targets[n],refs[n],rm,torch.zeros_like(stable))
   stable_photo,_=objective(pred,targets[n],model.correct(refs[n]),stable,torch.zeros_like(stable))
   global_reg,local_reg,smooth_reg=model.appearance_penalty()
   refine_penalty=model.refinement_penalty()
   retire=((1-gate[:,0])*(2-evidence[ids,0])).sum()/max(1,len(ids));add=(birth['opacity'][:,0]*(2-birth_support)).sum()/max(1,len(birth_support));geometry=(model.xyz_delta.square().sum()+model.scale_delta.square().sum())/max(1,model.xyz_delta.numel())
   loss=photo+cfg['refine_photo_weight']*refine_photo+cfg['refine_penalty_weight']*refine_penalty+cfg['static_weight']*preserve+cfg['retirement_weight']*retire+cfg['birth_weight']*add+cfg['geometry_weight']*geometry
   loss=loss+cfg['appearance_photo_weight']*stable_photo+cfg['global_appearance_weight']*global_reg+cfg['local_appearance_weight']*(local_reg+smooth_reg)
   loss=loss+cfg['soft_preservation_weight']*model.preservation(gate)
   assert torch.isfinite(loss);loss.backward();importance+=model.xyz_delta.grad.detach().norm(dim=-1);grad=torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True);opt.step()
   if cfg['arm']=='adaptive' and step in cfg['split_steps']:
    birth_support,event=model.split_births(opt,importance,birth_support,cfg['max_births']);event['step']=step;topology.append(event);seeds=model.seeds;importance=torch.zeros(len(seeds['xyz']),device='cuda');dump(a.run/'topology.json',topology)
   if step==1 or step%25==0 or step==steps:
    row=dict(stable_photo=float(stable_photo),global_reg=float(global_reg),local_reg=float(local_reg),smooth_reg=float(smooth_reg),refine_photo=float(refine_photo),refine_penalty=float(refine_penalty),step=step,camera=n,preserve=float(preserve),loss=float(loss),photo=float(photo),retirement=float(retire),birth_opacity=float(add),gradient=float(grad));logs.append(row);print(json.dumps(row),flush=True)
   if step in cfg['checkpoints'] and step<steps:
    save(a.run/f'checkpoint_p{a.period:04d}_s{step:05d}.pt',dict(period=a.period,step=step,model=cpu(model.state_dict()),refine_ids=refine_ids.cpu(),seeds=cpu(seeds),eligible=ids.cpu(),evidence=evidence[ids].cpu(),center=center.cpu(),extent=extent.cpu(),note='diagnostic checkpoint; full optimizer state not stored'))
  with torch.no_grad():
   alpha,birth,gate=model.values();soft_alpha=alpha[ids].clone();retired=ids[alpha[ids,0]<.005];alpha[retired]=0;keep=birth['opacity'][:,0]>=.01;birth={k:v[keep] for k,v in birth.items()}
   untouched=torch.ones(len(alpha),device='cuda',dtype=torch.bool);untouched[ids]=False
   assert torch.equal(alpha[untouched],old['opacity'][untouched]),'Unobserved Gaussian opacity mutated'
   assert all(torch.isfinite(v).all() for v in birth.values()) and torch.isfinite(alpha).all()
   save(a.run/f'period_{a.period:04d}.pt',dict(log_gain=model.log_gain.detach().cpu(),bias=model.bias.detach().cpu(),appearance_mode=cfg['appearance_mode'],eligible=ids.cpu(),soft_old_opacity=soft_alpha.cpu(),survival_gate=gate.cpu(),period=a.period,refine_ids=refine_ids.cpu(),refined=cpu(model.refined()),old_opacity=alpha.cpu(),birth=cpu(birth),birth_period=a.period,network=cpu(model.field.state_dict()),activation='immutable per-visit selected xyz/scale/DC and opacity;  soft retirement with <.005 hard zero; no physical removal claim'))
  rebuilt=state(a.run,a.period)
  assert len(rebuilt['xyz'])==len(old['xyz'])+int(keep.sum()) and all(torch.isfinite(v).all() for v in rebuilt.values())
  expected=dict(old)
  for k,v in model.refined().items():expected[k]=expected[k].index_copy(0,refine_ids,v)
  expected['opacity']=alpha
  for k in expected:assert torch.equal(rebuilt[k],torch.cat((expected[k],birth[k]),0)),k
  del rebuilt,expected
  assert history=={p:sha(p) for p in history};assert accessed=={str((src/'images'/n).resolve()) for n in names}
  dump(a.run/f'train_{a.period:04d}.json',dict(arm=cfg['arm'],topology=topology,refinement=refine_stats,period=a.period,steps=steps,names=names,accessed_real_rgb=sorted(accessed),historical_hashes=history,births_kept=int(keep.sum()),old_zeroed=int(((old['opacity']>0)&(alpha==0)).sum()),mean_gate=float(gate.sum()/max(1,gate.numel())),seconds=time.monotonic()-start,peak_vram=torch.cuda.max_memory_allocated(),losses=logs));return
 # Evaluation is a separate process; no evaluation RGB enters updates.
 import lpips
 from PIL import Image
 metric=lpips.LPIPS(net='vgg').cuda().eval();x=state(a.run,a.period);rows={};out=a.run/'renders'/str(a.period);out.mkdir(parents=True,exist_ok=True)
 with torch.no_grad():
  for i,n in enumerate(data.names(a.period,'test')):
   cam=data.camera(n);gt=data.rgb(n,cam);common=ROOT/'outputs/wat-20261007'/src.name/'test'/f'ours_40000_{a.period}'/'gt'/f'{i:05d}.png';assert np.array_equal(np.array(Image.open(common)),gt.mul(255).round().byte().permute(1,2,0).cpu().numpy())
   pred=draw(cam,view(x))
   if a.period>0:
    app=torch.load(a.run/f'period_{a.period:04d}.pt',map_location='cuda',weights_only=True);pred=app['log_gain'].exp()[:,None,None]*pred+app['bias'][:,None,None]
   pred=pred.clamp(0,1);arr=(pred*255+.5).clamp(0,255).byte().permute(1,2,0).cpu().numpy();dest=out/f'{i:05d}.png'
   if a.period==0:
    reference=ROOT/'outputs/ltgs-sequential-wat-20261007/sparse3'/src.name/'output'/src.name/'update/render_all/0'/f'{i:05d}.png'
    assert np.array_equal(arr,np.array(Image.open(reference))),'Shared initial rendering mismatch'
   if dest.exists():assert np.array_equal(arr,np.array(Image.open(dest))),'Historical pixel drift'
   else:Image.fromarray(arr).save(dest)
   scored=torch.tensor(arr,device='cuda').permute(2,0,1).float()/255
   rows[n]=dict(PSNR=float(-10*torch.log10((scored-gt).square().mean())),SSIM=float(ssim(scored,gt)),LPIPS=float(metric(scored[None],gt[None])))
 dump(a.run/f'evaluation_{a.period:04d}.json',dict(period=a.period,per_view=rows,gaussians=len(x['xyz']),heldout_rgb_exact_match=True,mean={m:float(np.mean([r[m] for r in rows.values()])) for m in ['PSNR','SSIM','LPIPS']}))
if __name__=='__main__':main()
