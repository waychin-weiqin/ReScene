"""Validate ReScene real histories and report arrival/final archival recall."""
import math
from statistics import mean
import numpy as np
from PIL import Image,ImageDraw
from real_multistate_common import *
def main():
 p=freeze();rows=[];scenes={};missing=[];gallery=O/'comparison';gallery.mkdir(exist_ok=True);html=['<h1>Multi-State: GT | ReScene, 24 current training views</h1><p>Supplied all-view SfM and offline TRAIN alignment; real initial SfM points; later depths estimated by MASt3R.</p>'];lines=['# Real Multi-State ReScene','','Supplied real heldouts; up-to24currentRGB,5000steps/update,standard3DGS initial30k nominal. No period0 update. Immutable versioned history rerendered after every update. All-view calibration and offline TRAIN alignment disclosed.','','| Scene | Period | PSNR | SSIM | LPIPS | Births |','|---|---:|---:|---:|---:|---:|']
 for s,j in p['jobs'].items():
  run=O/s/'full';last=max(j['periods'])
  if not (run/f'eval-{last}-after-{last}.done.json').exists():missing.append(s);continue
  q=read(Path(j['source'])/'wat_protocol.json');gtm=read(O/s/'gt-manifest.json');stats=[]
  assert read(run/'config.json')['initial_sha256']==sha(j['initial_ply'])
  for t in j['periods']:
   ev=read(run/f'evaluation_{t:04d}.json');names=sorted(n for n in q['test_names'] if n.startswith(f'IMG_{t:04d}/'));assert set(names)==set(ev['per_view'])
   births=0
   if t:
    tr=read(run/f'train_{t:04d}.json');assert tr['steps']==5000;assert tr['names']==sorted(n for n in q['update_names'] if n.startswith(f'IMG_{t:04d}/'));assert set(tr['accessed_real_rgb'])=={str((Path(j['source'])/'images'/n).resolve()) for n in tr['names']}
    assert read(run/f'prepare_audit_{t:04d}.json')['cache_sha']==sha(run/f'prepared_{t:04d}.pt')
    for f,h in tr['historical_hashes'].items():assert sha(f)==h
    for old in range(t+1):assert (run/f'eval-{old}-after-{t}.done.json').exists()
    births=tr['births_kept']
   for i,n in enumerate(names):
    gtpath=Path(gtm[n]['path']);assert sha(gtpath)==gtm[n]['sha'];gt=np.array(Image.open(gtpath));render=run/f'renders/{t}/{i:05d}.png';im=np.array(Image.open(render));scores=ev['per_view'][n];assert np.isfinite(im).all() and all(math.isfinite(v) for v in scores.values())
    recompute=float(-10*np.log10(max(np.mean(((im.astype(float)-gt)/255)**2),1e-12)))
    assert abs(recompute-scores['PSNR'])<1e-4
    panels=[('GT',Image.fromarray(gt)),(f'Ours {scores["PSNR"]:.2f}dB',Image.fromarray(im))]
    h,w=gt.shape[:2];canvas=Image.new('RGB',(600*len(panels),round(h*600/w)+36),'white');draw=ImageDraw.Draw(canvas)
    for k,(label,img) in enumerate(panels):canvas.paste(img.resize((600,round(h*600/w))),(600*k,36));draw.text((600*k+6,8),label,fill='black')
    fn=f'{s}-p{t}-{i:03d}.jpg';canvas.save(gallery/fn,quality=92);html.append(f'<h3>{s} period {t}: {n}</h3><img loading="lazy" width="100%" src="{fn}">')
    rows.append(dict(scene=s,period=t,name=n,scores=scores,render=str(render),render_sha=sha(render),gt=str(gtpath),gt_sha=sha(gtpath),history_pixel_exact=True))
   stat={'period':t,'metrics':ev['mean'],'births':births,'update_seconds':tr['seconds'] if t else 0};stats.append(stat);lines.append(f'| {s} | {t} | {ev["mean"]["PSNR"]:.3f} | {ev["mean"]["SSIM"]:.4f} | {ev["mean"]["LPIPS"]:.4f} | {births} |')
  scenes[s]={'periods':stats,'all_period':{m:mean(v['metrics'][m] for v in stats) for m in ['PSNR','SSIM','LPIPS']},'later_only':{m:mean(v['metrics'][m] for v in stats[1:]) for m in ['PSNR','SSIM','LPIPS']},'history_storage_bytes':sum(f.stat().st_size for f in [run/'initial.pt',*run.glob('period_*.pt')])}
 for f,h in p['inputs'].items():assert sha(f)==h,f
 dump(A/'results.json',dict(complete=not missing,missing=missing,scenes=scenes,rows=rows));(A/'report.md').write_text('\n'.join(lines)+'\n');(gallery/'index.html').write_text('\n'.join(html));print({'complete':not missing,'scenes':len(scenes),'views':len(rows),'missing':missing})
 if not missing:assert len(rows)==sum(j['test_views'] for j in p['jobs'].values())
if __name__=='__main__':main()
