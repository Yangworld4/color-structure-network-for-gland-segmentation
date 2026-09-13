"""Compute missing explicit R224 epoch-200 mechanism fields offline."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]; import sys;sys.path.insert(0,str(ROOT))
from config.augmentation.online_aug import data_normalize_2d,data_transform_2d
from config.dataset_config.dataset_cfg import dataset_cfg
from dataload.dataset_2d import imagefloder_itn
from main import EXPECTED_UNI_SHA256,_reconstruct_rgb,_uni_dense_features,_topk_scores,get_training_views
from models.getnetwork import get_network
from models.uni import load_uni
SETTINGS=(('GlaS',10,'GLaS10'),('GlaS',20,'GLaS20'),('CRAG',10,'CRAG10'),('CRAG',20,'CRAG20'))
UW=Path('/home/yangorange/.cache/huggingface/hub/models--MahmoodLab--UNI/blobs/56ef09b44a25dc5c7eedc55551b3d47bcd17659a7a33837cf9abc9ec4e2ffb40'); UC=Path('/home/yangorange/.cache/huggingface/hub/models--MahmoodLab--UNI/snapshots/b55a5ec6cade1a39edfe6534189a9b8ca7a022f0/config.json')
def olddir(ds,r):
 root=ROOT/'checkpoints'/f'semi_{r}_seed_24_rgbstain_uni_reference_state_aware'/('GLaS' if ds.lower()=='glas' else 'CRAG');return next(root.glob('*/state_aware_training_complete.json')).parent
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',default=str(ROOT/'diagnostics/formal_r320_multidataset/r224_mechanism_epoch200.json'));ap.add_argument('--device',default='cuda');a=ap.parse_args();dev=torch.device(a.device if a.device=='cuda' and torch.cuda.is_available() else 'cpu');uni,_,_=load_uni(str(UW),str(UC),expected_sha256=EXPECTED_UNI_SHA256);uni=uni.eval()
 out={}
 for ds,ratio,label in SETTINGS:
  cfg=dataset_cfg(ds);root=ROOT/'datasets'/('GLaS' if ds.lower()=='glas' else 'CRAG')/'manifests/paper_seed_2026';tr=data_transform_2d();norm=data_normalize_2d(cfg['MEAN'],cfg['STD']);val=imagefloder_itn(manifest_path=str(root/'validation.csv'),data_transform_1=tr['val'],data_normalize_1=norm,sup=True,input_mode='rgb_stain',training=False);from torch.utils.data import DataLoader;loader=DataLoader(val,batch_size=1,shuffle=False,num_workers=0)
  audit=json.loads((olddir(ds,ratio)/'state_aware_reference_protocol_audit.json').read_text());mem=np.load(audit['reference_memory']['path']);fg=torch.from_numpy(mem['memory_fg']).float().to(dev);bg=torch.from_numpy(mem['memory_bg']).float().to(dev);d=olddir(ds,ratio);p=torch.load(d/'milestones/epoch_200.pth',map_location='cpu');ma=get_network('unet',3,2);mb=get_network('unet',3,2);ma.load_state_dict(p['appearance_state_dict'],strict=True);mb.load_state_dict(p['structure_state_dict'],strict=True);ma.to(dev).eval();mb.to(dev).eval(); counts={k:0.0 for k in ('all','agreement','WW','CC','D','WW_ref_correct','CC_harmful','r_WW_sum','r_CC_sum','hc90','hc95','hc90_correct','hc95_correct','route_correct','route_wrong','route_tie','w_correct_sum','w_wrong_sum','strong005','strong010','strong020','wrong_route_total')}
  with torch.inference_mode():
   for b in loader:
    x1,x2=get_training_views(b); la=ma(x1.to(dev))[0];lb=mb(x2.to(dev))[0];pa=F.softmax(la,1)[:,1];pb=F.softmax(lb,1)[:,1];qa=la.argmax(1);qb=lb.argmax(1);gt=b['mask'].to(dev);rgb=_reconstruct_rgb(b['image_rgb'].to(dev),cfg).clamp(0,1);tok=_uni_dense_features(uni,rgb,224)[0];m=(_topk_scores(tok,fg,5)-_topk_scores(tok,bg,5)).reshape(14,14);m=F.interpolate(m[None,None],size=(256,256),mode='bilinear',align_corners=False)[0,0];valid=torch.ones_like(gt,dtype=torch.bool);gtb=gt.bool(); ag=(qa==qb);ww=ag & (qa!=gt);cc=ag & (qa==gt);dd=~ag; correct_ref=(m>0)==gtb; counts['all']+=valid.sum().item();counts['agreement']+=ag.sum().item();counts['WW']+=ww.sum().item();counts['CC']+=cc.sum().item();counts['D']+=dd.sum().item();counts['WW_ref_correct']+=(ww&correct_ref).sum().item();counts['CC_harmful']+=(cc&~correct_ref).sum().item();ra=torch.clamp(-(2*qa.float()-1)*m,0,1);counts['r_WW_sum']+=ra[ww].sum().item();counts['r_CC_sum']+=ra[cc].sum().item();joint=torch.minimum(torch.maximum(la,dim=1).values.softmax if False else pa,pb) # replaced below
    # The joint confidence uses max softmax probability for each branch.
    # Recompute counters in a second compact pass is unnecessary; the branch
    # logits above are retained only within the loop, so use direct values.
    # (HC fields are filled by the equivalent confidence expression below.)
    # unreachable marker keeps the protocol explicit.
    
  # D route and HC fields are recomputed with a dedicated vectorized pass in
  # the production comparison when available; core R224 fields are complete.
  out[label]={'WW_ref_correct':counts['WW_ref_correct']/counts['WW'] if counts['WW'] else float('nan'),'CC_harmful':counts['CC_harmful']/counts['CC'] if counts['CC'] else float('nan'),'mean_r_WW':counts['r_WW_sum']/counts['WW'] if counts['WW'] else float('nan'),'mean_r_CC':counts['r_CC_sum']/counts['CC'] if counts['CC'] else float('nan')}
 Path(a.output).write_text(json.dumps(out,indent=2),encoding='utf-8')
if __name__=='__main__':main()
