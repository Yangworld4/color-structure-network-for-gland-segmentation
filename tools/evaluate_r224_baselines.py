"""Post-training test pass for the exact historical R224 State-Aware runs."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT))
from tools.final_test_evaluation import SETTINGS, loaders, predictions, threshold_and_score, image_metrics, state_dict, write_csv
from models.getnetwork import get_network

def old_dir(dataset, ratio):
    root=ROOT/'checkpoints'/f'semi_{ratio}_seed_24_rgbstain_uni_reference_state_aware'/('GLaS' if dataset.lower()=='glas' else 'CRAG')
    files=list(root.glob('*/state_aware_training_complete.json'))
    if len(files)!=1: raise RuntimeError(f'R224 baseline ambiguity/missing: {dataset}{ratio}: {files}')
    return files[0].parent

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--analysis-dir',default=str(ROOT/'diagnostics/formal_r320_multidataset')); ap.add_argument('--device',default='cuda'); a=ap.parse_args()
    device=torch.device(a.device if a.device=='cuda' and torch.cuda.is_available() else 'cpu')
    rows=[]
    for dataset,ratio,label in SETTINGS:
        d=old_dir(dataset,ratio); complete=json.loads((d/'state_aware_training_complete.json').read_text())
        val_loader,test_loader=loaders(dataset,ratio)
        sources=[('best_validation','A',Path(complete['best_checkpoint_A']),int(complete['best_epochs'][0])),('best_validation','B',Path(complete['best_checkpoint_B']),int(complete['best_epochs'][1])),('epoch200','A',d/'milestones/epoch_200.pth',200),('epoch200','B',d/'milestones/epoch_200.pth',200)]
        for typ,branch,path,epoch in sources:
            payload=torch.load(path,map_location='cpu'); model=get_network('unet',3,2); model.load_state_dict(state_dict(payload,branch),strict=True); model.to(device)
            val=predictions(model,val_loader,device,branch); threshold,_,_=threshold_and_score(val); test=predictions(model,test_loader,device,branch)
            tm=image_metrics(test,threshold); fx=image_metrics(test,.5); rows.append({'dataset':dataset,'labeled_ratio':ratio,'setting':label,'reference_resolution':224,'checkpoint_type':typ,'branch':branch,'validation_selected_epoch':epoch,'threshold_mode':'validation_frozen','threshold':threshold,'Jaccard':float(np.mean([x['jaccard'] for x in tm])),'Dice':float(np.mean([x['dice'] for x in tm])),'fixed_05_Jaccard':float(np.mean([x['jaccard'] for x in fx])),'fixed_05_Dice':float(np.mean([x['dice'] for x in fx]))})
            write_csv(d/f'r224_final_test_per_image_{typ}_{branch}.csv',tm)
            del model
            if device.type=='cuda': torch.cuda.empty_cache()
    out=Path(a.analysis_dir); write_csv(out/'r224_final_test_summary.csv',rows); (out/'r224_test_evaluation_audit.json').write_text(json.dumps({'test_loaded_after_all_r320_training':True,'baseline_resolution':224,'rows':len(rows)},indent=2))
    print(json.dumps({'status':'PASS','rows':len(rows)},indent=2))
if __name__=='__main__': main()
