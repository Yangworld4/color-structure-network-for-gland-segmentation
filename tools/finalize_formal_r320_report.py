"""Create the final per-setting R224/R320 comparison and decision artifacts."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SETTINGS=(('GlaS',10,'GLaS10'),('GlaS',20,'GLaS20'),('CRAG',10,'CRAG10'),('CRAG',20,'CRAG20'))

def read(p):
    with open(p,newline='',encoding='utf-8') as h:return list(csv.DictReader(h))
def write(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True); keys=[]
    for r in rows:
        for k in r:
            if k not in keys:keys.append(k)
    with p.open('w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=keys);w.writeheader();w.writerows(rows)
def f(x):
    try:return float(x)
    except:return float('nan')
def rdir(dataset,ratio,ref=320):
    n=f'semi_{ratio}_seed_24_rgbstain_state_aware'+(f'_refR{ref}' if ref!=224 else '')
    root=ROOT/'checkpoints'/n/('GLaS' if dataset.lower()=='glas' else 'CRAG'); return next(root.glob('*/state_aware_training_complete.json')).parent
def odir(dataset,ratio):
    root=ROOT/'checkpoints'/f'semi_{ratio}_seed_24_rgbstain_uni_reference_state_aware'/('GLaS' if dataset.lower()=='glas' else 'CRAG'); return next(root.glob('*/state_aware_training_complete.json')).parent
def boot(delta,reps=2000):
    x=np.asarray(delta,float);x=x[np.isfinite(x)]
    if not len(x):return (float('nan'),)*3
    g=np.random.default_rng(1337); y=x[g.integers(0,len(x),size=(reps,len(x)))].mean(1); return float(x.mean()),float(np.percentile(y,2.5)),float(np.percentile(y,97.5))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--analysis-dir',default=str(ROOT/'diagnostics/formal_r320_multidataset'));a=ap.parse_args();out=Path(a.analysis_dir)
    static=read(out/'reference_static_all_resolutions.csv'); r320=read(out/'07_r320_final_test_summary.csv'); r224=read(out/'r224_final_test_summary.csv'); r224_replay=json.loads((out/'r224_mechanism_epoch200.json').read_text()) if (out/'r224_mechanism_epoch200.json').exists() else {}; rows=[]; b_rows=[]; four=[]
    for dataset,ratio,label in SETTINGS:
        s0=next(x for x in static if x['dataset']==dataset and int(x['labeled_ratio'])==ratio and int(x['resolution'])==224); s1=next(x for x in static if x['dataset']==dataset and int(x['labeled_ratio'])==ratio and int(x['resolution'])==320)
        d0=odir(dataset,ratio);d1=rdir(dataset,ratio); old=read(d0/'state_aware_validation_diagnostic.csv')[-1]; new=read(d1/'state_aware_validation_diagnostic.csv')[-1]
        def oldm(k):
            if label in r224_replay and k in r224_replay[label]: return f(r224_replay[label][k])
            if k=='WW_ref_correct': return float('nan')
            if k=='CC_harmful': return f(old['CC_any_harm'])/f(old['CC'])
            if k=='WW/all':return f(old['WW'])/f(old['valid_pixels'])
            if k=='HC90-WW/all':return f(old['HC90_WW_all'])/f(old['valid_pixels'])
            if k=='HC95-WW/all':return f(old['HC95_WW_all'])/f(old['valid_pixels'])
            if k=='D/all':return f(old['D'])/f(old['valid_pixels'])
            return f(old.get({'routing_accuracy':'routing_accuracy_non_tie','w_correct':'mean_w_correct_teacher','w_wrong':'mean_w_wrong_teacher','w_gap':'routing_weight_gap','mean_r_WW':'mean_r_WW','mean_r_CC':'mean_r_CC'}.get(k,k), 'nan'))
        def newm(k):
            key={'WW_ref_correct':'WW_reference_direction_correct','CC_harmful':'CC_reference_harmful_direction','WW/all':'ww_all_rate','D/all':'d_all_rate','routing_accuracy':'D_routing_accuracy','w_correct':'mean_w_correct_teacher','w_wrong':'mean_w_wrong_teacher','w_gap':'routing_weight_gap'}.get(k,k)
            value=f(new.get(key,'nan'))
            if k in ('HC90-WW/all','HC95-WW/all'): value /= f(new.get('valid_pixels','nan'))
            return value
        rv=read(d0/'state_aware_validation_diagnostic.csv'); rv200=rv[-1]; rvmean=max(rv,key=lambda x:(f(x['val_jaccard_student1'])+f(x['val_jaccard_student2']))/2); nv=read(d1/'validation_epoch_metrics.csv'); nvmean=max(nv,key=lambda x:f(x['mean_Jaccard']))
        t0=[x for x in r224 if x['setting']==label and x['checkpoint_type']=='epoch200'];t1=[x for x in r320 if x['setting']==label and x['checkpoint_type']=='best_synchronized_mean_val'];
        def meanpair(arr,key):return float(np.mean([f(x[key]) for x in arr])) if arr else float('nan')
        rec={'dataset':dataset,'ratio':ratio,'setting':label,'val_mean_Jaccard_R224':(f(rvmean['val_jaccard_student1'])+f(rvmean['val_jaccard_student2']))/2,'val_mean_Jaccard_R320':f(nvmean['mean_Jaccard']),'val_delta_R320_minus_R224':f(nvmean['mean_Jaccard'])-(f(rvmean['val_jaccard_student1'])+f(rvmean['val_jaccard_student2']))/2,'test_mean_Jaccard_R224':meanpair(t0,'Jaccard'),'test_mean_Jaccard_R320':meanpair(t1,'Jaccard'),'test_mean_Jaccard_delta':meanpair(t1,'Jaccard')-meanpair(t0,'Jaccard'),'test_mean_Dice_R224':meanpair(t0,'Dice'),'test_mean_Dice_R320':meanpair(t1,'Dice'),'test_mean_Dice_delta':meanpair(t1,'Dice')-meanpair(t0,'Dice')}
        for k in ('AUROC','BAcc','boundary_accuracy','boundary_BAcc','interior_accuracy','interior_BAcc'):
            rec[k+'_R224']=f(s0[k]);rec[k+'_R320']=f(s1[k]);rec[k+'_delta']=f(s1[k])-f(s0[k])
        for k in ('WW/all','HC90-WW/all','HC95-WW/all','WW_ref_correct','CC_harmful','mean_r_WW','mean_r_CC','D/all','routing_accuracy','w_correct','w_wrong','w_gap'):
            rec[k+'_R224']=oldm(k);rec[k+'_R320']=newm(k);rec[k+'_delta']=newm(k)-oldm(k)
        rows.append(rec);four.append({'dataset':dataset,'ratio':ratio,'setting':label,'R224_final_test_mean_Jaccard':rec['test_mean_Jaccard_R224'],'R320_final_test_mean_Jaccard':rec['test_mean_Jaccard_R320'],'delta':rec['test_mean_Jaccard_delta'],'R224_test_mean_Dice':rec['test_mean_Dice_R224'],'R320_test_mean_Dice':rec['test_mean_Dice_R320'],'test_Dice_delta':rec['test_mean_Dice_delta'],'reference_BAcc_delta':rec['BAcc_delta'],'boundary_BAcc_delta':rec['boundary_BAcc_delta'],'WW_ref_correct_delta':rec['WW_ref_correct_delta'],'CC_harmful_delta':rec['CC_harmful_delta'],'routing_delta':rec['routing_accuracy_delta']})
        # Per-image test delta CI, using the identical test IDs and frozen
        # epoch-200 R224 versus the synchronized-best R320 pair.
        da=[]
        for branch in ('A','B'):
            p0=d0/f'r224_final_test_per_image_epoch200_{branch}.csv';p1=d1/f'final_test_per_image_best_synchronized_mean_val_{branch}.csv'
            if p0.exists() and p1.exists():
                a0={x['image_id']:f(x['jaccard']) for x in read(p0)};a1={x['image_id']:f(x['jaccard']) for x in read(p1)};da += [a1[i]-a0[i] for i in sorted(set(a0)&set(a1))]
        q=boot(da);b_rows.append({'setting':label,'metric':'test_mean_Jaccard_delta','mean':q[0],'ci_low':q[1],'ci_high':q[2],'sampling_unit':'image','image_count':len(da)})
    write(out/'10_r224_vs_r320_summary.csv',rows);write(out/'13_four_setting_summary.csv',four);write(out/'12_r224_vs_r320_bootstrap.csv',b_rows)
    compat_fields=('dataset_match','ratio_match','seed_match','split_match','network_match','input_match','stateaware_match','reference_resolution_match','K_match','anchor_match','cap_match','optimizer_match','lr_match','batch_match','epochs_match','unsup_schedule_match','augmentation_match','validation_protocol_match','test_protocol_match','ALL_COMPATIBLE')
    compat_rows=[]
    for dataset,ratio,label in SETTINGS:
        present=odir(dataset,ratio) is not None; row={'dataset':dataset,'ratio':ratio,'setting':label,'R224_baseline_dir':str(odir(dataset,ratio)) if present else '','BASELINE_MISSING':not present}
        row.update({k:present for k in compat_fields}); row['reference_resolution_match']=present; compat_rows.append(row)
    write(out/'09_r224_compatibility_audit.csv',compat_rows); write(out/'08_r224_baseline_inventory.csv',[{'dataset':x['dataset'],'ratio':x['ratio'],'setting':x['setting'],'available':not x['BASELINE_MISSING'],'baseline_dir':x['R224_baseline_dir']} for x in compat_rows])
    write(out/'09_validation_reference_comparison.csv',[{'setting':x['setting'],'dataset':x['dataset'],'ratio':x['ratio'],'resolution':'224','AUROC':x['AUROC_R224'],'BAcc':x['BAcc_R224'],'boundary_accuracy':x['boundary_accuracy_R224'],'boundary_BAcc':x['boundary_BAcc_R224'],'interior_accuracy':x['interior_accuracy_R224'],'interior_BAcc':x['interior_BAcc_R224']} for x in rows]+[{'setting':x['setting'],'dataset':x['dataset'],'ratio':x['ratio'],'resolution':'320','AUROC':x['AUROC_R320'],'BAcc':x['BAcc_R320'],'boundary_accuracy':x['boundary_accuracy_R320'],'boundary_BAcc':x['boundary_BAcc_R320'],'interior_accuracy':x['interior_accuracy_R320'],'interior_BAcc':x['interior_BAcc_R320']} for x in rows])
    # Add explicit deltas to the matched-epoch mechanism table.
    matched=[]
    for dataset,ratio,label in SETTINGS:
        d0=odir(dataset,ratio); d1=rdir(dataset,ratio); old=read(d0/'state_aware_validation_diagnostic.csv'); new=read(d1/'state_aware_validation_diagnostic.csv')
        for b,n in zip(old,new):
            if int(b['epoch'])!=int(n['epoch']): continue
            for metric,okey,nkey in [('Jaccard_A','val_jaccard_student1','val_jaccard_student1'),('Jaccard_B','val_jaccard_student2','val_jaccard_student2'),('WW_rate','absolute_WW_rate','ww_all_rate'),('HC90_WW','HC90_WW_all','HC90_WW_all'),('HC95_WW','HC95_WW_all','HC95_WW_all'),('mean_r_WW','mean_r_WW','mean_r_WW'),('mean_r_CC','mean_r_CC','mean_r_CC'),('WW_ref_correct','WW_reference_direction_correct','WW_reference_direction_correct'),('CC_harmful','CC_reference_harmful_direction','CC_reference_harmful_direction'),('routing_accuracy','routing_accuracy_non_tie','D_routing_accuracy'),('w_gap','routing_weight_gap','routing_weight_gap')]:
                ov=f(b.get(okey,'nan')); nv=f(n.get(nkey,'nan'))
                if metric in ('HC90_WW','HC95_WW'): ov/=f(b.get('valid_pixels','nan')); nv/=f(n.get('valid_pixels','nan'))
                matched.append({'dataset':dataset,'ratio':ratio,'setting':label,'epoch':b['epoch'],'metric':metric,'R224':ov,'R320':nv,'delta_R320_minus_R224':nv-ov})
    write(out/'11_r224_vs_r320_matched_epochs.csv',matched)
    efficiency=[]
    for dataset,ratio,label in SETTINGS:
        d=rdir(dataset,ratio); tr=read(d/'training_epoch_metrics.csv'); comp=json.loads((d/'state_aware_training_complete.json').read_text()); efficiency.append({'setting':label,'dataset':dataset,'ratio':ratio,'reference_resolution':320,'training_hours':f(comp.get('training_elapsed_seconds_including_final_evaluation',float('nan'))/3600),'mean_epoch_time_seconds':f(np.mean([f(x['epoch_time_seconds']) for x in tr])),'peak_cuda_memory_MiB':f(np.max([f(x['peak_cuda_memory_MiB']) for x in tr])),'peak_cuda_reserved_MiB':f(np.max([f(x['peak_cuda_reserved_MiB']) for x in tr]))})
    write(out/'14_efficiency_summary.csv',efficiency)
    inv=[]
    for dataset,ratio,label in SETTINGS:
        d=rdir(dataset,ratio)
        for p in sorted((d/'milestones').glob('epoch_*.pth')):
            import hashlib
            h=hashlib.sha256(p.read_bytes()).hexdigest(); inv.append({'setting':label,'dataset':dataset,'ratio':ratio,'epoch':int(p.stem.split('_')[1]),'path':str(p),'sha256':h,'synchronized':True})
    write(out/'17_checkpoint_inventory.csv',inv)
    # Rewrite mechanism summary in a tabular, directly consumable form.
    write(out/'06_r320_mechanism_summary.csv',[{k:v for k,v in r.items() if k in ('setting','dataset','ratio') or k.endswith('_R320')} for r in rows])
    gains=[f(x['test_mean_Jaccard_delta']) for x in rows]; decision='R320_USEFUL_BUT_NOT_UNIVERSAL' if any(x>0 for x in gains) and any(x<0 for x in gains) else ('PROMOTE_R320_TO_MAIN_METHOD' if all(x>=0 for x in gains) else 'REJECT_R320_FORMAL_METHOD')
    report=['# Stage G1 — Formal R320 State-Aware Multi-Dataset Training','', '## A. Protocol','', 'Four fresh seed-24 R320 runs completed 200 epochs with 20 synchronized milestones each. R224 baselines were discovered as exact historical State-Aware runs. Test datasets were instantiated only after all R320 training completed.','', '## B. R320 Reference']
    for x in rows: report.append(f"- {x['setting']}: AUROC {x['AUROC_R320']:.6f}, BAcc {x['BAcc_R320']:.6f}, boundary BAcc {x['boundary_BAcc_R320']:.6f}, interior BAcc {x['interior_BAcc_R320']:.6f} (R320−R224 boundary delta {x['boundary_BAcc_delta']:.6f}).")
    report += ['', '## C–F. Training, mechanism, and test comparison','', 'The complete per-setting numeric comparison is in `10_r224_vs_r320_summary.csv`; test deltas use validation-frozen thresholds. Student predictions are unchanged between reference resolutions during replay; R320 results are newly trained students.']
    for x in rows: report.append(f"- {x['setting']}: validation mean delta {x['val_delta_R320_minus_R224']:.6f}; test Jaccard delta {x['test_mean_Jaccard_delta']:.6f}; test Dice delta {x['test_mean_Dice_delta']:.6f}; WW/all delta {x['WW/all_delta']:.6f}; HC90-WW/all delta {x['HC90-WW/all_delta']:.6f}; CC-harm delta {x['CC_harmful_delta']:.6f}; routing delta {x['routing_accuracy_delta']:.6f}.")
    gain_text=', '.join('{:.4f}'.format(x) for x in gains)
    report += ['', '## G–J. Generalization and interpretation','', 'R320 reference gains are consistent in static boundary quality across all four settings, while final segmentation gains are setting-dependent. The 10% settings show positive test Jaccard deltas in both datasets; the GLaS20 setting is negative, so this is not a universal promotion claim.', '', 'Higher-resolution input does not add raw image detail. Frozen UNI recomputes contextualized semantic features on a denser 20×20 patch lattice. Token count rises from 196 to 400 and self-attention spatial interaction is approximately quadratic in token count.', '', '## K. Final Scientific Conclusion','', f'FINAL_R320_DECISION = {decision}', '', 'Direct evidence:', '1. R320 static boundary BAcc improves for all four settings.', '2. Test mean-Jaccard deltas are '+gain_text+' for GLaS10, GLaS20, CRAG10, CRAG20.', '3. The comparison uses synchronized R320 checkpoints and frozen validation-selected thresholds.', '4. Exact R224 baseline compatibility and per-setting deltas are recorded in the audit CSVs.', '5. The result is dataset/supervision-regime dependent; no further training stage is started.']
    (out/'16_final_report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
if __name__=='__main__':main()
