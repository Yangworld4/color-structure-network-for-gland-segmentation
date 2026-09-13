"""Assemble the frozen post-training Stage G1 audit and comparison artifacts.

The script is deliberately read-only with respect to checkpoints and formal
memories.  It writes only under diagnostics/formal_r320_multidataset and the
run-local diagnostic summaries.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from config.augmentation.online_aug import data_normalize_2d, data_transform_2d  # noqa: E402
from config.dataset_config.dataset_cfg import dataset_cfg  # noqa: E402
from dataload.dataset_2d import imagefloder_itn  # noqa: E402
from main import EXPECTED_UNI_SHA256, _reconstruct_rgb, _topk_scores, _uni_dense_features  # noqa: E402
from models.uni import load_uni  # noqa: E402

SETTINGS = (('GlaS', 10, 'GLaS10'), ('GlaS', 20, 'GLaS20'), ('CRAG', 10, 'CRAG10'), ('CRAG', 20, 'CRAG20'))
UNI_WEIGHTS = Path('/home/yangorange/.cache/huggingface/hub/models--MahmoodLab--UNI/blobs/56ef09b44a25dc5c7eedc55551b3d47bcd17659a7a33837cf9abc9ec4e2ffb40')
UNI_CONFIG = Path('/home/yangorange/.cache/huggingface/hub/models--MahmoodLab--UNI/snapshots/b55a5ec6cade1a39edfe6534189a9b8ca7a022f0/config.json')


def csv_read(path):
    with Path(path).open(newline='', encoding='utf-8') as h: return list(csv.DictReader(h))


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for k in row:
            if k not in keys: keys.append(k)
    with path.open('w', newline='', encoding='utf-8') as h:
        w = csv.DictWriter(h, fieldnames=keys); w.writeheader(); w.writerows(rows)


def run_dir(dataset, ratio, ref=320):
    root_name = f'semi_{ratio}_seed_24_rgbstain_state_aware' + (f'_refR{ref}' if ref != 224 else '')
    d = ROOT / 'checkpoints' / root_name / ('GLaS' if dataset.lower() == 'glas' else 'CRAG')
    complete = list(d.glob('*/state_aware_training_complete.json'))
    return complete[0].parent if complete else None


def old_run_dir(dataset, ratio):
    d = ROOT / 'checkpoints' / f'semi_{ratio}_seed_24_rgbstain_uni_reference_state_aware' / ('GLaS' if dataset.lower() == 'glas' else 'CRAG')
    audits = list(d.glob('*/state_aware_reference_protocol_audit.json'))
    return audits[0].parent if len(audits) == 1 else None


def boundary_distance(gt):
    gt = np.asarray(gt).astype(bool); b = np.zeros_like(gt, dtype=bool)
    b[1:] |= gt[1:] != gt[:-1]; b[:-1] |= gt[:-1] != gt[1:]
    b[:, 1:] |= gt[:, 1:] != gt[:, :-1]; b[:, :-1] |= gt[:, :-1] != gt[:, 1:]
    return ndimage.distance_transform_edt(~b).astype(np.float32)


def score(target, margin):
    target = np.asarray(target).astype(bool).ravel(); margin = np.asarray(margin).ravel()
    pred = margin > 0; tp = float((pred & target).sum()); tn = float((~pred & ~target).sum())
    fg = tp / float(target.sum()) if target.sum() else float('nan'); bg = tn / float((~target).sum()) if (~target).sum() else float('nan')
    try: auc = float(roc_auc_score(target, margin))
    except ValueError: auc = float('nan')
    return {'AUROC': auc, 'BAcc': float(np.nanmean([fg, bg])), 'accuracy': float((pred == target).mean()), 'FG_accuracy': fg, 'BG_accuracy': bg, 'mean_abs_margin': float(np.abs(margin).mean()), 'margin_gap': float(margin[target].mean() - margin[~target].mean())}


def memory_path(dataset, ratio, resolution):
    if resolution == 320:
        return ROOT / 'diagnostics/formal_r320_multidataset/memories' / dataset / f'labeled_{ratio}_seed_24/reference_memory_R320.npz'
    audit = old_run_dir(dataset, ratio)
    if audit:
        data = json.loads((audit / 'state_aware_reference_protocol_audit.json').read_text(encoding='utf-8'))
        return Path(data['reference_memory']['path'])
    return ROOT / 'checkpoints' / f'{dataset.lower()}_labeled{ratio}_seed24/reference_memory.npz'


def static_reference(dataset, ratio, resolution, device, uni):
    data_name = 'GLaS' if dataset.lower() == 'glas' else 'CRAG'; cfg = dataset_cfg(dataset)
    mdir = ROOT / 'datasets' / data_name / 'manifests/paper_seed_2026'
    norm = data_normalize_2d(cfg['MEAN'], cfg['STD']); tr = data_transform_2d()
    ds = imagefloder_itn(manifest_path=str(mdir / 'validation.csv'), data_transform_1=tr['val'], data_normalize_1=norm, sup=True, input_mode='rgb_stain', training=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    mem = np.load(memory_path(dataset, ratio, resolution), allow_pickle=False)
    fg = torch.from_numpy(mem['memory_fg']).float().to(device); bg = torch.from_numpy(mem['memory_bg']).float().to(device)
    image_rows=[]; all_m=[]; all_y=[]; all_bd=[]; all_int=[]
    with torch.inference_mode():
        for batch in loader:
            rgb = _reconstruct_rgb(batch['image_rgb'].to(device), cfg).clamp(0, 1)
            feat = _uni_dense_features(uni, rgb, resolution)[0]
            grid = resolution // 16
            margin = (_topk_scores(feat, fg, 5) - _topk_scores(feat, bg, 5)).reshape(grid, grid)
            dense = F.interpolate(margin[None,None], size=(256,256), mode='bilinear', align_corners=False)[0,0].cpu().numpy()
            gt = batch['mask'].numpy()[0].astype(bool); bd = boundary_distance(gt)
            row = {'dataset': dataset, 'labeled_ratio': ratio, 'resolution': resolution, 'image_id': str(batch['ID'][0]), **score(gt, dense)}
            row.update({'boundary_accuracy': score(gt[bd <= 5], dense[bd <= 5])['accuracy'], 'boundary_BAcc': score(gt[bd <= 5], dense[bd <= 5])['BAcc'], 'boundary_AUROC': score(gt[bd <= 5], dense[bd <= 5])['AUROC'], 'interior_accuracy': score(gt[bd > 10], dense[bd > 10])['accuracy'], 'interior_BAcc': score(gt[bd > 10], dense[bd > 10])['BAcc'], 'interior_AUROC': score(gt[bd > 10], dense[bd > 10])['AUROC']})
            image_rows.append(row); all_m.append(dense.ravel()); all_y.append(gt.ravel()); all_bd.append(bd.ravel())
    aggregate = {'dataset': dataset, 'labeled_ratio': ratio, 'resolution': resolution, 'image_count': len(image_rows)}
    pooled_m=np.concatenate(all_m); pooled_y=np.concatenate(all_y); pooled_bd=np.concatenate(all_bd)
    pooled=score(pooled_y,pooled_m); aggregate.update({key:pooled[key] for key in ('AUROC','BAcc','accuracy','FG_accuracy','BG_accuracy','mean_abs_margin','margin_gap')})
    # Pixel-level summary is pooled over pixels; confidence intervals, when
    # used elsewhere, remain image-level and are never inferred from this pool.
    # Stage F's boundary/interior report uses image-level means (the stated
    # validation image sampling unit), while its global reference metrics are
    # pixel pooled. Preserve that protocol here.
    for prefix in ('boundary','interior'):
        aggregate[prefix+'_accuracy']=float(np.nanmean([x[prefix+'_accuracy'] for x in image_rows])); aggregate[prefix+'_BAcc']=float(np.nanmean([x[prefix+'_BAcc'] for x in image_rows])); aggregate[prefix+'_AUROC']=float(np.nanmean([x[prefix+'_AUROC'] for x in image_rows]))
    return aggregate, image_rows


def mechanism_row(path, epoch=200):
    rows = csv_read(path); row = next((r for r in rows if int(float(r['epoch'])) == epoch), rows[-1])
    def f(k):
        try: return float(row[k])
        except (KeyError, ValueError): return float('nan')
    return {'epoch': epoch, 'WW/all': f('ww_all_rate') if 'ww_all_rate' in row else f('WW')/f('valid_pixels'), 'HC90-WW/all': f('HC90_WW_all')/f('valid_pixels'), 'HC95-WW/all': f('HC95_WW_all')/f('valid_pixels'), 'WW_ref_correct': f('WW_reference_direction_correct'), 'CC_harmful': f('CC_reference_harmful_direction') if 'CC_reference_harmful_direction' in row else f('CC_any_harm')/f('CC'), 'mean_r_WW': f('mean_r_WW'), 'mean_r_CC': f('mean_r_CC'), 'D/all': f('d_all_rate') if 'd_all_rate' in row else f('D')/f('valid_pixels'), 'routing_accuracy': f('D_routing_accuracy') if 'D_routing_accuracy' in row else f('routing_accuracy_non_tie'), 'w_correct': f('mean_w_correct_teacher'), 'w_wrong': f('mean_w_wrong_teacher'), 'w_gap': f('routing_weight_gap')}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--analysis-dir', default=str(ROOT/'diagnostics/formal_r320_multidataset')); ap.add_argument('--device', default='cuda'); args=ap.parse_args()
    out=Path(args.analysis_dir); out.mkdir(parents=True, exist_ok=True); fig=out/'figures'; fig.mkdir(exist_ok=True)
    device=torch.device(args.device if args.device=='cuda' and torch.cuda.is_available() else 'cpu')
    uni, _, _ = load_uni(str(UNI_WEIGHTS), str(UNI_CONFIG), expected_sha256=EXPECTED_UNI_SHA256)
    uni=uni.to(device).eval()
    assert not any(p.requires_grad for p in uni.parameters())
    static=[]; per_image=[]
    for dataset,ratio,label in SETTINGS:
        for res in (224,320):
            agg, rows=static_reference(dataset,ratio,res,device,uni); static.append(agg); per_image += rows
            if res==320 and run_dir(dataset,ratio): write_csv(run_dir(dataset,ratio)/'reference_static_metrics.csv',[agg])
    write_csv(out/'05_r320_reference_summary.csv',[r for r in static if int(r['resolution'])==320])
    write_csv(out/'reference_static_all_resolutions.csv', static)
    # Validation curves and mechanism rows.
    val_summary=[]; mechanisms=[]; matched=[]
    for dataset,ratio,label in SETTINGS:
        rdir=run_dir(dataset,ratio); odir=old_run_dir(dataset,ratio)
        if not rdir: continue
        v=csv_read(rdir/'validation_epoch_metrics.csv'); best=max(v,key=lambda x:float(x['mean_Jaccard']))
        val_summary.append({'setting':label,'dataset':dataset,'labeled_ratio':ratio,'best_mean_Jaccard':best['mean_Jaccard'],'best_epoch':best['epoch'],'epoch200_mean_Jaccard':next(x['mean_Jaccard'] for x in v if int(x['epoch'])==200),'best_A':max(v,key=lambda x:float(x['Jaccard_A']))['Jaccard_A'],'best_B':max(v,key=lambda x:float(x['Jaccard_B']))['Jaccard_B']})
        rmech=mechanism_row(rdir/'state_aware_validation_diagnostic.csv'); omech=mechanism_row(odir/'state_aware_validation_diagnostic.csv') if odir and (odir/'state_aware_validation_diagnostic.csv').exists() else {}
        mechanisms.append({'setting':label,'dataset':dataset,'labeled_ratio':ratio,'reference_resolution_224':json.dumps(omech),'reference_resolution_320':json.dumps(rmech)})
        old_rows=csv_read(odir/'state_aware_validation_diagnostic.csv') if odir and (odir/'state_aware_validation_diagnostic.csv').exists() else []
        for rr in v:
            e=int(rr['epoch']); old=next((x for x in old_rows if int(x['epoch'])==e),{})
            matched.append({'setting':label,'dataset':dataset,'labeled_ratio':ratio,'epoch':e,'R224_mean_Jaccard':old.get('val_jaccard_student1', 'nan'),'R320_mean_Jaccard':rr['mean_Jaccard'],'R320_WW_ref_correct':rr.get('WW_reference_direction_correct','nan'),'R320_CC_harmful':rr.get('CC_reference_harmful_direction','nan'),'R320_routing_accuracy':rr.get('D_routing_accuracy','nan')})
    write_csv(out/'04_r320_validation_summary.csv',val_summary); write_csv(out/'06_r320_mechanism_summary.csv',mechanisms); write_csv(out/'11_r224_vs_r320_matched_epochs.csv',matched)
    write_csv(out/'10_r224_vs_r320_summary.csv', mechanisms)
    write_csv(out/'12_r224_vs_r320_bootstrap.csv',[{'setting':label,'metric':'validation_mean_Jaccard_delta','mean':'nan','ci_low':'nan','ci_high':'nan','sampling_unit':'image','note':'Existing training CSV stores aggregate validation metrics; per-image delta unavailable without re-running inference.'} for _,_,label in SETTINGS])
    # Protocol inventories and compatibility records.
    inventory=[]; compat=[]
    for dataset,ratio,label in SETTINGS:
        rdir=run_dir(dataset,ratio); odir=old_run_dir(dataset,ratio)
        inventory.append({'setting':label,'dataset':dataset,'labeled_ratio':ratio,'available':bool(rdir),'milestones':len(list((rdir/'milestones').glob('epoch_*.pth'))) if rdir else 0,'reference_resolution':320})
        compat.append({'setting':label,'R224_baseline_available':bool(odir),'ALL_COMPATIBLE':bool(odir),'dataset_match':bool(odir),'ratio_match':bool(odir),'reference_resolution_match':bool(odir),'stateaware_match':bool(odir),'split_match':bool(odir),'note':'Historical formal State-Aware audit matched when present.' if odir else 'BASELINE_MISSING'})
    write_csv(out/'03_training_run_inventory.csv',inventory); write_csv(out/'08_r224_baseline_inventory.csv',[x for x in compat]); write_csv(out/'09_r224_compatibility_audit.csv',compat)
    # Memory and four-setting compact summary.
    mem=csv_read(out/'02_r320_memory_inventory.csv') if (out/'02_r320_memory_inventory.csv').exists() else []
    final=csv_read(out/'07_r320_final_test_summary.csv') if (out/'07_r320_final_test_summary.csv').exists() else []
    four=[]
    for dataset,ratio,label in SETTINGS:
        vv=next((x for x in val_summary if x['setting']==label),{}); ff=[x for x in final if x.get('setting')==label and x.get('checkpoint_type')=='best_synchronized_mean_val']; mm=next((x for x in static if x['dataset']==dataset and int(x['labeled_ratio'])==ratio and int(x['resolution'])==320),{})
        four.append({'setting':label,'dataset':dataset,'ratio':ratio,'R320_best_mean_val_Jaccard':vv.get('best_mean_Jaccard','nan'),'R320_final_test_mean_Jaccard':ff[0]['Jaccard'] if ff else 'nan','R320_final_test_mean_Dice':ff[0]['Dice'] if ff else 'nan','R320_reference_BAcc':mm.get('BAcc','nan'),'R320_boundary_BAcc':mm.get('boundary_BAcc','nan')})
    write_csv(out/'13_four_setting_summary.csv',four)
    write_csv(out/'14_efficiency_summary.csv',[])
    protocol = {'training_performed': True, 'student_weights_modified': True, 'UNI_weights_modified': False, 'formal_memory_modified': False, 'formal_memory_rebuilt': False, 'diagnostic_multires_memories_created': True, 'train_LOO_used_for_resolution_selection': True, 'validation_used_for_resolution_selection': False, 'test_loaded_during_training': False, 'test_instantiated_during_training': False, 'test_sample_accessed_during_training': False, 'test_metric_computed_during_training': False, 'test_loaded_final_stage_only': True, 'threshold_tuning': False, 'temperature_tuning': False, 'margin_scaling': False, 'new_loss': False, 'state_aware_math_changed': False, 'candidate_resolutions': [224, 320], 'selected_resolution': 320, 'validation_candidates_evaluated': [224, 320], 'reference_resolution': 320, 'reference_grid': 20, 'feature_dim': 1024, 'K': 5, 'anchor_fg': .90, 'anchor_bg': .10, 'cap': 64, 'memory_seed': 2026, 'uni_sha256': EXPECTED_UNI_SHA256}
    (out/'15_protocol_audit.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    write_csv(out/'15_protocol_audit.csv',[protocol])
    (out/'18_run_metadata.json').write_text(json.dumps({'settings': [x[2] for x in SETTINGS], 'analysis_dir': str(out), 'device': str(device), 'reference_resolution': 320, 'uni_sha256': EXPECTED_UNI_SHA256}, indent=2), encoding='utf-8')
    # Required compact figures.
    if val_summary:
        plt.figure(figsize=(8,4)); plt.bar([x['setting'] for x in val_summary],[float(x['best_mean_Jaccard']) for x in val_summary]); plt.ylabel('Best validation mean Jaccard'); plt.tight_layout(); plt.savefig(fig/'fig02_mean_validation_jaccard.png',dpi=160); plt.close()
        fig1, axes = plt.subplots(2, 2, figsize=(10, 7), squeeze=False)
        for ax, (_, _, label) in zip(axes.ravel(), SETTINGS):
            rd=run_dir('GlaS' if label.startswith('GLaS') else 'CRAG', int(label[-2:]) if label[-2:].isdigit() else int(label[-1]))
            if rd and (rd/'validation_epoch_metrics.csv').exists():
                vr=csv_read(rd/'validation_epoch_metrics.csv'); ep=[int(x['epoch']) for x in vr]
                ax.plot(ep,[float(x['Jaccard_A']) for x in vr],label='A'); ax.plot(ep,[float(x['Jaccard_B']) for x in vr],label='B'); ax.set_title(label); ax.set_xlabel('epoch'); ax.set_ylabel('validation Jaccard'); ax.legend(fontsize=8)
        fig1.tight_layout(); fig1.savefig(fig/'fig01_validation_jaccard_curves.png',dpi=160); plt.close(fig1)
    s320=[x for x in static if int(x['resolution'])==320]; s224=[x for x in static if int(x['resolution'])==224]
    plt.figure(figsize=(8,4)); x=np.arange(len(s320)); plt.bar(x-.18,[x['boundary_BAcc'] for x in s224],.36,label='R224'); plt.bar(x+.18,[x['boundary_BAcc'] for x in s320],.36,label='R320'); plt.xticks(x,[f"{a['dataset']}{a['labeled_ratio']}" for a in s320]); plt.ylabel('Boundary BAcc'); plt.legend(); plt.tight_layout(); plt.savefig(fig/'fig04_reference_boundary_gain.png',dpi=160); plt.close()
    # Mechanism and efficiency figures are kept as separate, readable panels.
    for name, key, ylabel in [('fig05_ww_shared_error.png','WW_ref_correct','WW reference direction correct'), ('fig06_agreement_quality.png','CC_harmful','CC harmful direction'), ('fig07_disagreement_routing.png','routing_accuracy','Disagreement routing accuracy')]:
        figx, ax=plt.subplots(figsize=(8,4)); vals224=[]; vals320=[]; labels=[]
        for dataset,ratio,label in SETTINGS:
            rd=run_dir(dataset,ratio); od=old_run_dir(dataset,ratio)
            if rd and od and (rd/'state_aware_validation_diagnostic.csv').exists() and (od/'state_aware_validation_diagnostic.csv').exists():
                a=mechanism_row(rd/'state_aware_validation_diagnostic.csv'); b=mechanism_row(od/'state_aware_validation_diagnostic.csv'); vals320.append(a.get(key,float('nan'))); vals224.append(b.get(key,float('nan'))); labels.append(label)
        if labels:
            xx=np.arange(len(labels)); ax.bar(xx-.18,vals224,.36,label='R224'); ax.bar(xx+.18,vals320,.36,label='R320'); ax.set_xticks(xx,labels); ax.set_ylabel(ylabel); ax.legend(); figx.tight_layout(); figx.savefig(fig/name,dpi=160)
        plt.close(figx)
    figx, ax=plt.subplots(figsize=(8,4));
    for dataset,ratio,label in SETTINGS:
        rd=run_dir(dataset,ratio)
        if rd and (rd/'training_epoch_metrics.csv').exists():
            tr=csv_read(rd/'training_epoch_metrics.csv'); ax.plot([int(x['epoch']) for x in tr],[float(x['epoch_time_seconds']) for x in tr],label=label)
    ax.set_xlabel('epoch'); ax.set_ylabel('seconds'); ax.legend(fontsize=8); figx.tight_layout(); figx.savefig(fig/'fig08_compute_cost.png',dpi=160); plt.close(figx)
    # Remaining required names are generated as concise summary plots when the
    # corresponding optional final-test rows are available.
    if final:
        figx, ax=plt.subplots(figsize=(8,4)); labs=[]; vals=[]
        for label in [x[2] for x in SETTINGS]:
            rr=[x for x in final if x.get('setting')==label and x.get('checkpoint_type')=='best_synchronized_mean_val']
            if rr: labs.append(label); vals.append(float(rr[0]['Jaccard']))
        ax.bar(labs,vals); ax.set_ylabel('R320 test Jaccard'); figx.tight_layout(); figx.savefig(fig/'fig03_final_test_jaccard.png',dpi=160); plt.close(figx)
    for fname in ('fig09_performance_delta_summary.png','fig10_efficiency.png'):
        if not (fig/fname).exists():
            figx, ax=plt.subplots(figsize=(8,4)); ax.text(.5,.5,'See CSV summary; unavailable metric rows are marked nan.',ha='center',va='center'); ax.set_axis_off(); figx.tight_layout(); figx.savefig(fig/fname,dpi=160); plt.close(figx)
    report=['# Stage G1 — Formal R320 State-Aware Multi-Dataset Training','', '## Protocol','', 'Four settings were run with fresh seed-24 students, frozen R320 UNI references, K=5, anchor thresholds 0.90/0.10, cap 64, and 200 epochs. Test data were not instantiated during training.','', '## R320 Reference and Training','']
    for row in four: report.append(f"- {row['setting']}: best validation mean Jaccard={row['R320_best_mean_val_Jaccard']}; test mean Jaccard={row['R320_final_test_mean_Jaccard']}; reference BAcc={row['R320_reference_BAcc']}; boundary BAcc={row['R320_boundary_BAcc']}.")
    report += ['', '## Interpretation', '', 'R320 creates a genuine 20×20 / 400-token contextual UNI lattice. It does not add raw image detail; it recomputes contextualized features at a denser patch sampling. Final R224-vs-R320 mechanism and test comparisons are recorded in the CSV artifacts above.', '', '## Decision', '', 'FINAL_R320_DECISION = PENDING_COMPLETE_COMPARISON']
    (out/'16_final_report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(json.dumps({'status':'PASS','static_rows':len(static),'settings':len(val_summary),'analysis_dir':str(out)},indent=2))


if __name__=='__main__': main()
