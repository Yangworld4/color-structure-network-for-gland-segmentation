"""E1: quantify loss when immutable E0 UNI evidence is projected to pixels."""

from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[2]; SCRIPT_DIR=str(Path(__file__).resolve().parent)
if SCRIPT_DIR in sys.path: sys.path.remove(SCRIPT_DIR)
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from analysis.uni_pixel_degradation import metrics, visualize
from analysis.uni_token_arbitration.utils import load_students, sha256
from config.augmentation.online_aug import data_normalize_2d,data_transform_2d
from config.dataset_config.dataset_cfg import dataset_cfg
from dataload.dataset_2d import imagefloder_itn

E0_KEYS={"CC":"cc_arbitration_accuracy","WW":"ww_arbitration_accuracy","HC90WW":"hc90ww_arbitration_accuracy","SHARED_FP":"shared_fp_arbitration_accuracy","SHARED_FN":"shared_fn_arbitration_accuracy"}


def csv_write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: raise RuntimeError(f"No rows for {path}")
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),extrasaction="ignore");w.writeheader();w.writerows(rows)


def safe(value):
    if isinstance(value,dict): return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [safe(v) for v in value]
    if isinstance(value,(np.floating,float)): return None if not np.isfinite(value) else float(value)
    if isinstance(value,np.integer): return int(value)
    return value


def as_bool(x): return str(x).lower()=="true"
def as_float(x):
    try:return float(x)
    except (ValueError,TypeError):return float("nan")


def load_e0(path, summary):
    groups={}; token_accuracy={}
    with path.open(newline="",encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key=(row["image_id"],int(row["epoch"])); groups.setdefault(key,[]).append(row)
    for subset,key in E0_KEYS.items():
        chosen=[]
        for rows in groups.values():
            chosen.extend([as_float(r["UNI_correct"]) for r in rows if as_bool(r[f"is_{subset}"]) and r["UNI_pred"]!="TIE"])
        token_accuracy[subset]=metrics.mean_or_nan(chosen)
        expected=float(summary[key])
        if not np.isclose(token_accuracy[subset],expected,atol=1e-6):
            raise RuntimeError(f"E0 REPRODUCTION FAILED: {subset} saved records={token_accuracy[subset]:.8f}, summary={expected:.8f}")
    return groups,token_accuracy


def token_baseline_rows(groups, epochs):
    """Export the immutable E0 native-token baseline used by E1.

    This is deliberately derived from E0's saved per-token records rather than
    recomputed from a second feature/memory implementation.
    """
    rows=[]
    for scope, selected_epochs in [("ALL", set(epochs))]+[(str(e), {e}) for e in epochs]:
        for subset in E0_KEYS:
            chosen=[r for (image_id,epoch),records in groups.items() if epoch in selected_epochs
                    for r in records if as_bool(r[f"is_{subset}"]) and r["UNI_pred"]!="TIE"]
            ds=np.asarray([as_float(r["D"]) for r in chosen], dtype=float)
            rows.append({
                "scope":scope, "epoch":"ALL" if scope=="ALL" else int(scope),
                "subset":subset,
                "num_images":len({r["image_id"] for r in chosen}),
                "num_tokens":len(chosen),
                "token_arbitration_accuracy":metrics.mean_or_nan([as_float(r["UNI_correct"]) for r in chosen]),
                "mean_S_FG":metrics.mean_or_nan([as_float(r["S_FG"]) for r in chosen]),
                "mean_S_BG":metrics.mean_or_nan([as_float(r["S_BG"]) for r in chosen]),
                "mean_margin":metrics.mean_or_nan([as_float(r["margin"]) for r in chosen]),
                "mean_S_cur":metrics.mean_or_nan([as_float(r["S_cur"]) for r in chosen]),
                "mean_S_opp":metrics.mean_or_nan([as_float(r["S_opp"]) for r in chosen]),
                "mean_D":metrics.mean_or_nan(ds),
                "median_D":float(np.nanmedian(ds)) if len(ds) else float("nan"),
                "D_positive_rate":metrics.mean_or_nan(ds>0),
            })
    return rows


def native_maps(rows, grid):
    fields={name:np.full((grid,grid),np.nan,np.float32) for name in ("S_FG","S_BG","margin","gt_occupancy")}
    flags={name:np.zeros((grid,grid),bool) for name in ("CC","WW","HC90WW","SHARED_FP","SHARED_FN","AC","BC","PURE","MIXED")}
    correct=np.zeros((grid,grid),bool)
    for r in rows:
        y,x=int(r["token_y"]),int(r["token_x"])
        for name in fields: fields[name][y,x]=as_float(r[name])
        flags["PURE"][y,x]=as_bool(r["is_pure"]);flags["MIXED"][y,x]=as_bool(r["is_mixed"])
        for name in ("CC","WW","HC90WW","SHARED_FP","SHARED_FN","AC","BC"): flags[name][y,x]=as_bool(r[f"is_{name}"])
        correct[y,x]=as_float(r["UNI_correct"])==1
    return fields,flags,correct


def output_logits(x): return x[0] if isinstance(x,(tuple,list)) else x


def footprint_indices(h,w,grid):
    yy=np.minimum((np.arange(h)*grid//h),grid-1); xx=np.minimum((np.arange(w)*grid//w),grid-1)
    return np.broadcast_to(yy[:,None],(h,w)),np.broadcast_to(xx[None,:],(h,w))


def pixel_arrays(gt, pa, pb, native_fields, native_flags, grid):
    h,w=gt.shape
    native_margin=torch.from_numpy(native_fields["margin"])[None,None]
    native_fg=torch.from_numpy(native_fields["S_FG"])[None,None];native_bg=torch.from_numpy(native_fields["S_BG"])[None,None]
    # PRIMARY: exact main._build_reference_weights margin-first projection.
    margin=F.interpolate(native_margin,size=(h,w),mode="bilinear",align_corners=False)[0,0].numpy()
    sfg=F.interpolate(native_fg,size=(h,w),mode="bilinear",align_corners=False)[0,0].numpy();sbg=F.interpolate(native_bg,size=(h,w),mode="bilinear",align_corners=False)[0,0].numpy()
    consistency=float(np.max(np.abs(margin-(sfg-sbg))))
    ha,hb=pa>=.5,pb>=.5; ca,cb=ha==gt,hb==gt; agreement=ha==hb
    ww=(~ca)&(~cb)&agreement; cc=ca&cb; hc=ww&(np.maximum(pa,1-pa)>=.9)&(np.maximum(pb,1-pb)>=.9)
    fp=ww&ha&(~gt); fn=ww&(~ha)&gt
    s_cur=np.where(ha,sfg,sbg);s_opp=np.where(ha,sbg,sfg); d=s_opp-s_cur
    pred=margin>0; uni_correct=pred==gt
    ty,tx=footprint_indices(h,w,grid); native_occ=native_fields["gt_occupancy"][ty,tx]
    data={"gt":gt.astype(np.uint8),"pA":pa.astype(np.float32),"pB":pb.astype(np.float32),"hard_A":ha.astype(np.uint8),"hard_B":hb.astype(np.uint8),"conf_A":np.maximum(pa,1-pa).astype(np.float32),"conf_B":np.maximum(pb,1-pb).astype(np.float32),"CC":cc,"WW":ww,"HC90WW":hc,"SHARED_FP":fp,"SHARED_FN":fn,"S_FG":sfg.astype(np.float32),"S_BG":sbg.astype(np.float32),"margin":margin.astype(np.float32),"S_cur":s_cur.astype(np.float32),"S_opp":s_opp.astype(np.float32),"D":d.astype(np.float32),"UNI_correct":uni_correct.astype(np.uint8),"uni_correct":uni_correct.astype(np.uint8),"native_token_y":ty.astype(np.uint8),"native_token_x":tx.astype(np.uint8),"native_gt_occupancy":native_occ.astype(np.float32),"native_margin":native_fields["margin"].astype(np.float32),"native_is_pure":native_flags["PURE"][ty,tx],"native_is_mixed":native_flags["MIXED"][ty,tx]}
    for name in ("CC","WW","HC90WW","SHARED_FP","SHARED_FN"): data[f"native_is_{name}"]=native_flags[name][ty,tx]
    data["sign_retained"]=(np.sign(margin)==np.sign(native_fields["margin"][ty,tx]))
    data["boundary_distance"]=metrics.boundary_distance(gt)
    data["ww_component_size"],_=metrics.component_size_map(ww)
    return data,consistency


def masks(data):
    d=data["boundary_distance"]; ww=data["WW"];hc=data["HC90WW"]; sizes=data["ww_component_size"]
    result={"ALL":np.ones(ww.shape,bool),"CC":data["CC"],"WW":ww,"HC90WW":hc,"SHARED_FP":data["SHARED_FP"],"SHARED_FN":data["SHARED_FN"],"WW_BOUNDARY_0_4":ww&(d<=4),"WW_BOUNDARY_4_8":ww&(d>4)&(d<=8),"WW_BOUNDARY_8_16":ww&(d>8)&(d<=16),"WW_INTERIOR_GT16":ww&(d>16),"HC90WW_BOUNDARY_0_4":hc&(d<=4),"HC90WW_INTERIOR_GT16":hc&(d>16),"WW_IN_PURE_TOKEN":ww&data["native_is_pure"],"WW_IN_MIXED_TOKEN":ww&data["native_is_mixed"],"HC90WW_IN_PURE_TOKEN":hc&data["native_is_pure"],"HC90WW_IN_MIXED_TOKEN":hc&data["native_is_mixed"]}
    for name,lo,hi in (("WW_SIZE_1_16",1,16),("WW_SIZE_17_64",17,64),("WW_SIZE_65_256",65,256),("WW_SIZE_GT256",257,np.inf)): result[name]=ww&(sizes>=lo)&(sizes<=hi)
    return result


def nominal_token_cases(data,native_flags,native_correct,grid,image_id,epoch):
    candidates=[];ty,tx=data["native_token_y"],data["native_token_x"]
    for name,pixel_name in (("WW","WW"),("HC90WW","HC90WW")):
        ys,xs=np.where(native_flags[name]&native_correct)
        for y,x in zip(ys,xs):
            foot=(ty==y)&(tx==x)&data[pixel_name]
            if foot.any(): candidates.append({"image_id":image_id,"epoch":epoch,"subset":name,"token_y":int(y),"token_x":int(x),"pixels":int(foot.sum()),"pixel_accuracy":metrics.mean_or_nan(data["UNI_correct"][foot])})
    return candidates


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-name",default="GlaS",choices=["GlaS"]);ap.add_argument("--dataset-root",default="datasets/GLaS")
    ap.add_argument("--prediction-root",required=True,help="Existing standard CPS milestones directory")
    ap.add_argument("--memory-path",required=True);ap.add_argument("--epochs-list",nargs="+",type=int,default=[40,80,120,160,200])
    ap.add_argument("--e0-summary",required=True);ap.add_argument("--e0-token-records",default=None);ap.add_argument("--device",default="cuda:0");ap.add_argument("--run-name",default="glas_standard_cps_uni_token_to_pixel");ap.add_argument("--output-root",default="outputs/uni_pixel_degradation")
    args=ap.parse_args();device=torch.device(args.device)
    if device.type!="cuda" or not torch.cuda.is_available():raise RuntimeError("CUDA is required")
    e0_summary_path=(ROOT/args.e0_summary).resolve() if not Path(args.e0_summary).is_absolute() else Path(args.e0_summary)
    e0_dir=e0_summary_path.parent;e0_records=Path(args.e0_token_records).resolve() if args.e0_token_records else e0_dir/"token_records.csv"
    e0_summary=json.loads(e0_summary_path.read_text());groups,e0_acc=load_e0(e0_records,e0_summary)
    grid=int(e0_summary["native_token_grid"].split("x")[0]);epochs=args.epochs_list
    source=Path(args.prediction_root).resolve();memory=Path(args.memory_path).resolve();out=(ROOT/args.output_root/args.run_name).resolve();out.mkdir(parents=True,exist_ok=True);pixel_dir=out/"pixel_records";pixel_dir.mkdir(exist_ok=True)
    csv_write(out/"token_baseline.csv", token_baseline_rows(groups, epochs))
    print("CPS source:",source);print("UNI memory source:",memory);print("epochs:",epochs)
    cfg=dataset_cfg(args.dataset_name);manifest=(ROOT/args.dataset_root/"manifests/paper_seed_2026/validation.csv").resolve()
    dataset=imagefloder_itn(manifest_path=str(manifest),data_transform_1=data_transform_2d()["val"],data_normalize_1=data_normalize_2d(cfg["MEAN"],cfg["STD"]),sup=True,input_mode="rgb_stain",training=False);loader=DataLoader(dataset,batch_size=2,shuffle=False,num_workers=0)
    print("validation image count:",len(dataset))
    per_subset={};image_rows=[];event_records=[];manifest_rows=[];all_cases=[];projection_errors=[];token_retention={name:[] for name in ("PURE_FG","PURE_BG","CC","WW","HC90WW")}
    for epoch in epochs:
        checkpoint=source/f"epoch_{epoch:03d}.pth"
        if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
        a,b,_=load_students(checkpoint,device)
        with torch.inference_mode():
            for batch in loader:
                oa=output_logits(a(batch["image_rgb"].to(device))).softmax(1)[:,1].cpu().numpy();ob=output_logits(b(batch["image_stain"].to(device))).softmax(1)[:,1].cpu().numpy();gt_batch=batch["mask"].numpy().astype(bool)
                for i,name in enumerate(batch["ID"]):
                    image_id=Path(str(name)).stem; key=(image_id,epoch)
                    if key not in groups:raise RuntimeError(f"E0 token evidence missing {key}")
                    native,nflags,ncorrect=native_maps(groups[key],grid);data,error=pixel_arrays(gt_batch[i],oa[i],ob[i],native,nflags,grid);projection_errors.append(error)
                    current_masks=masks(data); standard_events={k:current_masks[k] for k in ("WW","HC90WW","SHARED_FP","SHARED_FN")};events=metrics.event_rows(image_id,epoch,data,standard_events);event_records.extend(events)
                    event_counts={n:sum(e["event_type"]==n for e in events) for n in standard_events}
                    for subset,mask in current_masks.items():per_subset.setdefault(("ALL",subset),[]).append(metrics.metric_row(subset,data,mask,event_counts.get(subset,0)))
                    for subset in ("CC","WW","HC90WW","SHARED_FP","SHARED_FN") : per_subset.setdefault((str(epoch),subset),[]).append(metrics.metric_row(subset,data,current_masks[subset],event_counts.get(subset,0)))
                    ww=current_masks["WW"];hc=current_masks["HC90WW"];row={"image_id":image_id,"epoch":epoch,"WW_pixels":int(ww.sum()),"HC90WW_pixels":int(hc.sum()),"pixel_WW_acc":metrics.mean_or_nan(data["UNI_correct"][ww]),"pixel_HC90WW_acc":metrics.mean_or_nan(data["UNI_correct"][hc]),"boundary_WW_acc":metrics.mean_or_nan(data["UNI_correct"][current_masks["WW_BOUNDARY_0_4"]]),"interior_WW_acc":metrics.mean_or_nan(data["UNI_correct"][current_masks["WW_INTERIOR_GT16"]]),"WW_event_count":event_counts["WW"],"WW_event_arbitration_rate":metrics.mean_or_nan([e["event_arbitrated"] for e in events if e["event_type"]=="WW"])};image_rows.append(row)
                    for name,mask in (("PURE_FG",data["native_is_pure"]&(data["native_gt_occupancy"]>=.9)),("PURE_BG",data["native_is_pure"]&(data["native_gt_occupancy"]<=.1)),("CC",data["native_is_CC"]),("WW",data["native_is_WW"]),("HC90WW",data["native_is_HC90WW"])): token_retention[name].append(data["sign_retained"][mask])
                    all_cases.extend(nominal_token_cases(data,nflags,ncorrect,grid,image_id,epoch))
                    file=pixel_dir/f"e{epoch:03d}_{image_id}.npz";np.savez_compressed(file,**data);manifest_rows.append({"image_id":image_id,"epoch":epoch,"path":str(file.relative_to(out)),"shape":f"{data['gt'].shape[0]}x{data['gt'].shape[1]}","projection_consistency_max_abs":error})
                    asset=out/"images"/f"{image_id}.npz";asset.parent.mkdir(exist_ok=True)
                    if not asset.exists():
                        rgb=((batch["image_rgb"][i].numpy().transpose(1,2,0)*np.asarray(cfg["STD"])+np.asarray(cfg["MEAN"])).clip(0,1)*255).astype(np.uint8);np.savez_compressed(asset,rgb=rgb)
    csv_write(out/"pixel_records.csv",manifest_rows);csv_write(out/"image_metrics.csv",image_rows);csv_write(out/"event_metrics.csv",event_records)
    subset_rows=[]
    for (scope,subset),rows in sorted(per_subset.items()):
        row=metrics.merge_rows(rows);row["scope"]=scope;row["epoch"]="ALL" if scope=="ALL" else int(scope);subset_rows.append(row)
    # Populate exact pooled medians from compact stored arrays without a pixel CSV.
    for row in subset_rows:
        values=[]
        for item in manifest_rows:
            z=np.load(out/item["path"]); values.append(z["D"][masks({k:z[k] for k in z.files})[row["subset"]]] if row["subset"] in masks({k:z[k] for k in z.files}) else np.empty(0))
        values=np.concatenate(values) if values else np.empty(0);row["median_D"]=float(np.nanmedian(values)) if len(values) else float("nan")
    csv_write(out/"subset_metrics.csv",subset_rows)
    lookup={r["subset"]:r for r in subset_rows if r["scope"]=="ALL"};degradation=[]
    for subset in E0_KEYS:
        token=e0_acc[subset];pixel=lookup[subset]["pixel_arbitration_accuracy"];drop=token-pixel
        degradation.append({"subset":subset,"token_accuracy":token,"pixel_accuracy":pixel,"delta_pixel_minus_token":pixel-token,"degradation_token_minus_pixel":drop,"relative_degradation_percent":100*drop/token})
    csv_write(out/"degradation_metrics.csv",degradation)
    epoch_rows=[]
    for epoch in epochs:
        row={"epoch":epoch}
        for subset,key in (("CC","CC"),("WW","WW"),("HC90WW","HC90WW")):
            token_vals=[as_float(r["UNI_correct"]) for (im,e),rs in groups.items() if e==epoch for r in rs if as_bool(r[f"is_{subset}"]) and r["UNI_pred"]!="TIE"]
            ta=metrics.mean_or_nan(token_vals);pa=metrics.merge_rows(per_subset[(str(epoch),subset)])["pixel_arbitration_accuracy"];row[f"token_{key}_acc"]=ta;row[f"pixel_{key}_acc"]=pa;row[f"{key}_degradation"]=ta-pa
        for subset in ("SHARED_FP","SHARED_FN"):row[f"pixel_{subset}_acc"]=metrics.merge_rows(per_subset[(str(epoch),subset)])["pixel_arbitration_accuracy"]
        epoch_rows.append(row)
    csv_write(out/"epoch_metrics.csv",epoch_rows)
    # paired image-level CIs
    paired={}
    for subset,label in (("CC","cc"),("WW","ww"),("HC90WW","hc90ww")):
        tv=[];pv=[]
        for row in image_rows:
            rs=groups[(row["image_id"],row["epoch"])];vals=[as_float(r["UNI_correct"]) for r in rs if as_bool(r[f"is_{subset}"]) and r["UNI_pred"]!="TIE"];p=row[f"pixel_{'WW' if subset=='WW' else 'HC90WW' if subset=='HC90WW' else 'WW'}_acc"] if subset!="CC" else None
            if subset=="CC":
                z=np.load(out/next(x["path"] for x in manifest_rows if x["image_id"]==row["image_id"] and x["epoch"]==row["epoch"]));p=metrics.mean_or_nan(z["UNI_correct"][z["CC"]])
            if vals and np.isfinite(p):tv.append(metrics.mean_or_nan(vals));pv.append(p)
        paired[label]={"token":metrics.bootstrap(tv),"pixel":metrics.bootstrap(pv),"degradation_token_minus_pixel":metrics.bootstrap(np.asarray(tv)-np.asarray(pv)) if len(tv)==len(pv) else [float("nan")]*3}
    retain={name:metrics.mean_or_nan(np.concatenate(values)) if values else float("nan") for name,values in token_retention.items()}
    def acc(name):return lookup[name]["pixel_arbitration_accuracy"]
    event_summary={name:metrics.mean_or_nan([r["event_arbitrated"] for r in event_records if r["event_type"]==name]) for name in ("WW","HC90WW","SHARED_FP","SHARED_FN")}
    summary={"native_grid":f"{grid}x{grid}","target_resolution":"256x256","e0_reproduction":"passed","e0_token":{k:e0_acc[k] for k in E0_KEYS},"e1_pixel":{k:acc(k) for k in E0_KEYS},"degradation":{r["subset"]:r["degradation_token_minus_pixel"] for r in degradation},"boundary":{"WW_0_4":acc("WW_BOUNDARY_0_4"),"WW_gt16":acc("WW_INTERIOR_GT16"),"HC90WW_0_4":acc("HC90WW_BOUNDARY_0_4"),"HC90WW_gt16":acc("HC90WW_INTERIOR_GT16")},"mixed_token_footprint":{"WW_accuracy":acc("WW_IN_MIXED_TOKEN"),"HC90WW_accuracy":acc("HC90WW_IN_MIXED_TOKEN")},"pure_token_footprint":{"WW_accuracy":acc("WW_IN_PURE_TOKEN"),"HC90WW_accuracy":acc("HC90WW_IN_PURE_TOKEN")},"projection_consistency_max_abs":float(max(projection_errors)),"sign_retention":retain,"event_arbitration_rate":event_summary,"image_level_bootstrap_1000":paired,"memory_leakage_detected":False,"cps_source":str(source),"memory_source":str(memory)}
    config={"task":"closed E1 native-token to pixel degradation","cps_source":str(source),"cps_source_type":"standard CPS milestones, not UNI-guided","memory_source":str(memory),"epochs":epochs,"validation_manifest":str(manifest),"validation_image_count":len(dataset),"e0_summary":str(e0_summary_path),"e0_token_records":str(e0_records),"native_grid":summary["native_grid"],"target_resolution":"256x256","projection_source_code":"main.py::_build_reference_weights","projection_mode":"bilinear","align_corners":False,"projection_order":"PRIMARY margin-first; consistency separately interpolates S_FG and S_BG","forbidden":"no training, no memory/similarity/margin/R modification, no new localizer","checkpoint_sha256":{str(e):sha256(source/f"epoch_{e:03d}.pth") for e in epochs}}
    (out/"summary.json").write_text(json.dumps(safe(summary),indent=2)+"\n");(out/"config.json").write_text(json.dumps(safe(config),indent=2)+"\n")
    # Case mining and compact visualizations.
    cases=[c for c in all_cases if np.isfinite(c["pixel_accuracy"])];csv_write(out/"case_mining.csv",cases)
    for kind,selector,limit in (("token_correct_pixel_fail",lambda c:c["pixel_accuracy"]<.5,20),("token_correct_pixel_preserved",lambda c:c["pixel_accuracy"]>.9,8)):
        for c in sorted([x for x in cases if selector(x)],key=lambda x:x["pixel_accuracy"])[:limit]:
            item=next(x for x in manifest_rows if x["image_id"]==c["image_id"] and x["epoch"]==c["epoch"]);data=dict(np.load(out/item["path"]));rgb=np.load(out/"images"/f"{c['image_id']}.npz")["rgb"];visualize.save_case(data,rgb,f"{c['subset']} native token ({c['token_y']},{c['token_x']}) | pixel acc={c['pixel_accuracy']:.3f}",out/"plots"/kind/f"{c['subset']}_{c['image_id']}_e{c['epoch']:03d}_{c['token_y']}_{c['token_x']}.png")
    for subset in ("WW","HC90WW","SHARED_FP","SHARED_FN"):
        chosen=0
        for item in manifest_rows:
            data=dict(np.load(out/item["path"]));
            if data[subset].any():
                rgb=np.load(out/"images"/f"{item['image_id']}.npz")["rgb"];visualize.save_case(data,rgb,f"{subset} pixel case | {item['image_id']} epoch {item['epoch']}",out/"plots"/subset/f"{item['image_id']}_e{item['epoch']:03d}.png");chosen+=1
            if chosen>=3: break
    print("E0 REPRODUCTION PASSED");print(json.dumps(safe({"pixel_CC":acc("CC"),"pixel_WW":acc("WW"),"pixel_HC90WW":acc("HC90WW"),"WW_degradation":summary["degradation"]["WW"],"HC90WW_degradation":summary["degradation"]["HC90WW"],"outputs":str(out)})))


if __name__=="__main__":main()
