"""E4: offline feasibility of UNI-256 native-scale CPS arbitration."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from analysis.native_scale_reference import metrics, visualize
from analysis.uni_resolution_diagnostic.run_diagnostic import direct_uni_tokens
from analysis.uni_token_arbitration.utils import load_memory, score_native, sha256
from config.augmentation.online_aug import data_normalize_2d, data_transform_2d
from config.dataset_config.dataset_cfg import dataset_cfg
from dataload.dataset_2d import imagefloder_itn
from models.uni import load_uni
import main as project_main

UNI_WEIGHTS = "/home/yangorange/.cache/huggingface/hub/models--MahmoodLab--UNI/snapshots/b55a5ec6cade1a39edfe6534189a9b8ca7a022f0/pytorch_model.bin"
UNI_CONFIG = str(Path(UNI_WEIGHTS).with_name("config.json"))
UNI_SHA = "56ef09b44a25dc5c7eedc55551b3d47bcd17659a7a33837cf9abc9ec4e2ffb40"
EVENT_TYPES = ("WW", "HC90WW", "SHARED_FP", "SHARED_FN")
SIZE_BINS = (("1_16", 1, 16), ("17_64", 17, 64), ("65_256", 65, 256), ("GT256", 257, np.inf))
DEAD_ZONE = .05


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: raise RuntimeError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def safe(v):
    if isinstance(v, dict): return {str(k): safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [safe(x) for x in v]
    if isinstance(v, (np.floating, float)): return None if not np.isfinite(v) else float(v)
    if isinstance(v, np.integer): return int(v)
    return v


def manifest_ids(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {Path(row["image"]).stem for row in csv.DictReader(handle)}


def native_fields(loader, uni, mem_fg, mem_bg, device, cfg):
    fields = {}
    with torch.inference_mode():
        for batch in loader:
            rgb = (batch["image_rgb"].to(device) * torch.tensor(cfg["STD"], device=device).view(1,3,1,1) + torch.tensor(cfg["MEAN"], device=device).view(1,3,1,1)).clamp(0,1)
            tokens = direct_uni_tokens(uni, rgb, 256)
            sfg, sbg, margin = score_native(tokens, mem_fg, mem_bg, 5, 16)
            occupancy = F.adaptive_avg_pool2d(batch["mask"].to(device).float()[:,None], (16,16))[:,0]
            for i, name in enumerate(batch["ID"]):
                fields[Path(str(name)).stem] = {"S_FG":sfg[i].cpu().numpy(), "S_BG":sbg[i].cpu().numpy(), "margin":margin[i].cpu().numpy(), "occupancy":occupancy[i].cpu().numpy()}
    return fields


def token_footprint_arrays(z, grid=16):
    h,w = z["gt"].shape
    ty = np.minimum(np.arange(h) * grid // h, grid-1)
    tx = np.minimum(np.arange(w) * grid // w, grid-1)
    return ty[:,None], tx[None,:]


def token_metrics(z, fields):
    gt = z["gt"].astype(bool); pa=z["pA"].astype(float); pb=z["pB"].astype(float)
    pa_t=F.adaptive_avg_pool2d(torch.from_numpy(pa)[None,None],(16,16))[0,0].numpy()
    pb_t=F.adaptive_avg_pool2d(torch.from_numpy(pb)[None,None],(16,16))[0,0].numpy()
    occ=fields["occupancy"]; ha=pa_t>=.5; hb=pb_t>=.5; agreement=ha==hb; consensus=ha
    conf_a=np.maximum(pa_t,1-pa_t); conf_b=np.maximum(pb_t,1-pb_t)
    sfg=fields["S_FG"]; sbg=fields["S_BG"]; margin=fields["margin"]
    uni_pred=np.where(sfg>sbg,1,np.where(sbg>sfg,0,-1)); ties=uni_pred==-1
    contradiction=agreement & (~ties) & (uni_pred!=consensus)
    d=np.where(agreement, np.where(consensus,sbg-sfg,sfg-sbg), np.nan)
    p_c=(pa_t+pb_t)/2
    ydir=np.where(occ-p_c>DEAD_ZONE,1,np.where(occ-p_c < -DEAD_ZONE,-1,0))
    udir=np.where(uni_pred==1,1,np.where(uni_pred==0,-1,0))
    direction_correct=np.where((agreement & (~ties) & (ydir!=0)), udir==ydir, np.nan)
    ty,tx=token_footprint_arrays(z)
    def ratio(mask):
        out=np.zeros((16,16),float)
        for y in range(16):
            for x in range(16): out[y,x]=float(mask[y*16:(y+1)*16,x*16:(x+1)*16].mean())
        return out
    ww_ratio=ratio(z["WW"].astype(bool)); hc_ratio=ratio(z["HC90WW"].astype(bool)); cc_ratio=ratio(z["CC"].astype(bool)); fp_ratio=ratio(z["SHARED_FP"].astype(bool)); fn_ratio=ratio(z["SHARED_FN"].astype(bool))
    pure=(occ>=.9)|(occ<=.1)
    return {"pA_token":pa_t,"pB_token":pb_t,"consensus_prob":p_c,"hard_A":ha,"hard_B":hb,"conf_A":conf_a,"conf_B":conf_b,"agreement":agreement,"S_FG":sfg,"S_BG":sbg,"margin":margin,"S_cur":np.where(consensus,sfg,sbg),"S_opp":np.where(consensus,sbg,sfg),"D":d,"abs_D":np.abs(d),"UNI_pred":uni_pred,"contradiction":contradiction,"hc_contradiction":contradiction&(conf_a>=.9)&(conf_b>=.9),"occupancy":occ,"pure":pure,"WW_ratio":ww_ratio,"HC90WW_ratio":hc_ratio,"CC_ratio":cc_ratio,"shared_FP_ratio":fp_ratio,"shared_FN_ratio":fn_ratio,"gt_direction":ydir,"uni_direction":udir,"direction_correct":direction_correct}


def token_rows(image_id, epoch, tm):
    rows=[]
    qvals=tm["D"][tm["contradiction"] & np.isfinite(tm["D"])]
    q=np.quantile(qvals,[.25,.5,.75]) if len(qvals) else np.asarray([np.nan]*3)
    for y in range(16):
        for x in range(16):
            d=float(tm["D"][y,x]); qlabel="" if not np.isfinite(d) or not tm["contradiction"][y,x] else f"Q{1+int(d>q[0])+int(d>q[1])+int(d>q[2])}"
            direction=tm["direction_correct"][y,x]
            rows.append({"image_id":image_id,"epoch":epoch,"token_y":y,"token_x":x,"pA_token":tm["pA_token"][y,x],"pB_token":tm["pB_token"][y,x],"consensus_prob":tm["consensus_prob"][y,x],"hard_A":int(tm["hard_A"][y,x]),"hard_B":int(tm["hard_B"][y,x]),"conf_A":tm["conf_A"][y,x],"conf_B":tm["conf_B"][y,x],"agreement":bool(tm["agreement"][y,x]),"S_FG":tm["S_FG"][y,x],"S_BG":tm["S_BG"][y,x],"margin":tm["margin"][y,x],"S_cur":tm["S_cur"][y,x] if tm["agreement"][y,x] else np.nan,"S_opp":tm["S_opp"][y,x] if tm["agreement"][y,x] else np.nan,"D":d,"abs_D":abs(d) if np.isfinite(d) else np.nan,"UNI_pred":"FG" if tm["UNI_pred"][y,x]==1 else "BG" if tm["UNI_pred"][y,x]==0 else "TIE","contradiction":bool(tm["contradiction"][y,x]),"hc_contradiction":bool(tm["hc_contradiction"][y,x]),"gt_occupancy":tm["occupancy"][y,x],"gt_pure_or_mixed":"PURE" if tm["pure"][y,x] else "MIXED","WW_ratio":tm["WW_ratio"][y,x],"HC90WW_ratio":tm["HC90WW_ratio"][y,x],"CC_ratio":tm["CC_ratio"][y,x],"shared_FP_ratio":tm["shared_FP_ratio"][y,x],"shared_FN_ratio":tm["shared_FN_ratio"][y,x],"gt_direction":"FG" if tm["gt_direction"][y,x]==1 else "BG" if tm["gt_direction"][y,x]==-1 else "STABLE","uni_direction":"FG" if tm["uni_direction"][y,x]==1 else "BG" if tm["uni_direction"][y,x]==-1 else "TIE","direction_correct":direction,"selectivity":(tm["WW_ratio"][y,x]+tm["HC90WW_ratio"][y,x])/(tm["CC_ratio"][y,x]+1e-6),"D_quartile":qlabel})
    return rows


def event_rows(z, tm, image_id, epoch):
    # Contradiction is only a 16x16 signal; expand the token footprint solely
    # to evaluate how much of each fixed pixel event it covers.
    ty,tx=token_footprint_arrays(z); cmap=tm["contradiction"][ty,tx]
    rows=[]
    for typ in EVENT_TYPES:
        labels,n=ndimage.label(z[typ].astype(bool),structure=np.ones((3,3),bool))
        for idx in range(1,n+1):
            e=labels==idx; cov=float(cmap[e].mean())
            rows.append({"image_id":image_id,"epoch":epoch,"event_type":typ,"event_index":idx,"area_pixels":int(e.sum()),"coverage":cov,"covered":bool(cov>.5)})
    return rows,cmap


def metric_stats(rows, subset):
    chosen=[r for r in rows if subset(r)]
    def m(k): return metrics.mean([r[k] for r in chosen])
    mw=m("WW_ratio"); mh=m("HC90WW_ratio"); mc=m("CC_ratio")
    return {"subset":subset.__name__,"num_tokens":len(chosen),"num_images":len({r["image_id"] for r in chosen}),"mean_WW_ratio":mw,"mean_HC90WW_ratio":mh,"mean_CC_ratio":mc,"mean_shared_FP_ratio":m("shared_FP_ratio"),"mean_shared_FN_ratio":m("shared_FN_ratio"),"direction_accuracy":m("direction_correct"),"mean_D":m("D"),"mean_selectivity":(mw+mh)/(mc+1e-6)}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root",default="datasets/GLaS"); ap.add_argument("--prediction-root",required=True); ap.add_argument("--memory-path",default="checkpoints/glas_labeled10_seed24/reference_memory_uni256.npz"); ap.add_argument("--e1-root",default="outputs/uni_pixel_degradation/glas_standard_cps_uni_token_to_pixel"); ap.add_argument("--epochs-list",nargs="+",type=int,default=[40,80,120,160,200]); ap.add_argument("--run-name",default="glas_standard_cps_uni256"); ap.add_argument("--output-root",default="outputs/native_scale_reference"); ap.add_argument("--device",default="cuda:0"); ap.add_argument("--uni-weights",default=UNI_WEIGHTS); ap.add_argument("--uni-config",default=UNI_CONFIG)
    args=ap.parse_args(); device=torch.device(args.device)
    if device.type!="cuda" or not torch.cuda.is_available(): raise RuntimeError("E4 requires CUDA")
    out=(ROOT/args.output_root/args.run_name).resolve(); out.mkdir(parents=True,exist_ok=True); (out/"plots").mkdir(exist_ok=True)
    e1=(ROOT/args.e1_root).resolve(); memory=(ROOT/args.memory_path).resolve(); dataset_root=(ROOT/args.dataset_root).resolve(); manifest_dir=dataset_root/"manifests/paper_seed_2026"; val_manifest=manifest_dir/"validation.csv"; source_manifest=manifest_dir/"labeled_10.csv"
    leakage=manifest_ids(val_manifest)&manifest_ids(source_manifest)
    if leakage: raise RuntimeError("ERROR: memory leakage: "+", ".join(sorted(leakage)))
    manifests=list(csv.DictReader((e1/"pixel_records.csv").open(encoding="utf-8")))
    if len(manifests)!=85: raise RuntimeError("Expected 17 validation images x 5 epochs from E1")
    if not memory.is_file(): raise FileNotFoundError(memory)
    with np.load(memory,allow_pickle=False) as m:
        if int(m["token_grid"])!=16 or int(m["feature_dim"])!=1024 or int(m["reference_resolution"])!=256: raise RuntimeError("UNI256 memory metadata mismatch")
    cfg=dataset_cfg("GlaS"); dataset=imagefloder_itn(manifest_path=str(val_manifest),data_transform_1=data_transform_2d()["val"],data_normalize_1=data_normalize_2d(cfg["MEAN"],cfg["STD"]),sup=True,input_mode="rgb_stain",training=False); loader=DataLoader(dataset,batch_size=2,shuffle=False,num_workers=0)
    uni,_,_=load_uni(args.uni_weights,args.uni_config,expected_sha256=UNI_SHA); mem_fg,mem_bg,mem_meta=load_memory(memory,device); fields=native_fields(loader,uni,mem_fg,mem_bg,device,cfg)
    print("UNI input: 256x256"); print("Native grid: 16x16"); print("Images / epochs: 17 / 5"); print("Memory leakage: false")
    records=[]; events=[]; image_rows=[]; raw_cases=[]; grouped={}
    for item in manifests:
        image_id=item["image_id"]; epoch=int(item["epoch"]); z=dict(np.load(e1/item["path"])); tm=token_metrics(z,fields[image_id]); records.extend(token_rows(image_id,epoch,tm)); ev,cmap=event_rows(z,tm,image_id,epoch); events.extend(ev); grouped[(image_id,epoch)]={"tm":tm,"z":z,"cmap":cmap}
        contra=tm["contradiction"]; hc=tm["hc_contradiction"]; ty,tx=token_footprint_arrays(z)
        image_rows.append({"image_id":image_id,"epoch":epoch,"num_agreement_tokens":int(tm["agreement"].sum()),"num_contradiction_tokens":int(contra.sum()),"num_hc_contradiction_tokens":int(hc.sum()),"mean_WW_ratio_contradiction":metrics.mean(tm["WW_ratio"][contra]),"mean_CC_ratio_contradiction":metrics.mean(tm["CC_ratio"][contra]),"WW_coverage":float((z["WW"].astype(bool)&cmap).sum()/max(1,z["WW"].sum())),"HC90WW_coverage":float((z["HC90WW"].astype(bool)&cmap).sum()/max(1,z["HC90WW"].sum())),"direction_accuracy":metrics.mean(tm["direction_correct"][contra]),"WW_event_coverage":metrics.mean([x["covered"] for x in ev if x["event_type"]=="WW"]),"HC90WW_event_coverage":metrics.mean([x["covered"] for x in ev if x["event_type"]=="HC90WW"])})
    # D quartiles are descriptive only and use D, never GT-derived values.
    dvals=np.asarray([float(r["D"]) for r in records if bool(r["contradiction"]) and np.isfinite(float(r["D"]))]); q=np.quantile(dvals,[.25,.5,.75]) if len(dvals) else np.asarray([np.nan]*3)
    for r in records:
        d=float(r["D"]); r["D_quartile"]="" if not np.isfinite(d) or not r["contradiction"] else f"D_Q{1+int(d>q[0])+int(d>q[1])+int(d>q[2])}"
    def is_sub(name,r):
        if name=="ALL_AGREEMENT": return r["agreement"] and r["UNI_pred"]!="TIE"
        if name=="UNI_CPS_AGREE": return r["agreement"] and r["UNI_pred"]!="TIE" and not r["contradiction"]
        if name=="CONTRADICTION": return r["contradiction"]
        if name=="HC_CONTRADICTION": return r["hc_contradiction"]
        if name=="CONTRADICTION_PURE": return r["contradiction"] and r["gt_pure_or_mixed"]=="PURE"
        if name=="CONTRADICTION_MIXED": return r["contradiction"] and r["gt_pure_or_mixed"]=="MIXED"
        if name=="WW_HEAVY": return r["WW_ratio"]>.5
        if name=="HC90WW_HEAVY": return r["HC90WW_ratio"]>.5
        if name.startswith("D_Q"): return r["D_quartile"]==name
        return False
    subset_names=("ALL_AGREEMENT","UNI_CPS_AGREE","CONTRADICTION","HC_CONTRADICTION","CONTRADICTION_PURE","CONTRADICTION_MIXED","WW_HEAVY","HC90WW_HEAVY","D_Q1","D_Q2","D_Q3","D_Q4")
    subset_rows=[]
    for name in subset_names:
        subset_rows.append(metric_stats(records,lambda r,n=name:is_sub(n,r))|{"subset":name})
    write_csv(out/"token_records.csv",records); write_csv(out/"subset_metrics.csv",subset_rows)
    # Coverage, event aggregates, and size strata.
    crows=[]
    for name,fn in (("CONTRADICTION",lambda r:r["contradiction"]),("UNI_CPS_AGREE",lambda r:r["agreement"] and r["UNI_pred"]!="TIE" and not r["contradiction"])):
        rr=[r for r in records if fn(r)]; mw=metrics.mean([r["WW_ratio"] for r in rr]); mh=metrics.mean([r["HC90WW_ratio"] for r in rr]); mc=metrics.mean([r["CC_ratio"] for r in rr]); crows.append({"subset":name,"WW_pixel_coverage":mw,"HC90WW_pixel_coverage":mh,"CC_pixel_exposure":mc,"WW_pixel_precision_in_contradiction_regions":metrics.mean([r["WW_ratio"]/(r["WW_ratio"]+r["CC_ratio"]+1e-6) for r in rr]),"HC90WW_pixel_precision_in_contradiction_regions":metrics.mean([r["HC90WW_ratio"]/(r["HC90WW_ratio"]+r["CC_ratio"]+1e-6) for r in rr]),"mean_selectivity":(mw+mh)/(mc+1e-6)})
    contradiction_pixels=sum(int(r["WW_ratio"]*256) for r in records if r["contradiction"]); contradiction_area=sum(256 for r in records if r["contradiction"])
    crows.append({"subset":"CONTRADICTION_REGION_UNION","WW_pixel_coverage":metrics.mean([float((z["WW"].astype(bool)&v["cmap"]).sum()/max(1,z["WW"].sum())) for v in grouped.values() for z in [v["z"]]]),"HC90WW_pixel_coverage":metrics.mean([float((z["HC90WW"].astype(bool)&v["cmap"]).sum()/max(1,z["HC90WW"].sum())) for v in grouped.values() for z in [v["z"]]]),"CC_pixel_exposure":metrics.mean([float((z["CC"].astype(bool)&v["cmap"]).sum()/max(1,z["CC"].sum())) for v in grouped.values() for z in [v["z"]]]),"WW_pixel_precision_in_contradiction_regions":contradiction_pixels/max(1,contradiction_area),"HC90WW_pixel_precision_in_contradiction_regions":float(sum(int(v["z"]["HC90WW"].astype(bool)[v["cmap"]].sum()) for v in grouped.values())/max(1,contradiction_area)),"mean_selectivity":float("nan")})
    write_csv(out/"coverage_metrics.csv",crows)
    # Per-event records plus compact aggregate rows by event type and size.
    event_agg=[]
    for typ in EVENT_TYPES:
        rr=[r for r in events if r["event_type"]==typ]; event_agg.append({"scope":"ALL","event_type":typ,"size_bin":"ALL","num_events":len(rr),"mean_coverage":metrics.mean([r["coverage"] for r in rr]),"event_coverage_rate":metrics.mean([r["covered"] for r in rr])})
        for label,lo,hi in SIZE_BINS:
            x=[r for r in rr if lo<=r["area_pixels"]<=hi]; event_agg.append({"scope":"ALL","event_type":typ,"size_bin":label,"num_events":len(x),"mean_coverage":metrics.mean([r["coverage"] for r in x]),"event_coverage_rate":metrics.mean([r["covered"] for r in x])})
    write_csv(out/"event_metrics.csv",event_agg)
    # Direction metrics and fixed hypothetical token steps.
    for eta in (.1,.25):
        for r in records:
            if r["contradiction"]:
                p=float(r["consensus_prob"]); target=float(r["gt_occupancy"]); sign=1 if r["uni_direction"]=="FG" else -1; after=np.clip(p+eta*sign,0,1); r[f"eta_{eta}_improved"]=bool(abs(after-target)<abs(p-target))
            else: r[f"eta_{eta}_improved"]=np.nan
    # Rebuild rows after adding counterfactual columns.
    direction_rows=[]
    for name in subset_names:
        rr=[r for r in records if is_sub(name,r)]; direction_rows.append({"subset":name,"num_tokens":len(rr),"direction_accuracy":metrics.mean([r["direction_correct"] for r in rr]),"counterfactual_eta_0.1":metrics.mean([r["eta_0.1_improved"] for r in rr]),"counterfactual_eta_0.25":metrics.mean([r["eta_0.25_improved"] for r in rr])})
    write_csv(out/"direction_metrics.csv",direction_rows)
    # Image and epoch aggregates.
    write_csv(out/"image_metrics.csv",image_rows)
    epoch_rows=[]
    for epoch in sorted({int(r["epoch"]) for r in records}):
        ir=[r for r in image_rows if int(r["epoch"])==epoch]; rr=[r for r in records if int(r["epoch"])==epoch]; cr=[r for r in rr if r["contradiction"]]; nr=[r for r in rr if r["agreement"] and r["UNI_pred"]!="TIE" and not r["contradiction"]]
        epoch_rows.append({"epoch":epoch,"contradiction_tokens":len(cr),"hc_contradiction_tokens":sum(r["hc_contradiction"] for r in cr),"WW_enrichment":metrics.mean([r["WW_ratio"] for r in cr])/(metrics.mean([r["WW_ratio"] for r in nr])+1e-6),"HC90WW_enrichment":metrics.mean([r["HC90WW_ratio"] for r in cr])/(metrics.mean([r["HC90WW_ratio"] for r in nr])+1e-6),"WW_coverage":metrics.mean([r["WW_coverage"] for r in ir]),"HC90WW_coverage":metrics.mean([r["HC90WW_coverage"] for r in ir]),"CC_exposure":metrics.mean([r["mean_CC_ratio_contradiction"] for r in ir]),"direction_accuracy":metrics.mean([r["direction_accuracy"] for r in ir]),"WW_event_coverage":metrics.mean([r["WW_event_coverage"] for r in ir]),"HC90WW_event_coverage":metrics.mean([r["HC90WW_event_coverage"] for r in ir])})
    write_csv(out/"epoch_metrics.csv",epoch_rows)
    # Image-level paired bootstrap (all values are image means, never token bootstrap).
    def image_bootstrap(field): return metrics.bootstrap([r[field] for r in image_rows])
    cr=[r for r in records if r["contradiction"]]; nr=[r for r in records if r["agreement"] and r["UNI_pred"]!="TIE" and not r["contradiction"]]
    mean_contra=lambda key: metrics.mean([r[key] for r in cr]); mean_non=lambda key: metrics.mean([r[key] for r in nr])
    enrichment={"WW":mean_contra("WW_ratio")/(mean_non("WW_ratio")+1e-6),"HC90WW":mean_contra("HC90WW_ratio")/(mean_non("HC90WW_ratio")+1e-6)}
    image_enrichment=[]
    for image_id, epoch in sorted({(r["image_id"], int(r["epoch"])) for r in records}):
        rr=[r for r in records if r["image_id"]==image_id and int(r["epoch"])==epoch]
        ci=[r for r in rr if r["contradiction"]]; ni=[r for r in rr if r["agreement"] and r["UNI_pred"]!="TIE" and not r["contradiction"]]
        image_enrichment.append({"WW":metrics.mean([r["WW_ratio"] for r in ci])/(metrics.mean([r["WW_ratio"] for r in ni])+1e-6),"HC90WW":metrics.mean([r["HC90WW_ratio"] for r in ci])/(metrics.mean([r["HC90WW_ratio"] for r in ni])+1e-6)})
    union_coverage=next(r for r in crows if r["subset"]=="CONTRADICTION_REGION_UNION")
    # Plot 1: contradiction versus agreement distributions.
    plots=out/"plots"; fig,ax=plt.subplots(figsize=(8,4)); labels=["WW ratio","HC90WW ratio","CC ratio"]; x=np.arange(3); a=[mean_contra("WW_ratio"),mean_contra("HC90WW_ratio"),mean_contra("CC_ratio")]; b=[mean_non("WW_ratio"),mean_non("HC90WW_ratio"),mean_non("CC_ratio")]; ax.bar(x-.18,a,.36,label="contradiction"); ax.bar(x+.18,b,.36,label="UNI=CPS"); ax.set_xticks(x,labels); ax.legend(); fig.tight_layout(); fig.savefig(plots/"contradiction_vs_agreement.png",dpi=150); plt.close(fig)
    # Plot 2: D quartile relationships.
    fig,ax=plt.subplots(figsize=(8,4)); qlabels=[]; vals=[]
    for qn in ("D_Q1","D_Q2","D_Q3","D_Q4"):
        x=[r for r in cr if r["D_quartile"]==qn]; qlabels.append(qn); vals.append(metrics.mean([r["WW_ratio"] for r in x]))
    ax.plot(qlabels,vals,"o-"); ax.set_ylabel("mean WW ratio"); fig.tight_layout(); fig.savefig(plots/"D_quartile_WW.png",dpi=150); plt.close(fig)
    # Plot 3: epoch trajectory.
    fig,ax=plt.subplots(figsize=(8,4)); ax.plot([r["epoch"] for r in epoch_rows],[r["WW_enrichment"] for r in epoch_rows],"o-",label="WW enrichment"); ax.plot([r["epoch"] for r in epoch_rows],[r["HC90WW_enrichment"] for r in epoch_rows],"o-",label="HC90WW enrichment"); ax.plot([r["epoch"] for r in epoch_rows],[r["direction_accuracy"] for r in epoch_rows],"o-",label="direction"); ax.legend(); ax.set_ylim(0,None); fig.tight_layout(); fig.savefig(plots/"epoch_feasibility.png",dpi=150); plt.close(fig)
    # Representative native token maps: first cases from contradiction and HC contradiction.
    for (image_id,epoch),v in list(grouped.items())[:8]:
        rgb=np.load(e1/"images"/f"{image_id}.npz")["rgb"].astype(np.float32)/255
        tm=v["tm"]; visualize.save_token_case(rgb,{"agreement":tm["agreement"],"contradiction":tm["contradiction"],"WW_ratio":tm["WW_ratio"],"HC90WW_ratio":tm["HC90WW_ratio"],"occupancy":tm["occupancy"],"D":tm["D"]},f"{image_id} epoch {epoch}",plots/f"token_case_{image_id}_e{epoch}.png")
    summary={"uni_input":256,"native_grid":"16x16","images":17,"epochs":args.epochs_list,"memory_leakage":False,"contradiction":{"num_tokens":len(cr),"num_images":len({r["image_id"] for r in cr}),"mean_WW_ratio":mean_contra("WW_ratio"),"median_WW_ratio":metrics.median([r["WW_ratio"] for r in cr]),"mean_HC90WW_ratio":mean_contra("HC90WW_ratio"),"mean_CC_ratio":mean_contra("CC_ratio"),"mean_shared_FP_ratio":mean_contra("shared_FP_ratio"),"mean_shared_FN_ratio":mean_contra("shared_FN_ratio")},"non_contradiction":{"mean_WW_ratio":mean_non("WW_ratio"),"mean_HC90WW_ratio":mean_non("HC90WW_ratio"),"mean_CC_ratio":mean_non("CC_ratio")},"enrichment":enrichment,"coverage":{"WW_pixel":union_coverage["WW_pixel_coverage"],"HC90WW_pixel":union_coverage["HC90WW_pixel_coverage"],"CC_exposure":union_coverage["CC_pixel_exposure"],"WW_event":metrics.mean([r["covered"] for r in events if r["event_type"]=="WW"]),"HC90WW_event":metrics.mean([r["covered"] for r in events if r["event_type"]=="HC90WW"])},"direction":{"all_contradiction":mean_contra("direction_correct"),"hc_contradiction":metrics.mean([r["direction_correct"] for r in records if r["hc_contradiction"]]),"mixed_contradiction":metrics.mean([r["direction_correct"] for r in records if r["contradiction"] and r["gt_pure_or_mixed"]=="MIXED"]),"WW_heavy":metrics.mean([r["direction_correct"] for r in records if r["WW_ratio"]>.5]),"HC90WW_heavy":metrics.mean([r["direction_correct"] for r in records if r["HC90WW_ratio"]>.5])},"counterfactual_step":{"eta_0.1_improvement_rate":mean_contra("eta_0.1_improved"),"eta_0.25_improvement_rate":mean_contra("eta_0.25_improved")},"purity":{"P_WW_ratio_gt_25":metrics.mean([r["WW_ratio"]>.25 for r in cr]),"P_WW_ratio_gt_50":metrics.mean([r["WW_ratio"]>.5 for r in cr]),"P_HC90WW_ratio_gt_25":metrics.mean([r["HC90WW_ratio"]>.25 for r in cr]),"P_HC90WW_ratio_gt_50":metrics.mean([r["HC90WW_ratio"]>.5 for r in cr]),"P_CC_ratio_gt_50":metrics.mean([r["CC_ratio"]>.5 for r in cr])},"image_bootstrap_1000":{"WW_enrichment":metrics.bootstrap([r["WW"] for r in image_enrichment]),"HC90WW_enrichment":metrics.bootstrap([r["HC90WW"] for r in image_enrichment]),"WW_coverage":image_bootstrap("WW_coverage"),"HC90WW_coverage":image_bootstrap("HC90WW_coverage"),"direction_accuracy":image_bootstrap("direction_accuracy"),"WW_event_coverage":image_bootstrap("WW_event_coverage"),"HC90WW_event_coverage":image_bootstrap("HC90WW_event_coverage")},"cps_prediction_source":"frozen standard CPS current-weight milestones; validation states loaded from E1","memory_path":str(memory),"output_directory":str(out)}
    config={"task":"closed E4 native-scale reference feasibility","uni_input_resolution":256,"native_grid":"16x16","feature_dim":1024,"memory_path":str(memory),"memory_sha256":sha256(memory),"memory_source_manifest":str(source_manifest),"memory_metadata":mem_meta,"memory_leakage":False,"cps_source":str((ROOT/args.prediction_root).resolve()),"epochs":args.epochs_list,"validation_manifest":str(val_manifest),"validation_image_count":17,"top_k":5,"gt_used_for":"offline evaluation only; never contradiction signal","direction_dead_zone":DEAD_ZONE,"counterfactual_eta":[.1,.25],"forbidden":"no training, no pixel interpolation, no R, no threshold selection, no CPS modification"}
    (out/"config.json").write_text(json.dumps(safe(config),indent=2)+"\n"); (out/"summary.json").write_text(json.dumps(safe(summary),indent=2)+"\n")
    (out/"README.md").write_text((ROOT/"analysis/native_scale_reference/README.md").read_text())
    print("=== E4 Native-Scale Reference Feasibility ==="); print("UNI input: 256x256"); print("Native grid: 16x16"); print("Images / epochs: 17 / 5"); print("Contradiction tokens:",len(cr)); print("HC contradiction tokens:",sum(r["hc_contradiction"] for r in cr)); print("Contradiction mean WW/HC90WW/CC:",summary["contradiction"]["mean_WW_ratio"],summary["contradiction"]["mean_HC90WW_ratio"],summary["contradiction"]["mean_CC_ratio"]); print("UNI=CPS mean WW/HC90WW/CC:",summary["non_contradiction"]["mean_WW_ratio"],summary["non_contradiction"]["mean_HC90WW_ratio"],summary["non_contradiction"]["mean_CC_ratio"]); print("WW enrichment:",enrichment["WW"]); print("HC90WW enrichment:",enrichment["HC90WW"]); print("WW pixel coverage:",summary["coverage"]["WW_pixel"]); print("HC90WW pixel coverage:",summary["coverage"]["HC90WW_pixel"]); print("CC pixel exposure:",summary["coverage"]["CC_exposure"]); print("WW event coverage:",summary["coverage"]["WW_event"]); print("HC90WW event coverage:",summary["coverage"]["HC90WW_event"]); print("Direction accuracy:",summary["direction"]["all_contradiction"]); print("HC contradiction direction accuracy:",summary["direction"]["hc_contradiction"]); print("MIXED contradiction direction accuracy:",summary["direction"]["mixed_contradiction"]); print("Counterfactual eta=0.1/0.25:",summary["counterfactual_step"]["eta_0.1_improvement_rate"],summary["counterfactual_step"]["eta_0.25_improvement_rate"]); print("Output:",out)


if __name__ == "__main__": main()
