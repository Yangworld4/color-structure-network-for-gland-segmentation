"""E3: closed, offline UNI input-resolution diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
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

from analysis.uni_resolution_diagnostic import metrics, visualize
from analysis.uni_token_arbitration.utils import load_memory, load_students, reconstruct_rgb, score_native, sha256
from config.augmentation.online_aug import data_normalize_2d, data_transform_2d
from config.dataset_config.dataset_cfg import dataset_cfg
from dataload.dataset_2d import imagefloder_itn
from models.uni import load_uni
import main as project_main

UNI_WEIGHTS = "/home/yangorange/.cache/huggingface/hub/models--MahmoodLab--UNI/snapshots/b55a5ec6cade1a39edfe6534189a9b8ca7a022f0/pytorch_model.bin"
UNI_CONFIG = str(Path(UNI_WEIGHTS).with_name("config.json"))
UNI_SHA = "56ef09b44a25dc5c7eedc55551b3d47bcd17659a7a33837cf9abc9ec4e2ffb40"
RESOLUTIONS = {"UNI224": (224, 14), "UNI256": (256, 16)}
PRIMARY = ("CC", "WW", "HC90WW", "SHARED_FP", "SHARED_FN")
BOUNDARY = ("WW_BOUNDARY_0_4", "WW_BOUNDARY_4_8", "WW_BOUNDARY_8_16", "WW_INTERIOR_GT16",
            "HC90WW_BOUNDARY_0_4", "HC90WW_BOUNDARY_4_8", "HC90WW_BOUNDARY_8_16", "HC90WW_INTERIOR_GT16")
FOOTPRINT = ("WW_PURE", "WW_MIXED", "HC90WW_PURE", "HC90WW_MIXED")
SIZE_BINS = (("WW_SMALL_1_16", 1, 16), ("WW_SMALL_17_64", 17, 64),
             ("WW_SMALL_65_256", 65, 256), ("WW_SMALL_GT256", 257, np.inf))


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


def ids_from_manifest(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {Path(row["image"]).stem for row in csv.DictReader(handle)}


def load_e1_records(e1_root):
    rows = list(csv.DictReader((e1_root / "pixel_records.csv").open(encoding="utf-8")))
    if not rows: raise RuntimeError("E1 pixel manifest is empty")
    return rows


def direct_uni_tokens(uni, rgb, resolution):
    """Project RGB to the exact UNI input; 256 takes no spatial resize."""
    if rgb.ndim != 4 or tuple(rgb.shape[-2:]) != (256, 256):
        raise RuntimeError(f"Expected aligned RGB [B,3,256,256], got {tuple(rgb.shape)}")
    if resolution == 224:
        x = F.interpolate(rgb, size=(224, 224), mode="bilinear", align_corners=False)
    elif resolution == 256:
        x = rgb
    else:
        raise ValueError(resolution)
    mean = torch.tensor(project_main.UNI_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(project_main.UNI_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    tokens = uni.dense_patch_tokens((x - mean) / std)
    expected = RESOLUTIONS["UNI224" if resolution == 224 else "UNI256"][1]
    if tuple(x.shape[-2:]) != (resolution, resolution): raise AssertionError(tuple(x.shape))
    if tuple(tokens.shape) != (rgb.shape[0], expected * expected, 1024):
        raise RuntimeError(f"Unexpected UNI-{resolution} token shape: {tuple(tokens.shape)}")
    print(f"UNI input shape: {list(x.shape)} native patch tokens: {list(tokens.shape)} native grid: {expected}x{expected} feature dim: 1024")
    return tokens


def build_256_memory(args, cfg, manifest_dir, uni, path):
    """Reproduce the project's memory algorithm for the new 16x16 grid.

    The existing training helper intentionally accepts only the historical
    224 resolution.  This local diagnostic copy keeps every memory rule and
    constant identical while changing only the requested native resolution.
    """
    ns = argparse.Namespace(**vars(args))
    ns.dataset_name = "GlaS"; ns.path_dataset = str((ROOT / args.dataset_root).resolve())
    ns.input1 = "image"; ns.input_mode = "rgb_stain"; ns.labeled_ratio = 10
    ns.sup_mark = "9"; ns.reference_resolution = 256; ns.batch_size = 2
    dataset = project_main._reference_memory_dataset(ns, cfg, str(manifest_dir), True)
    loader = DataLoader(dataset, batch_size=ns.batch_size, shuffle=False, num_workers=0)
    memory_by_class = {0: [], 1: []}; rng = np.random.default_rng(project_main.REFERENCE_MEMORY_SEED)
    fallback_count = 0
    with torch.inference_mode():
        for batch in loader:
            rgb = project_main._reconstruct_rgb(batch["image_rgb"].cuda(non_blocking=True), cfg)
            tokens = direct_uni_tokens(uni, rgb, 256)
            gt = batch["mask"].cuda(non_blocking=True).float().unsqueeze(1)
            occupancy = F.adaptive_avg_pool2d(gt, (16, 16)).flatten(1)
            token_cpu = tokens.detach().cpu().numpy(); occupancy_cpu = occupancy.detach().cpu().numpy()
            for index in range(len(batch["ID"])):
                for cls, threshold in ((1, project_main.REFERENCE_MEMORY_ANCHOR_FG), (0, project_main.REFERENCE_MEMORY_ANCHOR_BG)):
                    row = occupancy_cpu[index]
                    candidates = np.flatnonzero(row >= threshold) if cls == 1 else np.flatnonzero(row <= threshold)
                    order = np.argsort(-row, kind="stable") if cls == 1 else np.argsort(row, kind="stable")
                    if not len(candidates):
                        fallback_count += 1
                        fallback_size = max(1, min(project_main.REFERENCE_MEMORY_PER_IMAGE_PER_CLASS, int(math.ceil(len(row) * project_main.REFERENCE_MEMORY_FALLBACK_FRACTION))))
                        candidates = order[:fallback_size]
                    chosen = candidates
                    if len(candidates) > project_main.REFERENCE_MEMORY_PER_IMAGE_PER_CLASS:
                        chosen = np.sort(rng.choice(candidates, project_main.REFERENCE_MEMORY_PER_IMAGE_PER_CLASS, replace=False))
                    for token_index in chosen: memory_by_class[cls].append(token_cpu[index, token_index])
    if not memory_by_class[0] or not memory_by_class[1]: raise RuntimeError("UNI256 memory is empty")
    fg = F.normalize(torch.from_numpy(np.stack(memory_by_class[1])).float(), p=2, dim=1).numpy().astype(np.float32)
    bg = F.normalize(torch.from_numpy(np.stack(memory_by_class[0])).float(), p=2, dim=1).numpy().astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), memory_fg=fg, memory_bg=bg,
        memory_fg_count=np.array(fg.shape[0]), memory_bg_count=np.array(bg.shape[0]),
        feature_dim=np.array(1024), token_grid=np.array(16), top_k=np.array(5),
        anchor_fg=np.array(project_main.REFERENCE_MEMORY_ANCHOR_FG), anchor_bg=np.array(project_main.REFERENCE_MEMORY_ANCHOR_BG),
        memory_per_image_class=np.array(project_main.REFERENCE_MEMORY_PER_IMAGE_PER_CLASS),
        anchor_fallback_fraction=np.array(project_main.REFERENCE_MEMORY_FALLBACK_FRACTION), anchor_fallback_count=np.array(fallback_count),
        source_labeled_ratio=np.array(10), seed=np.array(project_main.REFERENCE_MEMORY_SEED), source_dataset=np.array("GlaS"), reference_resolution=np.array(256))


def state_masks(gt, pa, pb, occupancy, fg=.9, bg=.1):
    gt_token_fg = occupancy >= fg; pure = (occupancy >= fg) | (occupancy <= bg)
    ha, hb = pa >= .5, pb >= .5
    ca, cb = ha == gt_token_fg, hb == gt_token_fg
    agreement = ha == hb
    cc = pure & ca & cb
    ww = pure & (~ca) & (~cb) & agreement
    hc = ww & (np.maximum(pa, 1-pa) >= .9) & (np.maximum(pb, 1-pb) >= .9)
    fp = ww & ha & (~gt_token_fg); fn = ww & (~ha) & gt_token_fg
    return {"PURE": pure, "PURE_FG": pure & gt_token_fg, "PURE_BG": pure & (~gt_token_fg),
            "MIXED": ~pure, "CC":cc, "WW":ww, "HC90WW":hc,
            "SHARED_FP":fp, "SHARED_FN":fn, "gt_fg":gt_token_fg}


def native_stats_for(fields, z, resolution, grid):
    gt = z["gt"].astype(np.float32)
    gt_in = torch.from_numpy(gt)[None, None]
    if resolution == 224: gt_in = F.interpolate(gt_in, size=(224,224), mode="nearest")
    occupancy = F.adaptive_avg_pool2d(gt_in, (grid,grid))[0,0].numpy()
    pa = F.adaptive_avg_pool2d(torch.from_numpy(z["pA"])[None,None], (grid,grid))[0,0].numpy()
    pb = F.adaptive_avg_pool2d(torch.from_numpy(z["pB"])[None,None], (grid,grid))[0,0].numpy()
    st = state_masks(gt, pa, pb, occupancy)
    margin=fields["margin"]; pred=margin > 0; ties=margin == 0
    correct=(pred == st["gt_fg"])
    d=np.where((pa >= .5), -margin, margin)
    return {"occupancy":occupancy,"margin":margin,"correct":correct,"D":d,"ties":ties,"states":st}


def pixel_masks(z, occupancy, grid):
    h,w=z["gt"].shape; yy=np.minimum(np.arange(h)*grid//h,grid-1); xx=np.minimum(np.arange(w)*grid//w,grid-1)
    occ=occupancy[yy[:,None],xx[None,:]]; pure= (occ>=.9)|(occ<=.1)
    out={"ALL":np.ones((h,w),bool),"CC":z["CC"].astype(bool),"WW":z["WW"].astype(bool),"HC90WW":z["HC90WW"].astype(bool),"SHARED_FP":z["SHARED_FP"].astype(bool),"SHARED_FN":z["SHARED_FN"].astype(bool)}
    dist=z["boundary_distance"]
    out.update({"WW_BOUNDARY_0_4":out["WW"]&(dist<=4),"WW_BOUNDARY_4_8":out["WW"]&(dist>4)&(dist<=8),"WW_BOUNDARY_8_16":out["WW"]&(dist>8)&(dist<=16),"WW_INTERIOR_GT16":out["WW"]&(dist>16),"HC90WW_BOUNDARY_0_4":out["HC90WW"]&(dist<=4),"HC90WW_BOUNDARY_4_8":out["HC90WW"]&(dist>4)&(dist<=8),"HC90WW_BOUNDARY_8_16":out["HC90WW"]&(dist>8)&(dist<=16),"HC90WW_INTERIOR_GT16":out["HC90WW"]&(dist>16),"WW_PURE":out["WW"]&pure,"WW_MIXED":out["WW"]&(~pure),"HC90WW_PURE":out["HC90WW"]&pure,"HC90WW_MIXED":out["HC90WW"]&(~pure)})
    size=z["ww_component_size"]
    for name,lo,hi in SIZE_BINS: out[name]=out["WW"]&(size>=lo)&(size<=hi)
    return out, occ


def project_pixel(native_margin):
    return F.interpolate(torch.from_numpy(native_margin)[None,None], size=(256,256), mode="bilinear", align_corners=False)[0,0].numpy().astype(np.float32)


def append_stats(store, resolution, unit, subset, correct, d, mask):
    mask=mask.astype(bool); store.setdefault((resolution,unit,subset), []).append({"correct":correct[mask].astype(float),"D":d[mask].astype(float),"n":int(mask.sum())})


def merge_stats(rows):
    if not rows:return {"num":0,"accuracy":float("nan"),"mean_D":float("nan"),"D_positive_rate":float("nan")}
    c=np.concatenate([r["correct"] for r in rows if len(r["correct"])]) if any(len(r["correct"]) for r in rows) else np.empty(0)
    d=np.concatenate([r["D"] for r in rows if len(r["D"])]) if any(len(r["D"]) for r in rows) else np.empty(0)
    return {"num":len(c),"accuracy":metrics.mean(c),"mean_D":metrics.mean(d),"D_positive_rate":metrics.mean(d>0)}


def event_values(mask, correct, d):
    labels,n=ndimage.label(mask.astype(bool),structure=np.ones((3,3),bool)); rows=[]
    for i in range(1,n+1):
        e=labels==i; rows.append((i,int(e.sum()),metrics.mean(correct[e]),metrics.mean(d[e]),bool(metrics.mean(correct[e])>.5)))
    return rows


def write_case_dirs(out):
    for name in ("fixed_by_256","broken_by_256","no_change"):(out/"cases"/name).mkdir(parents=True,exist_ok=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root",default="datasets/GLaS"); ap.add_argument("--prediction-root",required=True)
    ap.add_argument("--memory-224",default="checkpoints/glas_labeled10_seed24/reference_memory.npz")
    ap.add_argument("--memory-256",default="checkpoints/glas_labeled10_seed24/reference_memory_uni256.npz")
    ap.add_argument("--e1-root",default="outputs/uni_pixel_degradation/glas_standard_cps_uni_token_to_pixel")
    ap.add_argument("--epochs-list",nargs="+",type=int,default=[40,80,120,160,200]); ap.add_argument("--run-name",default="glas_standard_cps_e3")
    ap.add_argument("--output-root",default="outputs/uni_resolution_diagnostic"); ap.add_argument("--device",default="cuda:0")
    ap.add_argument("--uni-weights",default=UNI_WEIGHTS); ap.add_argument("--uni-config",default=UNI_CONFIG)
    args=ap.parse_args(); device=torch.device(args.device)
    if device.type!="cuda" or not torch.cuda.is_available(): raise RuntimeError("E3 requires CUDA")
    out=(ROOT/args.output_root/args.run_name).resolve(); out.mkdir(parents=True,exist_ok=True); write_case_dirs(out)
    e1_root=(ROOT/args.e1_root).resolve(); manifests=load_e1_records(e1_root); cfg=dataset_cfg("GlaS"); dataset_root=(ROOT/args.dataset_root).resolve(); manifest_dir=dataset_root/"manifests/paper_seed_2026"; eval_manifest=manifest_dir/"validation.csv"; mem224=(ROOT/args.memory_224).resolve(); mem256=(ROOT/args.memory_256).resolve()
    source_manifest=manifest_dir/"labeled_10.csv"; leakage=ids_from_manifest(eval_manifest)&ids_from_manifest(source_manifest)
    if leakage: raise RuntimeError("ERROR: memory/evaluation image overlap: "+", ".join(sorted(leakage)))
    if len(manifests)!=85 or len({r["image_id"] for r in manifests})!=17: raise RuntimeError("Unexpected E1 validation record count")
    if not mem224.is_file(): raise FileNotFoundError(mem224)
    with np.load(mem224,allow_pickle=False) as m:
        memory_resolution = int(m["reference_resolution"]) if "reference_resolution" in m.files else 224
        if memory_resolution != 224 or int(m["token_grid"]) != 14: raise RuntimeError("UNI224 memory metadata mismatch")
    dataset=imagefloder_itn(manifest_path=str(eval_manifest),data_transform_1=data_transform_2d()["val"],data_normalize_1=data_normalize_2d(cfg["MEAN"],cfg["STD"]),sup=True,input_mode="rgb_stain",training=False)
    loader=DataLoader(dataset,batch_size=2,shuffle=False,num_workers=0)
    uni,_,_=load_uni(args.uni_weights,args.uni_config,expected_sha256=UNI_SHA)
    print("CPS source:",Path(args.prediction_root).resolve()); print("UNI memory source 224:",mem224); print("UNI memory source 256:",mem256); print("epochs:",args.epochs_list); print("validation image count:",len(dataset)); print("memory leakage: false")
    memfg224,mbg224,meta224=load_memory(mem224,device)
    # First pass computes both resolutions from exactly the same aligned RGB.
    fields={"UNI224":{},"UNI256":{}}
    with torch.inference_mode():
        for batch in loader:
            rgb=reconstruct_rgb(batch["image_rgb"].to(device),tuple(cfg["MEAN"]),tuple(cfg["STD"]))
            for name,(resolution,grid) in RESOLUTIONS.items():
                if name == "UNI256":
                    # UNI-256 is extracted only after its train-only memory is
                    # rebuilt below; this first pass establishes UNI-224.
                    continue
                tokens=direct_uni_tokens(uni,rgb,resolution)
                if name=="UNI224": fg,bg=memfg224,mbg224
                sf, sb, margin=score_native(tokens,fg,bg,5,grid)
                gt=batch["mask"].to(device).float()[:,None]
                gt_in=F.interpolate(gt,(resolution,resolution),mode="nearest") if resolution==224 else gt
                occ=F.adaptive_avg_pool2d(gt_in,(grid,grid))[:,0]
                for i,name_id in enumerate(batch["ID"]): fields["UNI224"][Path(str(name_id)).stem]={"S_FG":sf[i].cpu().numpy(),"S_BG":sb[i].cpu().numpy(),"margin":margin[i].cpu().numpy(),"occupancy":occ[i].cpu().numpy()}
    # Rebuild UNI-256 memory from the same labeled_10.csv using project code, then extract UNI-256.
    build_256_memory(args,cfg,manifest_dir,uni,mem256)
    memfg256,mbg256,meta256=load_memory(mem256,device)
    if int(meta256["reference_resolution"])!=256 or int(meta256["token_grid"])!=16: raise RuntimeError("UNI256 memory metadata mismatch")
    with torch.inference_mode():
        for batch in loader:
            rgb=reconstruct_rgb(batch["image_rgb"].to(device),tuple(cfg["MEAN"]),tuple(cfg["STD"]))
            tokens=direct_uni_tokens(uni,rgb,256); sf,sb,margin=score_native(tokens,memfg256,mbg256,5,16)
            gt=batch["mask"].to(device).float()[:,None]; occ=F.adaptive_avg_pool2d(gt,(16,16))[:,0]
            for i,name_id in enumerate(batch["ID"]): fields["UNI256"][Path(str(name_id)).stem]={"S_FG":sf[i].cpu().numpy(),"S_BG":sb[i].cpu().numpy(),"margin":margin[i].cpu().numpy(),"occupancy":occ[i].cpu().numpy()}
    if set(fields["UNI224"])!=set(fields["UNI256"]): raise RuntimeError("UNI field image IDs differ")
    # Immutable E0/E1 baseline values used for the reproduction gate.
    e0=json.loads((ROOT/"outputs/uni_token_arbitration/glas_standard_cps_uni_native_token/summary.json").read_text()); e1_summary=json.loads((e1_root/"summary.json").read_text())
    native_store={}; pixel_store={}; event_records=[]; image_rows=[]; event_by_key={}; case_pool=[]; native_counts={"UNI224":{},"UNI256":{}}
    epoch_records={}
    for item in manifests:
        image_id=item["image_id"]; epoch=int(item["epoch"]); z=dict(np.load(e1_root/item["path"]))
        for name,(resolution,grid) in RESOLUTIONS.items():
            ns=native_stats_for(fields[name][image_id],z,resolution,grid); st=ns["states"]
            for subset in ("PURE","PURE_FG","PURE_BG","MIXED","CC","WW","HC90WW","SHARED_FP","SHARED_FN"):
                mask=st[subset]
                append_stats(native_store,name,"native",subset,ns["correct"],ns["D"],mask)
            # Pixel state is identical for both methods and comes from E1's raw CPS state.
            pm,occ=pixel_masks(z,fields[name][image_id]["occupancy"],grid); pixel_margin=project_pixel(fields[name][image_id]["margin"])
            correct=(pixel_margin>0)==z["gt"].astype(bool); d=np.where(z["hard_A"].astype(bool),-pixel_margin,pixel_margin)
            for subset,mask in pm.items(): append_stats(pixel_store,name,"pixel",subset,correct,d,mask)
            ev_masks={k:pm[k] for k in ("WW","HC90WW","SHARED_FP","SHARED_FN")}; ev_by_type={}
            for typ,mask in ev_masks.items():
                ev_by_type[typ]=event_values(mask,correct,d)
                for idx,area,acc,md,success in ev_by_type[typ]: event_records.append({"image_id":image_id,"epoch":epoch,"resolution":name,"event_type":typ,"event_index":idx,"area_pixels":area,"event_accuracy":acc,"event_arbitrated":success,"mean_D":md})
            row=image_rows[ -1 ] if image_rows and image_rows[-1]["image_id"]==image_id and image_rows[-1]["epoch"]==epoch else {"image_id":image_id,"epoch":epoch}
            row.update({f"{name}_WW":metrics.mean(correct[pm["WW"]]),f"{name}_HC90WW":metrics.mean(correct[pm["HC90WW"]]),f"{name}_boundary_WW":metrics.mean(correct[pm["WW_BOUNDARY_0_4"]]),f"{name}_mixed_WW":metrics.mean(correct[pm["WW_MIXED"]]),f"{name}_event_WW":metrics.mean([x[4] for x in ev_by_type["WW"]]),f"{name}_event_HC90WW":metrics.mean([x[4] for x in ev_by_type["HC90WW"]])})
            if row not in image_rows: image_rows.append(row)
            # Keep per-event paired records for flip analysis and case mining.
            for typ in ev_masks:
                for idx,area,acc,md,success in ev_by_type[typ]: event_by_key.setdefault((image_id,epoch,typ,idx),{})[name]={"area":area,"acc":acc,"success":success,"mean_D":md}
            epoch_records.setdefault(epoch,[]).append((name,pm,correct,d,ev_by_type))
    # Baseline reproduction: fresh UNI-224 native and pixel aggregate metrics must match E0/E1.
    def agg(store,name,unit,subset): return merge_stats(store[(name,unit,subset)])["accuracy"]
    repro={"WW_native":agg(native_store,"UNI224","native","WW"),"HC_native":agg(native_store,"UNI224","native","HC90WW"),"WW_pixel":agg(pixel_store,"UNI224","pixel","WW"),"HC_pixel":agg(pixel_store,"UNI224","pixel","HC90WW")}
    expected={"WW_native":e0["ww_arbitration_accuracy"],"HC_native":e0["hc90ww_arbitration_accuracy"],"WW_pixel":e1_summary["e1_pixel"]["WW"],"HC_pixel":e1_summary["e1_pixel"]["HC90WW"]}
    if any(not np.isclose(repro[k],expected[k],atol=1e-5) for k in expected): raise RuntimeError(f"E3 BASELINE REPRODUCTION FAILED: {repro} vs {expected}")
    native_rows=[]; pixel_rows=[]; subset_rows=[]
    native_metric_map={"PURE accuracy":("PURE",),"balanced accuracy":("PURE_FG","PURE_BG"),"CC":("CC",),"WW":("WW",),"HC90WW":("HC90WW",),"Shared FP":("SHARED_FP",),"Shared FN":("SHARED_FN",)}
    for label,subs in native_metric_map.items():
        vals=[]
        for name in RESOLUTIONS:
            if label=="balanced accuracy":
                a=merge_stats(native_store[(name,"native","PURE_FG")])["accuracy"]; b=merge_stats(native_store[(name,"native","PURE_BG")])["accuracy"]; vals.append(metrics.mean([a,b]))
            else: vals.append(agg(native_store,name,"native",subs[0]))
            r=merge_stats(native_store[(name,"native",subs[0])]); subset_rows.append({"resolution":name,"unit":"native","subset":label,"num":r["num"],"accuracy":r["accuracy"],"mean_D":r["mean_D"],"D_positive_rate":r["D_positive_rate"]})
        native_rows.append({"metric":label,"UNI224":vals[0],"UNI256":vals[1],"delta_256_minus_224":vals[1]-vals[0]})
    for label,subs in (("mean D WW",("WW",)),("mean D HC90WW",("HC90WW",))):
        vals=[merge_stats(native_store[(name,"native",subs[0])])["mean_D"] for name in RESOLUTIONS]; native_rows.append({"metric":label,"UNI224":vals[0],"UNI256":vals[1],"delta_256_minus_224":vals[1]-vals[0]})
    pixel_metric_names={"CC":"CC","WW":"WW","HC90WW":"HC90WW","Shared FP":"SHARED_FP","Shared FN":"SHARED_FN","WW boundary 0-4":"WW_BOUNDARY_0_4","WW boundary 4-8":"WW_BOUNDARY_4_8","WW boundary 8-16":"WW_BOUNDARY_8_16","WW interior >16":"WW_INTERIOR_GT16","HC90WW boundary 0-4":"HC90WW_BOUNDARY_0_4","HC90WW boundary 4-8":"HC90WW_BOUNDARY_4_8","HC90WW boundary 8-16":"HC90WW_BOUNDARY_8_16","HC90WW interior >16":"HC90WW_INTERIOR_GT16","WW PURE":"WW_PURE","WW MIXED":"WW_MIXED","HC90WW PURE":"HC90WW_PURE","HC90WW MIXED":"HC90WW_MIXED"}
    for label,sub in pixel_metric_names.items():
        vals=[]
        for name in RESOLUTIONS:
            r=merge_stats(pixel_store[(name,"pixel",sub)]); vals.append(r["accuracy"]); subset_rows.append({"resolution":name,"unit":"pixel","subset":label,"num":r["num"],"accuracy":r["accuracy"],"mean_D":r["mean_D"],"D_positive_rate":r["D_positive_rate"]})
        pixel_rows.append({"metric":label,"UNI224":vals[0],"UNI256":vals[1],"delta_256_minus_224":vals[1]-vals[0]})
    event_summary={}; event_rows=[]
    for name in RESOLUTIONS:
        for typ in ("WW","HC90WW","SHARED_FP","SHARED_FN"):
            ev=[x for x in event_records if x["resolution"]==name and x["event_type"]==typ]; event_summary.setdefault(name,{})[typ]=metrics.mean([x["event_arbitrated"] for x in ev]); event_rows.append({"resolution":name,"event_type":typ,"num_events":len(ev),"event_arbitration_rate":metrics.mean([x["event_arbitrated"] for x in ev]),"mean_event_accuracy":metrics.mean([x["event_accuracy"] for x in ev]),"mean_D":metrics.mean([x["mean_D"] for x in ev])})
        for label,lo,hi in SIZE_BINS:
            ev=[x for x in event_records if x["resolution"]==name and x["event_type"]=="WW" and lo<=x["area_pixels"]<=hi]; event_rows.append({"resolution":name,"event_type":label,"num_events":len(ev),"event_arbitration_rate":metrics.mean([x["event_arbitrated"] for x in ev]),"mean_event_accuracy":metrics.mean([x["event_accuracy"] for x in ev]),"mean_D":metrics.mean([x["mean_D"] for x in ev])})
    fixed=broken=0; fixed_all=broken_all=0
    for key,pair in event_by_key.items():
        if "UNI224" in pair and "UNI256" in pair:
            is_fixed=int((not pair["UNI224"]["success"]) and pair["UNI256"]["success"])
            is_broken=int(pair["UNI224"]["success"] and (not pair["UNI256"]["success"]))
            fixed_all += is_fixed; broken_all += is_broken
            if key[2] == "WW": fixed += is_fixed; broken += is_broken
    write_csv(out/"native_metrics.csv",native_rows); write_csv(out/"pixel_metrics.csv",pixel_rows); write_csv(out/"subset_metrics.csv",subset_rows); write_csv(out/"event_metrics.csv",event_rows); write_csv(out/"image_metrics.csv",image_rows)
    epoch_rows=[]
    for epoch in sorted(epoch_records):
        row={"epoch":epoch}
        # These fields are calculated directly from image rows for stable
        # paired epoch reporting.
        er=[r for r in image_rows if r["epoch"]==epoch]
        for name in RESOLUTIONS:
            for field in ("WW","HC90WW","boundary_WW","mixed_WW","event_WW","event_HC90WW"): row[f"{name}_{field}"]=metrics.mean([r[f"{name}_{field}"] for r in er])
        epoch_rows.append(row)
    write_csv(out/"epoch_metrics.csv",epoch_rows)
    bootstrap={}
    for field in ("WW","HC90WW","boundary_WW","event_WW","event_HC90WW"):
        vals=[r["UNI256_"+field]-r["UNI224_"+field] for r in image_rows if np.isfinite(r["UNI224_"+field]) and np.isfinite(r["UNI256_"+field])]; bootstrap[field]=metrics.bootstrap(vals)
    for field in ("WW","HC90WW","boundary_WW","event_WW","event_HC90WW"): pass
    # Case mining uses paired event success and stores representative images.
    for key,pair in event_by_key.items():
        if "UNI224" not in pair or "UNI256" not in pair: continue
        status="fixed_by_256" if (not pair["UNI224"]["success"] and pair["UNI256"]["success"]) else "broken_by_256" if (pair["UNI224"]["success"] and not pair["UNI256"]["success"]) else "no_change"
        case_pool.append((status,key,pair))
    for status in ("fixed_by_256","broken_by_256","no_change"):
        for _,(image_id,epoch,typ,idx),pair in [x for x in case_pool if x[0]==status][:20]:
            item=next(x for x in manifests if x["image_id"]==image_id and int(x["epoch"])==epoch); z=dict(np.load(e1_root/item["path"])); rgb=np.load(e1_root/"images"/f"{image_id}.npz")["rgb"].astype(np.float32)/255
            mm={"UNI224_native":fields["UNI224"][image_id]["margin"],"UNI256_native":fields["UNI256"][image_id]["margin"],"UNI224_pixel":project_pixel(fields["UNI224"][image_id]["margin"]),"UNI256_pixel":project_pixel(fields["UNI256"][image_id]["margin"])}
            visualize.save_case(rgb,z,mm,f"{status} {typ} {image_id} e{epoch} event {idx}",out/"cases"/status/f"{typ}_{image_id}_e{epoch}_{idx}.png")
    # Summary and plots.
    def px(name,sub): return agg(pixel_store,name,"pixel",sub)
    def nt(name,sub): return agg(native_store,name,"native",sub)
    mixed_counts={};
    for name in RESOLUTIONS:
        vals=[r["occupancy"] for r in [native_stats_for(fields[name][item["image_id"]],dict(np.load(e1_root/item["path"])),RESOLUTIONS[name][0],RESOLUTIONS[name][1]) for item in manifests]]
        all_occ=np.concatenate([v.ravel() for v in vals]); mixed_counts[name]={"num_mixed_tokens":int((~((all_occ>=.9)|(all_occ<=.1))).sum()),"fraction_mixed_tokens":metrics.mean(~((all_occ>=.9)|(all_occ<=.1))),"ww_mixed_pixels":merge_stats(pixel_store[(name,"pixel","WW_MIXED")])["num"],"ww_mixed_fraction":merge_stats(pixel_store[(name,"pixel","WW_MIXED")])["num"]/merge_stats(pixel_store[(name,"pixel","WW")])["num"]}
    summary={"baseline_reproduced":True,"validation_images":17,"epochs":args.epochs_list,"memory_leakage":False,"uni224":{"input_resolution":224,"native_grid":"14x14","WW_native":nt("UNI224","WW"),"HC90WW_native":nt("UNI224","HC90WW"),"Shared_FP_native":nt("UNI224","SHARED_FP"),"Shared_FN_native":nt("UNI224","SHARED_FN"),"CC_native":nt("UNI224","CC"),"WW_pixel":px("UNI224","WW"),"HC90WW_pixel":px("UNI224","HC90WW"),"Shared_FP_pixel":px("UNI224","SHARED_FP"),"Shared_FN_pixel":px("UNI224","SHARED_FN"),"CC_pixel":px("UNI224","CC"),"WW_boundary_0_4":px("UNI224","WW_BOUNDARY_0_4"),"HC90WW_boundary_0_4":px("UNI224","HC90WW_BOUNDARY_0_4"),"WW_event":event_summary["UNI224"]["WW"],"HC90WW_event":event_summary["UNI224"]["HC90WW"],"WW_MIXED":px("UNI224","WW_MIXED"),"HC90WW_MIXED":px("UNI224","HC90WW_MIXED")},"uni256":{"input_resolution":256,"native_grid":"16x16","WW_native":nt("UNI256","WW"),"HC90WW_native":nt("UNI256","HC90WW"),"Shared_FP_native":nt("UNI256","SHARED_FP"),"Shared_FN_native":nt("UNI256","SHARED_FN"),"CC_native":nt("UNI256","CC"),"WW_pixel":px("UNI256","WW"),"HC90WW_pixel":px("UNI256","HC90WW"),"Shared_FP_pixel":px("UNI256","SHARED_FP"),"Shared_FN_pixel":px("UNI256","SHARED_FN"),"CC_pixel":px("UNI256","CC"),"WW_boundary_0_4":px("UNI256","WW_BOUNDARY_0_4"),"HC90WW_boundary_0_4":px("UNI256","HC90WW_BOUNDARY_0_4"),"WW_event":event_summary["UNI256"]["WW"],"HC90WW_event":event_summary["UNI256"]["HC90WW"],"WW_MIXED":px("UNI256","WW_MIXED"),"HC90WW_MIXED":px("UNI256","HC90WW_MIXED")},"delta_256_minus_224":{},"event_flip":{"fixed_by_256":fixed,"broken_by_256":broken,"net_gain":fixed-broken,"all_event_types_fixed":fixed_all,"all_event_types_broken":broken_all},"mixed_prevalence":mixed_counts,"paired_bootstrap_1000":bootstrap,"memory_224":meta224,"memory_256":meta256,"cps_prediction_source":"frozen standard CPS current-weight milestones; validation predictions from E1 records","output_directory":str(out)}
    for k in summary["uni224"]:
        if k != "input_resolution" and k in summary["uni256"] and isinstance(summary["uni224"][k], (int, float)) and isinstance(summary["uni256"][k], (int, float)):
            summary["delta_256_minus_224"][k]=summary["uni256"][k]-summary["uni224"][k]
    config={"task":"closed E3 UNI input resolution diagnostic","cps_source":str((ROOT/args.prediction_root).resolve()),"epochs":args.epochs_list,"validation_manifest":str(eval_manifest),"validation_image_count":17,"memory_224":str(mem224),"memory_256":str(mem256),"memory_source_manifest":str(source_manifest),"memory_leakage":False,"uni_weights":str(Path(args.uni_weights).resolve()),"uni_weights_sha256":sha256(Path(args.uni_weights)),"preprocessing_224":"aligned RGB256 -> bilinear resize 224 -> same UNI mean/std normalization -> UNI dynamic image size -> 14x14","preprocessing_256":"aligned RGB256 -> no spatial resize -> same UNI mean/std normalization -> UNI dynamic image size -> 16x16","feature_dim":1024,"top_k":5,"occupancy_thresholds":{"fg":.9,"bg":.1},"projection":"margin-first bilinear align_corners=False to 256x256","forbidden":"no training, no test split, no CPS/UNI/memory/similarity/margin/state modification"}
    (out/"config.json").write_text(json.dumps(safe(config),indent=2)+"\n"); (out/"summary.json").write_text(json.dumps(safe(summary),indent=2)+"\n")
    fig,ax=plt.subplots(figsize=(8,4)); labels=["WW","HC90WW","WW boundary","HC90WW boundary","WW event","HC90WW event"]; v224=[summary["uni224"]["WW_pixel"],summary["uni224"]["HC90WW_pixel"],summary["uni224"]["WW_boundary_0_4"],summary["uni224"]["HC90WW_boundary_0_4"],summary["uni224"]["WW_event"],summary["uni224"]["HC90WW_event"]]; v256=[summary["uni256"]["WW_pixel"],summary["uni256"]["HC90WW_pixel"],summary["uni256"]["WW_boundary_0_4"],summary["uni256"]["HC90WW_boundary_0_4"],summary["uni256"]["WW_event"],summary["uni256"]["HC90WW_event"]]; x=np.arange(len(labels)); ax.plot(x,v224,"o-",label="UNI224"); ax.plot(x,v256,"o-",label="UNI256"); ax.set_xticks(x,labels,rotation=25,ha="right"); ax.set_ylim(0,1); ax.legend(); fig.tight_layout(); (out/"plots").mkdir(exist_ok=True); fig.savefig(out/"plots"/"uni224_vs_uni256.png",dpi=150); plt.close(fig)
    print("=== E3 UNI-224 vs UNI-256 Diagnostic ==="); print("Baseline reproduced: yes");
    for name in RESOLUTIONS:
        key = name.lower()
        print(name, {k:summary[key][k] for k in ("native_grid","WW_native","HC90WW_native","WW_pixel","HC90WW_pixel","WW_boundary_0_4","HC90WW_boundary_0_4","WW_event","HC90WW_event")})
    print("Delta 256-224:", {k:summary["delta_256_minus_224"].get(k) for k in ("WW_pixel","HC90WW_pixel","WW_boundary_0_4","HC90WW_boundary_0_4","WW_event","HC90WW_event")}); print("Events fixed by 256:",fixed); print("Events broken by 256:",broken); print("Net event gain:",fixed-broken); print("Output directory:",out)


if __name__ == "__main__": main()
