"""Closed UNI native-token arbitration diagnostic; never changes CPS or memory."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = str(Path(__file__).resolve().parent)
if SCRIPT_DIR in sys.path: sys.path.remove(SCRIPT_DIR)
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from analysis.uni_token_arbitration import metrics, visualize
from analysis.uni_token_arbitration.utils import (load_memory, load_students, manifest_ids, reconstruct_rgb, score_native, sha256, uni_tokens_native)
from config.augmentation.online_aug import data_normalize_2d, data_transform_2d
from config.dataset_config.dataset_cfg import dataset_cfg
from dataload.dataset_2d import imagefloder_itn
from models.uni import load_uni

UNI_WEIGHTS = "/home/yangorange/.cache/huggingface/hub/models--MahmoodLab--UNI/snapshots/b55a5ec6cade1a39edfe6534189a9b8ca7a022f0/pytorch_model.bin"
UNI_CONFIG = str(Path(UNI_WEIGHTS).with_name("config.json"))
UNI_SHA = "56ef09b44a25dc5c7eedc55551b3d47bcd17659a7a33837cf9abc9ec4e2ffb40"


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: raise RuntimeError(f"No rows for {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def json_safe(value):
    if isinstance(value, dict): return {str(k): json_safe(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [json_safe(v) for v in value]
    if isinstance(value, (np.floating,float)): return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer,)): return int(value)
    return value


def logits(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def create_records(image_id, epoch, occupancy, p_a, p_b, s_fg, s_bg):
    records = []
    h, w = occupancy.shape
    for y in range(h):
        for x in range(w):
            occ, pa, pb, fg, bg = (float(occupancy[y,x]), float(p_a[y,x]), float(p_b[y,x]), float(s_fg[y,x]), float(s_bg[y,x]))
            pure_fg, pure_bg = occ >= create_records.fg_threshold, occ <= create_records.bg_threshold
            pure = pure_fg or pure_bg; mixed = not pure
            gt = "FG" if pure_fg else "BG" if pure_bg else "MIXED"
            ha, hb = int(pa >= .5), int(pb >= .5)
            ca = pure and ha == (gt == "FG"); cb = pure and hb == (gt == "FG")
            agreement = pure and ha == hb; disagreement = pure and ha != hb
            cc, ww = bool(ca and cb), bool((not ca) and (not cb) and pure)
            ac, bc = bool(ca and not cb), bool((not ca) and cb)
            hc = bool(ww and agreement and max(pa,1-pa) >= .9 and max(pb,1-pb) >= .9)
            shared_fp = bool(ww and agreement and ha == 1 and gt == "BG")
            shared_fn = bool(ww and agreement and ha == 0 and gt == "FG")
            state = "MIXED" if mixed else ("HC90WW" if hc else "CC" if cc else "WW" if ww else "AC" if ac else "BC")
            pred = "FG" if fg > bg else "BG" if bg > fg else "TIE"
            consensus = ha if agreement else None
            s_cur = fg if consensus == 1 else bg if consensus == 0 else float("nan")
            s_opp = bg if consensus == 1 else fg if consensus == 0 else float("nan")
            records.append({
                "image_id": image_id, "epoch": epoch, "token_y": y, "token_x": x,
                "gt_occupancy": occ, "gt_token": gt, "is_pure": pure, "is_mixed": mixed,
                "pA_token": pa, "pB_token": pb, "hard_A": ha, "hard_B": hb,
                "conf_A": max(pa,1-pa), "conf_B": max(pb,1-pb), "state": state,
                "is_CC": cc, "is_WW": ww, "is_HC90WW": hc, "is_SHARED_FP": shared_fp,
                "is_SHARED_FN": shared_fn, "is_AC": ac, "is_BC": bc, "agreement": agreement,
                "disagreement": disagreement, "S_FG": fg, "S_BG": bg, "margin": fg-bg,
                "S_cur": s_cur, "S_opp": s_opp, "D": s_opp-s_cur if np.isfinite(s_cur) else float("nan"),
                "UNI_pred": pred, "UNI_correct": int(pred == gt) if pure and pred != "TIE" else float("nan"),
            })
    return records


def prepare_uni_fields(loader, uni, mem_fg, mem_bg, k, cfg, device, resolution):
    fields = {}
    with torch.inference_mode():
        for batch in loader:
            rgb = reconstruct_rgb(batch["image_rgb"].to(device), tuple(cfg["MEAN"]), tuple(cfg["STD"]))
            tokens, ht, dim = uni_tokens_native(uni, rgb, resolution)
            s_fg, s_bg, margin = score_native(tokens, mem_fg, mem_bg, k, ht)
            assert s_fg.shape[-2:] == (ht, ht) and s_bg.shape[-2:] == (ht, ht) and margin.shape[-2:] == (ht, ht)
            # GT sees the same evaluation resize then the UNI's 224 resize; it
            # is only pooled after that spatially aligned preprocessing.
            gt_224 = F.interpolate(batch["mask"].float().to(device)[:,None], (resolution,resolution), mode="nearest")
            occupancy = F.adaptive_avg_pool2d(gt_224, (ht,ht))[:,0]
            for i, name in enumerate(batch["ID"]):
                fields[Path(str(name)).stem] = {"occupancy": occupancy[i].cpu().numpy(), "S_FG": s_fg[i].cpu().numpy(), "S_BG": s_bg[i].cpu().numpy(), "margin": margin[i].cpu().numpy(), "grid": ht, "dim": dim}
    return fields


def infer_epoch(loader, checkpoint, epoch, fields, device):
    a, b, payload = load_students(checkpoint, device); output=[]
    with torch.inference_mode():
        for batch in loader:
            pa = logits(a(batch["image_rgb"].to(device))).softmax(1)[:,1]
            pb = logits(b(batch["image_stain"].to(device))).softmax(1)[:,1]
            grid = next(iter(fields.values()))["grid"]
            pa = F.adaptive_avg_pool2d(pa[:,None], (grid,grid))[:,0].cpu().numpy()
            pb = F.adaptive_avg_pool2d(pb[:,None], (grid,grid))[:,0].cpu().numpy()
            for i,name in enumerate(batch["ID"]):
                image_id=Path(str(name)).stem; f=fields[image_id]
                output.extend(create_records(image_id,epoch,f["occupancy"],pa[i],pb[i],f["S_FG"],f["S_BG"]))
    return output, payload


def compact(summary, output):
    def show(key, count_key="num_tokens"):
        x=summary[key]; return x[count_key], x["uni_accuracy"], x["D_positive_rate"], x["mean_D"]
    pure, cc, ww, hc, fp, fn = (summary[k] for k in ("ALL_PURE","CC","WW","HC90WW","SHARED_FP","SHARED_FN"))
    print("=== UNI Native Token Arbitration Diagnostic ===\n")
    print(f"Native grid: {summary['native_grid']}\nFeature dim: {summary['feature_dim']}\nImages: {summary['num_eval_images']}\n")
    print(f"PURE:\n  token acc: {pure['uni_accuracy']:.4f}\n  balanced acc: {pure['balanced_accuracy']:.4f}\n")
    print(f"CC:\n  tokens: {cc['num_tokens']}\n  arbitration acc: {cc['uni_accuracy']:.4f}\n  mean D: {cc['mean_D']:.4f}\n")
    print(f"WW:\n  tokens: {ww['num_tokens']}\n  arbitration acc: {ww['uni_accuracy']:.4f}\n  D>0 rate: {ww['D_positive_rate']:.4f}\n  mean D: {ww['mean_D']:.4f}\n")
    print(f"HC90WW:\n  tokens: {hc['num_tokens']}\n  images: {hc['num_images']}\n  arbitration acc: {hc['uni_accuracy']:.4f}\n  D>0 rate: {hc['D_positive_rate']:.4f}\n  mean D: {hc['mean_D']:.4f}\n")
    print(f"Shared FP:\n  arbitration acc: {fp['uni_accuracy']:.4f}\n\nShared FN:\n  arbitration acc: {fn['uni_accuracy']:.4f}\n")
    print(f"Mixed:\n  Spearman(margin, occupancy): {summary['mixed_margin_occupancy_spearman']:.4f}\n")
    print(f"Memory leakage: {summary['memory_leakage_detected']}\nCPS prediction source: {summary['cps_prediction_source']}\n\nOutputs:\n{output}")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-name",default="GlaS",choices=["GlaS"]); ap.add_argument("--dataset-root",default="datasets/GLaS")
    ap.add_argument("--checkpoint-dir",required=True); ap.add_argument("--memory-root",required=True)
    ap.add_argument("--epochs-list",nargs="+",type=int,default=[40,80,120,160,200]); ap.add_argument("--run-name",default="glas_uni_native_token")
    ap.add_argument("--output-root",default="outputs/uni_token_arbitration"); ap.add_argument("--eval-manifest",default=None)
    ap.add_argument("--fg-occ-thresh",type=float,default=.9); ap.add_argument("--bg-occ-thresh",type=float,default=.1); ap.add_argument("--top-k",type=int,default=None)
    ap.add_argument("--uni-weights",default=UNI_WEIGHTS); ap.add_argument("--uni-config",default=UNI_CONFIG); ap.add_argument("--device",default="cuda:0")
    ap.add_argument("--allow-memory-leak-debug",action="store_true")
    args=ap.parse_args(); create_records.fg_threshold=args.fg_occ_thresh; create_records.bg_threshold=args.bg_occ_thresh
    if not 0 <= args.bg_occ_thresh < args.fg_occ_thresh <= 1: raise ValueError("Require 0 <= bg < fg <= 1")
    device=torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available(): raise RuntimeError("UNI diagnostic requires an available CUDA device")
    dataset_root=(ROOT/args.dataset_root).resolve(); manifests=dataset_root/"manifests/paper_seed_2026"
    eval_manifest=Path(args.eval_manifest).resolve() if args.eval_manifest else manifests/"validation.csv"
    memory_path=Path(args.memory_root).resolve(); checkpoint_dir=Path(args.checkpoint_dir).resolve(); out=(ROOT/args.output_root/args.run_name).resolve(); out.mkdir(parents=True,exist_ok=True)
    mem_fg,mem_bg,mem_meta=load_memory(memory_path,device); source_ratio=int(mem_meta.get("source_labeled_ratio",10)); source_manifest=manifests/f"labeled_{source_ratio}.csv"
    if not source_manifest.is_file(): raise RuntimeError(f"Cannot audit inferred memory source manifest: {source_manifest}")
    leakage=manifest_ids(eval_manifest).intersection(manifest_ids(source_manifest))
    if leakage and not args.allow_memory_leak_debug: raise RuntimeError("ERROR:\nEvaluation image was used to construct UNI reference memory.\nToken-level arbitration result would be leaked.\nUse fold-specific / train-only memory.\nLeaked IDs: "+", ".join(sorted(leakage)))
    k=int(mem_meta["top_k"]) if args.top_k is None else args.top_k
    if k != int(mem_meta["top_k"]): raise RuntimeError(f"--top-k={k} differs from immutable memory top_k={mem_meta['top_k']}")
    resolution=int(mem_meta.get("reference_resolution",224)); cfg=dataset_cfg(args.dataset_name)
    dataset=imagefloder_itn(manifest_path=str(eval_manifest),data_transform_1=data_transform_2d()["val"],data_normalize_1=data_normalize_2d(cfg["MEAN"],cfg["STD"]),sup=True,input_mode="rgb_stain",training=False)
    loader=DataLoader(dataset,batch_size=2,shuffle=False,num_workers=0)
    uni,_,_=load_uni(args.uni_weights,args.uni_config,expected_sha256=UNI_SHA)
    fields=prepare_uni_fields(loader,uni,mem_fg,mem_bg,k,cfg,device,resolution)
    grid={f["grid"] for f in fields.values()}; dims={f["dim"] for f in fields.values()}
    if len(grid)!=1 or len(dims)!=1: raise RuntimeError("UNI native token shape varied across images")
    all_records=[]; used_checkpoints={}
    for epoch in args.epochs_list:
        checkpoint=checkpoint_dir/"milestones"/f"epoch_{epoch:03d}.pth"
        if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
        records,payload=infer_epoch(loader,checkpoint,epoch,fields,device); all_records.extend(records); used_checkpoints[str(epoch)]={"path":str(checkpoint),"sha256":sha256(checkpoint),"input_mode":payload.get("input_mode"),"selection_split":payload.get("selection_split")}
    for record in all_records: record["INVALID_FOR_SCIENTIFIC_CONCLUSION"] = bool(leakage)
    write_csv(out/"token_records.csv",all_records)
    subset_rows=[metrics.calculate(all_records,s) for s in metrics.SUBSETS]; pearson,spearman,mixed_bins=metrics.mixed_stats(all_records); subset_rows.extend(mixed_bins); write_csv(out/"subset_metrics.csv",subset_rows)
    images=metrics.image_rows(all_records); epochs=metrics.epoch_rows(all_records); write_csv(out/"image_metrics.csv",images); write_csv(out/"epoch_metrics.csv",epochs)
    lookup={row["subset"]:row for row in subset_rows}
    summary={"native_token_grid": f"{next(iter(grid))}x{next(iter(grid))}","feature_dim":next(iter(dims)),"num_eval_images":len(fields),"num_pure_tokens":lookup["ALL_PURE"]["num_tokens"],"num_mixed_tokens":lookup["MIXED"]["num_tokens"],"uni_pure_token_accuracy":lookup["ALL_PURE"]["uni_accuracy"],"uni_pure_token_balanced_accuracy":lookup["ALL_PURE"]["balanced_accuracy"],"cc_arbitration_accuracy":lookup["CC"]["uni_accuracy"],"cc_contradiction_rate":lookup["CC"]["contradiction_rate"],"ww_arbitration_accuracy":lookup["WW"]["uni_accuracy"],"hc90ww_arbitration_accuracy":lookup["HC90WW"]["uni_accuracy"],"shared_fp_arbitration_accuracy":lookup["SHARED_FP"]["uni_accuracy"],"shared_fn_arbitration_accuracy":lookup["SHARED_FN"]["uni_accuracy"],"mean_D_CC":lookup["CC"]["mean_D"],"mean_D_WW":lookup["WW"]["mean_D"],"mean_D_HC90WW":lookup["HC90WW"]["mean_D"],"ww_D_positive_rate":lookup["WW"]["D_positive_rate"],"hc90ww_D_positive_rate":lookup["HC90WW"]["D_positive_rate"],"mixed_margin_occupancy_pearson":pearson,"mixed_margin_occupancy_spearman":spearman,"memory_leakage_detected":bool(leakage),"memory_leakage_ids":sorted(leakage),"cps_prediction_source":"frozen existing CPS current-weight milestones; validation inference","image_macro_bootstrap_1000":{"cc":metrics.bootstrap_image_ci(images,"uni_accuracy_cc","cc"),"ww":metrics.bootstrap_image_ci(images,"uni_accuracy_ww","ww"),"hc90ww":metrics.bootstrap_image_ci(images,"uni_accuracy_hc90ww","hc90ww")}}
    config={"task":"closed UNI native token-level arbitration diagnostic","dataset":args.dataset_name,"evaluation_manifest":str(eval_manifest),"evaluation_image_ids":sorted(fields),"memory_path":str(memory_path),"memory_sha256":sha256(memory_path),"memory_source_manifest_inferred":str(source_manifest),"memory_metadata":mem_meta,"memory_leakage_detected":bool(leakage),"invalid_for_scientific_conclusion":bool(leakage),"cps_prediction_source":summary["cps_prediction_source"],"checkpoints":used_checkpoints,"uni":{"weights":str(Path(args.uni_weights).resolve()),"weights_sha256":sha256(Path(args.uni_weights)),"architecture":"vit_large_patch16_224","input_resolution":resolution,"patch_size":16,"native_grid":summary["native_token_grid"],"feature_dim":summary["feature_dim"]},"gt_mapping":{"fg_occ_threshold":args.fg_occ_thresh,"bg_occ_threshold":args.bg_occ_thresh,"preprocess":"evaluation resize 256 then nearest resize to UNI input resolution; adaptive_avg_pool2d to native grid"},"similarity":"exact project main._topk_scores: normalized query @ normalized memory.T then mean largest K","top_k":k,"margin":"S_FG - S_BG","forbidden_operations":"no score/margin interpolation; no R; no training; no CPS modification"}
    (out/"config.json").write_text(json.dumps(json_safe(config),indent=2,allow_nan=False)+"\n",encoding="utf-8"); (out/"summary.json").write_text(json.dumps(json_safe(summary),indent=2,allow_nan=False)+"\n",encoding="utf-8")
    visualize.plots(all_records,args.epochs_list,out/"plots"); visualize.token_maps(all_records,out/"plots")
    summary_for_print={**lookup,"native_grid":summary["native_token_grid"],"feature_dim":summary["feature_dim"],"num_eval_images":summary["num_eval_images"],"mixed_margin_occupancy_spearman":spearman,"memory_leakage_detected":bool(leakage),"cps_prediction_source":summary["cps_prediction_source"]}; compact(summary_for_print,out)


if __name__=="__main__": main()
