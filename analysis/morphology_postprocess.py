#!/usr/bin/env python3
"""Closed morphology-only post-processing diagnostic for saved test masks."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import disk, opening, remove_small_holes, remove_small_objects


ROOT = Path(__file__).resolve().parents[1]
TEST_RESULTS = ROOT / "test_results"
OUTPUT = TEST_RESULTS / "postprocessed"
VIS_DIR = TEST_RESULTS / "postprocessed_visualization"
MIN_SIZES = (0, 16, 32, 64, 128)
HOLE_SIZES = (0, 16, 32, 64)
OPENING_RADII = (0, 1, 2)
THRESHOLD = 0.5
SELECTION_STATUS = "POST_HOC_TEST_DIAGNOSTIC_ONLY"


def image_files(directory: Path) -> list[Path]:
    exts = {".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg"}
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in exts)


def discover_groups() -> list[dict]:
    groups = []
    for pred_dir in sorted(TEST_RESULTS.rglob("predictions")):
        if OUTPUT in pred_dir.parents or VIS_DIR in pred_dir.parents:
            continue
        preds = image_files(pred_dir)
        if not preds:
            continue
        group_dir = pred_dir.parent.parent
        rel = group_dir.relative_to(TEST_RESULTS)
        dataset = rel.parts[0]
        group_name = rel.parts[-1]
        seed_match = re.search(r"seed_(\d+)", group_name)
        ratio_match = re.search(r"(\d+(?:\.\d+)?)%", group_name)
        if seed_match is None or ratio_match is None:
            raise RuntimeError(f"Cannot identify seed/ratio from prediction group: {group_dir}")
        method = group_name[: seed_match.start()].rstrip("_") or "unknown"
        groups.append({
            "dataset": dataset,
            "method": method,
            "seed": int(seed_match.group(1)),
            "ratio": ratio_match.group(1) + "%",
            "group_dir": group_dir,
            "pred_dir": pred_dir,
            "preds": preds,
        })
    if not groups:
        raise RuntimeError(f"No prediction directories found under {TEST_RESULTS}")
    return groups


def locate_gt_dir(dataset: str, pred_names: set[str]) -> Path:
    base = ROOT / "datasets" / dataset
    candidates = []
    for directory in base.rglob("*"):
        if not directory.is_dir():
            continue
        name = directory.name.lower()
        if name not in {"mask", "masks", "label", "labels", "ground_truth", "gt"}:
            continue
        for subdir in directory.iterdir():
            if not subdir.is_dir():
                continue
            files = image_files(subdir)
            names = {p.name for p in files}
            if names == pred_names:
                candidates.append(subdir)
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one GT directory for {dataset} with exact prediction basenames; "
            f"found {len(candidates)}: {[str(p) for p in candidates]}"
        )
    return candidates[0]


def read_binary(path: Path, is_prediction: bool) -> tuple[np.ndarray, dict]:
    im = Image.open(path)
    arr = np.asarray(im)
    info = {"mode": im.mode, "rgb": arr.ndim == 3, "probability_map": False,
            "encoding_0_255": False, "threshold": None}
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        if not np.all(arr == arr[:, :, :1]):
            raise RuntimeError(f"Non-grayscale RGB mask needs explicit semantics: {path}")
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        raise RuntimeError(f"Expected 2D mask, got shape={arr.shape}: {path}")
    values = np.unique(arr)
    if len(values) > 2 or (len(values) == 2 and not set(values.tolist()).issubset({0, 1, 255})):
        if not is_prediction:
            raise RuntimeError(f"GT is not binary (unique values={values[:12]}): {path}")
        info["probability_map"] = True
        info["threshold"] = THRESHOLD
        numeric = arr.astype(np.float64)
        if numeric.max(initial=0) > 1:
            numeric /= 255.0
        mask = numeric >= THRESHOLD
    else:
        if set(values.tolist()).issubset({0, 255}) and 255 in values:
            info["encoding_0_255"] = True
        mask = arr > 0
    return np.asarray(mask, dtype=bool), info


def scores(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    denom_d = 2 * tp + fp + fn
    denom_i = tp + fp + fn
    dice = 1.0 if denom_d == 0 else 2.0 * tp / denom_d
    iou = 1.0 if denom_i == 0 else tp / denom_i
    return dice, iou


def apply_ops(raw: np.ndarray, min_size: int, hole_size: int, radius: int) -> np.ndarray:
    result = raw.copy()
    if min_size:
        result = remove_small_objects(result, min_size=min_size, connectivity=1)
    if radius:
        result = opening(result, footprint=disk(radius))
    if hole_size:
        result = remove_small_holes(result, area_threshold=hole_size, connectivity=1)
    return np.asarray(result, dtype=bool)


def component_stats(mask: np.ndarray) -> dict:
    structure = ndi.generate_binary_structure(2, 1)
    labels, n = ndi.label(mask, structure=structure)
    sizes = np.bincount(labels.ravel())[1:]
    inv_labels, inv_n = ndi.label(~mask, structure=structure)
    inv_sizes = np.bincount(inv_labels.ravel())
    border_labels = np.unique(np.concatenate((inv_labels[0], inv_labels[-1], inv_labels[:, 0], inv_labels[:, -1])))
    hole_sizes = [int(inv_sizes[i]) for i in range(1, inv_n + 1) if i not in border_labels]
    return {
        "num_foreground_components": int(n),
        "largest_component_area": int(sizes.max()) if len(sizes) else 0,
        "small_component_count_lt32": int(np.count_nonzero(sizes < 32)),
        "hole_count_lt32": int(sum(size < 32 for size in hole_sizes)),
        "foreground_area_ratio": float(mask.mean()),
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_audit_summaries() -> None:
    """Materialize raw distributions, explicit one-factor ablations, and interpretation."""
    with (TEST_RESULTS / "postprocess_per_image.csv").open(newline="", encoding="utf-8") as f:
        per_image = list(csv.DictReader(f))
    with (TEST_RESULTS / "postprocess_grid_summary.csv").open(newline="", encoding="utf-8") as f:
        grid = list(csv.DictReader(f))
    key_fields = ("dataset", "method", "seed", "ratio")
    buckets = defaultdict(list)
    grid_buckets = defaultdict(list)
    for row in per_image:
        buckets[tuple(row[k] for k in key_fields)].append(row)
    for row in grid:
        grid_buckets[tuple(row[k] for k in key_fields)].append(row)
    raw_rows, single_rows = [], []
    for key, rows in buckets.items():
        dices = np.array([float(r["raw_dice"]) for r in rows])
        ious = np.array([float(r["raw_iou"]) for r in rows])
        raw_rows.append({**dict(zip(key_fields, key)), "num_images": len(rows),
                         "mean_dice": float(dices.mean()), "std_dice": float(dices.std()),
                         "median_dice": float(np.median(dices)), "min_dice": float(dices.min()),
                         "max_dice": float(dices.max()), "mean_iou": float(ious.mean()),
                         "std_iou": float(ious.std()), "median_iou": float(np.median(ious)),
                         "min_iou": float(ious.min()), "max_iou": float(ious.max())})
        lookup = {(int(r["min_size"]), int(r["hole_size"]), int(r["opening_radius"])): r
                  for r in grid_buckets[key]}
        requested = [("Raw", 0, 0, 0)]
        requested += [("small_object_only", s, 0, 0) for s in MIN_SIZES[1:]]
        requested += [("small_hole_only", 0, h, 0) for h in HOLE_SIZES[1:]]
        requested += [("opening_only", 0, 0, r) for r in OPENING_RADII[1:]]
        for operation, min_size, hole_size, radius in requested:
            row = lookup[(min_size, hole_size, radius)]
            single_rows.append({**dict(zip(key_fields, key)), "operation": operation,
                                "min_size": min_size, "hole_size": hole_size,
                                "opening_radius": radius,
                                **{field: float(row[field]) for field in
                                   ("mean_dice", "std_dice", "mean_iou", "std_iou",
                                    "delta_mean_dice", "delta_mean_iou", "mean_fg_pixel_change")},
                                "num_improved": int(row["num_improved"]),
                                "num_degraded": int(row["num_degraded"]),
                                "num_equal": int(row["num_equal"])})
    write_csv(TEST_RESULTS / "postprocess_raw_summary.csv", raw_rows)
    write_csv(TEST_RESULTS / "postprocess_single_operation_summary.csv", single_rows)

    op_rows = defaultdict(list)
    best_parameter_counts = defaultdict(int)
    for key, rows in grid_buckets.items():
        raw_row = next(r for r in rows if int(r["min_size"]) == int(r["hole_size"]) == int(r["opening_radius"]) == 0)
        raw_dice = float(raw_row["mean_dice"])
        configs = [("small_object", r) for r in rows if int(r["min_size"]) > 0 and int(r["hole_size"]) == 0 and int(r["opening_radius"]) == 0]
        configs += [("small_hole", r) for r in rows if int(r["min_size"]) == 0 and int(r["hole_size"]) > 0 and int(r["opening_radius"]) == 0]
        configs += [("opening", r) for r in rows if int(r["min_size"]) == 0 and int(r["hole_size"]) == 0 and int(r["opening_radius"]) > 0]
        for op, r in configs:
            op_rows[op].append({"delta": float(r["mean_dice"]) - raw_dice,
                                "fg_change": float(r["mean_fg_pixel_change"])})
    with (TEST_RESULTS / "postprocess_best_summary.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for op, field in (("small_object", "best_min_size"), ("small_hole", "best_hole_size"),
                              ("opening", "best_opening_radius")):
                if int(row[field]) > 0:
                    best_parameter_counts[op] += 1
    mean_gain = {op: float(np.mean([x["delta"] for x in vals])) for op, vals in op_rows.items()}
    best_single = max(mean_gain, key=mean_gain.get)
    opening_pixel = float(np.mean([x["fg_change"] for x in op_rows["opening"]]))
    hole_pixel = float(np.mean([x["fg_change"] for x in op_rows["small_hole"]]))
    report = ["# Morphological post-processing diagnostic", "",
              f"Selection status: **{SELECTION_STATUS}**. All 60 parameter combinations were ranked using test Dice for diagnostic purposes; these values are not unbiased test performance.",
              "", "## Per group", ""]
    with (TEST_RESULTS / "postprocess_best_summary.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            report.append(f"- {row['dataset']} / {row['method']} / seed {row['seed']} / {row['ratio']}: "
                          f"Dice {float(row['raw_mean_dice']):.6f} → {float(row['post_mean_dice']):.6f} "
                          f"(Δ {float(row['delta_dice']):+.6f}); IoU {float(row['raw_mean_iou']):.6f} → {float(row['post_mean_iou']):.6f} "
                          f"(Δ {float(row['delta_iou']):+.6f}); improved {row['improved_images']}, degraded {row['degraded_images']}; "
                          f"parameters ({row['best_min_size']}, {row['best_hole_size']}, {row['best_opening_radius']}).")
    report += ["", "## Single-operation findings", "",
               f"- Mean Dice change by best single-operation setting, averaged equally across the 14 groups: "
               + "; ".join(f"{op} {value:+.6f}" for op, value in sorted(mean_gain.items(), key=lambda x: x[1], reverse=True)) + ".",
               f"- Best single-operation category by this diagnostic is **{best_single}**. Across groupwise best configurations, nonzero parameter counts are "
               + ", ".join(f"{op} {count}/{len(buckets)}" for op, count in best_parameter_counts.items()) + ".",
               f"- Opening-only settings change foreground by {opening_pixel:+.1f} pixels per image on average (negative means erosion); small-hole-only settings change it by {hole_pixel:+.1f} pixels per image (positive means inflation).",
               "- Mean Dice and IoU, improved/degraded image counts, and groupwise deltas are in `postprocess_best_summary.csv`; detailed single-factor values are in `postprocess_single_operation_summary.csv`.",
               "", "All original prediction and GT files were hash-checked before and after processing. Shape alignment used the repository's deterministic nearest-neighbor 256×256 test mask resize in memory. Sanity checks passed.", ""]
    (TEST_RESULTS / "postprocess_report.md").write_text("\n".join(report), encoding="utf-8")


def csv_baseline(group: dict) -> dict[str, dict[str, float]]:
    path = group["group_dir"] / "test" / "per_image_metrics.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for row in rows:
        key = Path(row.get("image", row.get("image_id", ""))).stem
        if key:
            out[key] = {"dice": float(row["dice"]), "iou": float(row.get("jaccard", row.get("iou")))}
    return out


def make_visuals(items: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    VIS_DIR.mkdir(parents=True, exist_ok=True)
    categories = [
        ("largest_dice_gains", sorted(items, key=lambda x: x["delta_dice"], reverse=True)[:5]),
        ("largest_dice_losses", sorted(items, key=lambda x: x["delta_dice"])[:5]),
        ("lowest_raw_dice", sorted(items, key=lambda x: x["raw_dice"])[:5]),
    ]
    for category, chosen in categories:
        for rank, item in enumerate(chosen, 1):
            gt, raw, post = item["gt_mask"], item["raw_mask"], item["post_mask"]
            removed = raw & ~post
            added = post & ~raw
            diff = np.zeros((*raw.shape, 3), dtype=float)
            diff[removed] = (0.95, 0.15, 0.15)
            diff[added] = (0.10, 0.35, 1.00)
            fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), constrained_layout=True)
            axes[0].imshow(gt, cmap="gray", vmin=0, vmax=1)
            axes[0].set_title("GT")
            axes[1].imshow(raw, cmap="gray", vmin=0, vmax=1)
            axes[1].set_title(f"Raw (Dice {item['raw_dice']:.4f})")
            axes[2].imshow(post, cmap="gray", vmin=0, vmax=1)
            axes[2].set_title(f"Post (Dice {item['post_dice']:.4f})")
            axes[3].imshow(diff)
            axes[3].set_title("Difference: removed red / added blue")
            for ax in axes:
                ax.axis("off")
            label = f"{item['dataset']}_{item['method']}_s{item['seed']}_{item['ratio']}_{item['image_id']}"
            fig.suptitle(f"{category} #{rank}: {label}", fontsize=9)
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
            fig.savefig(VIS_DIR / f"{category}_{rank}_{safe}.png", dpi=160)
            plt.close(fig)


def main() -> None:
    groups = discover_groups()
    print("Detected groups:")
    for g in groups:
        print(f"  {g['dataset']}/{g['group_dir'].name}: predictions={len(g['preds'])}, dir={g['pred_dir']}")

    # Full exact-basename correspondence check is completed for every group before processing.
    for group in groups:
        pred_names = {p.name for p in group["preds"]}
        if len(pred_names) != len(group["preds"]):
            raise RuntimeError(f"Duplicate prediction basename in {group['pred_dir']}")
        gt_dir = locate_gt_dir(group["dataset"], pred_names)
        gt_files = image_files(gt_dir)
        gt_by_name = {p.name: p for p in gt_files}
        if len(gt_by_name) != len(gt_files) or set(gt_by_name) != pred_names:
            raise RuntimeError(f"Prediction/GT basename mismatch: {group['pred_dir']} vs {gt_dir}")
        group["gt_dir"] = gt_dir
        group["gt_by_name"] = gt_by_name
        group["baseline"] = csv_baseline(group)
        if group["baseline"] and set(group["baseline"]) != {p.stem for p in group["preds"]}:
            raise RuntimeError(f"per_image_metrics.csv mismatch in {group['group_dir']}")

    # Independent validation prediction+GT pairs are required for non-post-hoc selection.
    validation_prediction_dirs = [p for p in TEST_RESULTS.rglob("predictions")
                                  if "validation" in str(p).lower() and OUTPUT not in p.parents]
    selection_source = "test"
    if validation_prediction_dirs:
        # This diagnostic intentionally supports validation selection only after complete,
        # explicit validation prediction/GT discovery; fail closed if such data are present.
        raise RuntimeError("Validation prediction directories detected but not yet included in the closed test-only input groups; refusing test selection")

    snapshots = {p: sha256(p) for g in groups for p in g["preds"]}
    snapshots.update({g["gt_by_name"][p.name]: sha256(g["gt_by_name"][p.name])
                      for g in groups for p in g["preds"]})

    grid_rows, best_rows, per_image_rows, component_rows = [], [], [], []
    all_visual_items = []
    per_group_scores = {}
    diagnostic_ops = defaultdict(list)
    per_group_best_single = {}
    all_sanity = {"basename_match": True, "shape_match": True, "binary_masks": True,
                  "dice_iou_identity": True, "raw_log_match": True,
                  "identity_config": True, "object_filter_nonincreasing": True,
                  "hole_fill_non_decreasing": True, "opening_no_material_increase": True,
                  "finite_outputs": True}
    mask_notes = defaultdict(int)

    for group in groups:
        cases = []
        for pred_path in group["preds"]:
            gt_path = group["gt_by_name"][pred_path.name]
            raw, pred_info = read_binary(pred_path, True)
            gt, gt_info = read_binary(gt_path, False)
            if raw.shape != gt.shape:
                # The repository's explicit test transform uses Albumentations A.Resize(256,256).
                # Its mask path uses OpenCV INTER_NEAREST; reproduce that deterministic transform
                # in memory so the saved test prediction is compared at the original eval size.
                if raw.shape == (256, 256):
                    gt = cv2.resize(gt.astype(np.uint8), (raw.shape[1], raw.shape[0]),
                                    interpolation=cv2.INTER_NEAREST).astype(bool)
                    mask_notes["gt_resized_to_prediction_shape_in_memory"] += 1
                if raw.shape != gt.shape:
                    raise RuntimeError(f"Shape mismatch after repository test resize: {pred_path} {raw.shape} vs {gt_path} {gt.shape}")
            mask_notes["prediction_rgb"] += int(pred_info["rgb"])
            mask_notes["gt_rgb"] += int(gt_info["rgb"])
            mask_notes["prediction_0_255"] += int(pred_info["encoding_0_255"])
            mask_notes["gt_0_255"] += int(gt_info["encoding_0_255"])
            mask_notes["probability_maps"] += int(pred_info["probability_map"])
            rd, ri = scores(raw, gt)
            if abs(rd - (2 * ri / (1 + ri))) > 1e-10:
                all_sanity["dice_iou_identity"] = False
            old = group["baseline"].get(pred_path.stem)
            if old and (abs(rd - old["dice"]) > 1e-6 or abs(ri - old["iou"]) > 1e-6):
                all_sanity["raw_log_match"] = False
            cases.append({"pred_path": pred_path, "gt_path": gt_path, "raw": raw, "gt": gt,
                          "raw_dice": rd, "raw_iou": ri, "pred_info": pred_info})

        config_scores = {}
        raw_dice_values = np.array([c["raw_dice"] for c in cases])
        raw_iou_values = np.array([c["raw_iou"] for c in cases])
        combos = list(itertools.product(MIN_SIZES, HOLE_SIZES, OPENING_RADII))
        if len(combos) != 60:
            raise RuntimeError(f"Unexpected grid size: {len(combos)}")
        for min_size, hole_size, radius in combos:
            per_case = []
            for case in cases:
                raw = case["raw"]
                processed = apply_ops(raw, min_size, hole_size, radius)
                d, iou = scores(processed, case["gt"])
                per_case.append((processed, d, iou))
            ds = np.array([x[1] for x in per_case])
            is_ = np.array([x[2] for x in per_case])
            key = (min_size, hole_size, radius)
            config_scores[key] = per_case
            grid_rows.append({
                **{k: group[k] for k in ("dataset", "method", "seed", "ratio")},
                "min_size": min_size, "hole_size": hole_size, "opening_radius": radius,
                "mean_dice": float(ds.mean()), "std_dice": float(ds.std(ddof=0)),
                "mean_iou": float(is_.mean()), "std_iou": float(is_.std(ddof=0)),
                "delta_mean_dice": float(ds.mean() - raw_dice_values.mean()),
                "delta_mean_iou": float(is_.mean() - raw_iou_values.mean()),
                "num_improved": int(np.count_nonzero(ds > raw_dice_values + 1e-12)),
                "num_degraded": int(np.count_nonzero(ds < raw_dice_values - 1e-12)),
                "num_equal": int(np.count_nonzero(np.abs(ds - raw_dice_values) <= 1e-12)),
                "mean_fg_pixel_change": float(np.mean([int(x[0].sum()) - int(c["raw"].sum()) for x, c in zip(per_case, cases)])),
            })
        best_mean = max(float(np.mean([x[1] for x in vals])) for vals in config_scores.values())
        eligible = [k for k, vals in config_scores.items()
                    if best_mean - float(np.mean([x[1] for x in vals])) <= 1e-6]
        best_key = min(eligible, key=lambda x: (x[2], x[0], x[1]))
        best_per_case = config_scores[best_key]
        post_ds = np.array([x[1] for x in best_per_case])
        post_is = np.array([x[2] for x in best_per_case])
        raw_md, raw_mi = float(raw_dice_values.mean()), float(raw_iou_values.mean())
        post_md, post_mi = float(post_ds.mean()), float(post_is.mean())
        improved = int(np.count_nonzero(post_ds > raw_dice_values + 1e-12))
        degraded = int(np.count_nonzero(post_ds < raw_dice_values - 1e-12))
        per_group_scores[(group["dataset"], group["method"], group["seed"], group["ratio"])] = {
            "raw_dice": raw_md, "raw_iou": raw_mi, "post_dice": post_md, "post_iou": post_mi,
            "delta_dice": post_md - raw_md, "delta_iou": post_mi - raw_mi,
            "improved": improved, "degraded": degraded,
        }
        best_rows.append({
            **{k: group[k] for k in ("dataset", "method", "seed", "ratio")},
            "raw_mean_dice": raw_md, "raw_mean_iou": raw_mi,
            "best_min_size": best_key[0], "best_hole_size": best_key[1],
            "best_opening_radius": best_key[2], "post_mean_dice": post_md,
            "post_mean_iou": post_mi, "delta_dice": post_md - raw_md,
            "delta_iou": post_mi - raw_mi, "improved_images": improved,
            "degraded_images": degraded, "selection_status": SELECTION_STATUS,
        })

        out_group_dir = OUTPUT / group["group_dir"].relative_to(TEST_RESULTS)
        out_pred_dir = OUTPUT / group["pred_dir"].relative_to(TEST_RESULTS)
        out_pred_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "min_size": best_key[0], "hole_size": best_key[1], "opening_radius": best_key[2],
            "selection_source": selection_source, "selection_status": SELECTION_STATUS,
            "dataset": group["dataset"], "method": group["method"],
            "seed": group["seed"], "ratio": group["ratio"],
            "order": ["remove_small_objects", "opening", "remove_small_holes"],
            "probability_threshold": THRESHOLD,
        }
        (out_group_dir / "postprocess_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        for case, (post, pd, pi) in zip(cases, best_per_case):
            raw, gt = case["raw"], case["gt"]
            Image.fromarray((post.astype(np.uint8) * 255), mode="L").save(out_pred_dir / case["pred_path"].name)
            removed = int(np.count_nonzero(raw & ~post))
            added = int(np.count_nonzero(post & ~raw))
            item = {
                **{k: group[k] for k in ("dataset", "method", "seed", "ratio")},
                "image_id": case["pred_path"].stem,
                "raw_dice": case["raw_dice"], "raw_iou": case["raw_iou"],
                "post_dice": pd, "post_iou": pi,
                "delta_dice": pd - case["raw_dice"], "delta_iou": pi - case["raw_iou"],
                "raw_fg_pixels": int(raw.sum()), "post_fg_pixels": int(post.sum()),
                "removed_fg_pixels": removed, "added_fg_pixels": added,
                "min_size": best_key[0], "hole_size": best_key[1], "opening_radius": best_key[2],
                "selection_status": SELECTION_STATUS,
                "gt_mask": gt, "raw_mask": raw, "post_mask": post,
            }
            per_image_rows.append({k: v for k, v in item.items() if k not in {"gt_mask", "raw_mask", "post_mask"}})
            all_visual_items.append(item)
            for phase, mask in (("raw", raw), ("post", post)):
                component_rows.append({**{k: group[k] for k in ("dataset", "method", "seed", "ratio")},
                                       "image_id": case["pred_path"].stem, "phase": phase,
                                       **component_stats(mask)})

        # Required monotonicity and identity checks are verified over this exact collection.
        for case in cases:
            raw = case["raw"]
            obj_only = apply_ops(raw, best_key[0], 0, 0) if best_key[0] else raw
            hole_only = apply_ops(raw, 0, best_key[1], 0) if best_key[1] else raw
            opening_only = apply_ops(raw, 0, 0, best_key[2]) if best_key[2] else raw
            if obj_only.sum() > raw.sum(): all_sanity["object_filter_nonincreasing"] = False
            if hole_only.sum() < raw.sum(): all_sanity["hole_fill_non_decreasing"] = False
            if opening_only.sum() > raw.sum() + max(32, 0.001 * raw.size):
                all_sanity["opening_no_material_increase"] = False
            if apply_ops(raw, 0, 0, 0).shape != raw.shape or not np.array_equal(apply_ops(raw, 0, 0, 0), raw):
                all_sanity["identity_config"] = False

        # Summarize best single-operation effects by component, for causal interpretation.
        single_keys = {
            "small_object": [(s, 0, 0) for s in MIN_SIZES[1:]],
            "small_hole": [(0, h, 0) for h in HOLE_SIZES[1:]],
            "opening": [(0, 0, r) for r in OPENING_RADII[1:]],
        }
        single_deltas = {}
        for operation, keys in single_keys.items():
            best_single = max(keys, key=lambda k: float(np.mean([x[1] for x in config_scores[k]])))
            delta = float(np.mean([x[1] for x in config_scores[best_single]]) - raw_md)
            diagnostic_ops[operation].append(delta)
            single_deltas[operation] = delta
        best_single_operation = max(single_deltas, key=single_deltas.get)
        per_group_best_single[(group["dataset"], group["method"], group["seed"], group["ratio"])] = best_single_operation
        for item in all_visual_items[-len(cases):]:
            if not np.isfinite([item["raw_dice"], item["raw_iou"], item["post_dice"], item["post_iou"]]).all():
                all_sanity["finite_outputs"] = False

    # Check rendered files are binary and exact source hashes did not change.
    for source, before in snapshots.items():
        if sha256(source) != before:
            raise RuntimeError(f"Source prediction/GT changed during processing: {source}")
    for group in groups:
        out_dir = OUTPUT / group["pred_dir"].relative_to(TEST_RESULTS)
        for path in image_files(out_dir):
            arr = np.asarray(Image.open(path))
            if arr.ndim != 2 or not set(np.unique(arr).tolist()).issubset({0, 255}):
                all_sanity["binary_masks"] = False
            if not np.isfinite(arr).all():
                all_sanity["finite_outputs"] = False

    if not all(all_sanity.values()):
        raise RuntimeError(f"Sanity checks failed: {all_sanity}")

    write_csv(TEST_RESULTS / "postprocess_per_image.csv", per_image_rows)
    write_csv(TEST_RESULTS / "postprocess_grid_summary.csv", grid_rows)
    write_csv(TEST_RESULTS / "postprocess_best_summary.csv", best_rows)
    write_csv(TEST_RESULTS / "postprocess_component_diagnostics.csv", component_rows)
    make_visuals(all_visual_items)

    raw_file_hashes = {str(p.relative_to(ROOT)): h for p, h in snapshots.items()}
    sanity_report = {
        "status": "PASS", "checks": all_sanity,
        "source_prediction_files_unchanged": True,
        "source_gt_files_unchanged": True,
        "source_hashes_sha256": raw_file_hashes,
        "prediction_gt_groups": len(groups),
        "prediction_count": sum(len(g["preds"]) for g in groups),
        "mask_sanity_counts": dict(mask_notes),
        "validation_prediction_directories": validation_prediction_dirs,
        "independent_validation_prediction_and_gt_available": bool(validation_prediction_dirs),
        "selection_source": selection_source,
        "selection_status": SELECTION_STATUS,
        "test_gt_used_for_diagnostic_metrics_only": True,
        "test_gt_used_for_training_or_model_inference": False,
        "raw_metric_tolerance_vs_existing_per_image_csv": 1e-6,
    }
    (TEST_RESULTS / "postprocess_sanity_checks.json").write_text(json.dumps(sanity_report, indent=2) + "\n", encoding="utf-8")

    # A compact results interpretation, directly answering the requested morphology questions.
    lines = ["# Morphological post-processing diagnostic", "",
             f"Selection status: **{SELECTION_STATUS}**. The 60-way search used test Dice for diagnostic ranking; this is not unbiased test performance.",
             "", "## Per group", ""]
    for row in best_rows:
        lines.append(f"- {row['dataset']} / {row['method']} / seed {row['seed']} / {row['ratio']}: "
                     f"Dice {row['raw_mean_dice']:.6f} → {row['post_mean_dice']:.6f} "
                     f"(Δ {row['delta_dice']:+.6f}); IoU {row['raw_mean_iou']:.6f} → {row['post_mean_iou']:.6f} "
                     f"(Δ {row['delta_iou']:+.6f}); improved {row['improved_images']}, degraded {row['degraded_images']}; "
                     f"parameters ({row['best_min_size']}, {row['best_hole_size']}, {row['best_opening_radius']}).")
    lines += ["", "## Morphology effects", ""]
    for op, vals in diagnostic_ops.items():
        lines.append(f"- Best single {op} mean Dice change averaged across groups: {np.mean(vals):+.6f}.")
    lines += ["", "Opening foreground erosion is quantified by removed pixels in per-image rows and foreground pixel changes in the grid; hole filling inflation is quantified the same way. See the CSVs for per-group values.",
              "", "All source masks were SHA-256 checked before/after; sanity checks passed.", ""]
    (TEST_RESULTS / "postprocess_report.md").write_text("\n".join(lines), encoding="utf-8")
    write_audit_summaries()

    print("\n[Post-processing Summary]")
    for row in best_rows:
        print(f"{row['dataset']} / {row['method']} / {row['seed']} / {row['ratio']}")
        print(f"Raw:  Dice={row['raw_mean_dice']:.6f}  IoU={row['raw_mean_iou']:.6f}")
        print(f"Best morphology: min_size={row['best_min_size']} hole_size={row['best_hole_size']} opening_radius={row['best_opening_radius']}")
        print(f"Post: Dice={row['post_mean_dice']:.6f}  IoU={row['post_mean_iou']:.6f}")
        print(f"Delta: Dice={row['delta_dice']:+.6f}  IoU={row['delta_iou']:+.6f}")
        print(f"Images improved={row['improved_images']} degraded={row['degraded_images']}")
        group_key = (row["dataset"], row["method"], row["seed"], row["ratio"])
        print(f"Best single operation: {per_group_best_single[group_key]}")
        print(f"Selection status: {SELECTION_STATUS}")
    print("All sanity checks: PASS")
    print(f"Mask encodings: {dict(mask_notes)}")


if __name__ == "__main__":
    main()
