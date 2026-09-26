"""Closed, no-training Anti-Confirmation intervention-budget diagnostic.

This script intentionally has no optimizer.step() and never writes checkpoint
files.  It compares beta=0.5 and beta=1.0 on exactly the same detached q and
fine-selective risky pixels for four fixed training batches at epochs 40/120/200.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main as project
from config.dataset_config.dataset_cfg import dataset_cfg
from config.uni_config.uni_config import UNI_CONFIG_DEFAULT, UNI_WEIGHTS_DEFAULT
from models.getnetwork import get_network
from loss.loss_function import DiceLoss


OUTPUT = ROOT / "outputs/q_budget_gradient_diagnostic"
EPOCHS = (40, 120, 200)
BATCHES = 4
BATCH_SIZE = 2
SEED = 240921
EPS = 1e-12


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_args(memory_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_name="GlaS",
        path_dataset=str(ROOT / "datasets/GlaS"),
        manifest_dir=str(ROOT / "datasets/GlaS/manifests/paper_seed_2026"),
        input1="image",
        input_mode="rgb_stain",
        labeled_ratio=10,
        sup_mark="9",
        unsup_mark="76",
        batch_size=BATCH_SIZE,
        num_epochs=200,
        network="unet",
        cps_mode="standard",
        reference_mode="native_suppress_d_adaptive_fine_reverse",
        native_suppress_gamma=0.5,
        native_hc_thresh=0.9,
        lambda_native_ref=0.5,
        uni_reference_memory=str(memory_path),
        uni_build_reference_memory=False,
        uni_weights=UNI_WEIGHTS_DEFAULT,
        uni_config=UNI_CONFIG_DEFAULT,
        uni_k=50,
        reference_resolution=256,
        reference_grid=16,
        lr=0.5,
        step_size=50,
        gamma=0.5,
        unsup_weight=5.0,
        warm_up_duration=20,
        momentum=0.9,
        wd=-5.0,
        sd=24,
    )


def discover_paths() -> tuple[Path, dict[int, Path], Path]:
    roots = list((ROOT / "checkpoints/topk50_native_d_adaptive_fine_reverse_seed24/GLaS").glob("*/"))
    if len(roots) != 1:
        raise RuntimeError(f"Expected one k=50 GLaS run directory, found {roots}")
    run_dir = roots[0]
    milestones = {
        epoch: run_dir / "milestones" / f"epoch_{epoch:03d}.pth"
        for epoch in EPOCHS
    }
    missing = [str(path) for path in milestones.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing milestone checkpoint(s): " + ", ".join(missing))
    memory = run_dir / "reference_memory_uni256.npz"
    if not memory.is_file():
        raise FileNotFoundError(memory)
    return run_dir, milestones, memory


def build_training_batches(args: SimpleNamespace, cfg: dict) -> tuple[list[dict], list[dict]]:
    """Materialize four fixed labeled/unlabeled training batches only."""
    manifest_dir = Path(args.manifest_dir)
    transforms = project.data_transform_2d()
    normalize = project.data_normalize_2d(cfg["MEAN"], cfg["STD"])
    common = {
        "data_normalize_1": normalize,
        "input_mode": args.input_mode,
    }
    labeled = manifest_dir / "labeled_10.csv"
    unlabeled = manifest_dir / "unlabeled_10.csv"
    train_unsup_unique = project.imagefloder_itn(
        manifest_path=str(unlabeled),
        **common,
        data_transform_1=transforms["train"],
        sup=False,
        training=True,
    )
    train_sup_unique = project.imagefloder_itn(
        manifest_path=str(labeled),
        **common,
        data_transform_1=transforms["train"],
        sup=True,
        training=True,
    )
    train_sup = project.imagefloder_itn(
        manifest_path=str(labeled),
        **common,
        data_transform_1=transforms["train"],
        sup=True,
        num_images=len(train_unsup_unique),
        training=True,
    )
    sup_loader = DataLoader(
        train_sup, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        pin_memory=True,
    )
    unsup_loader = DataLoader(
        train_unsup_unique, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        pin_memory=True,
    )
    sup_batches = []
    unsup_batches = []
    for index, (sup_batch, unsup_batch) in enumerate(zip(sup_loader, unsup_loader)):
        if index >= BATCHES:
            break
        sup_batches.append({key: value.clone() if torch.is_tensor(value) else value
                            for key, value in sup_batch.items()})
        unsup_batches.append({key: value.clone() if torch.is_tensor(value) else value
                              for key, value in unsup_batch.items()})
    if len(sup_batches) != BATCHES or len(unsup_batches) != BATCHES:
        raise RuntimeError("Unable to materialize exactly four training batches")
    return sup_batches, unsup_batches


def move_tensor_batch(batch: dict) -> dict:
    return {
        key: value.cuda(non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def load_reference_context(args: SimpleNamespace, cfg: dict) -> dict:
    uni, _, _ = project._load_uni(args)
    with np.load(args.uni_reference_memory, allow_pickle=False) as memory:
        fg = np.asarray(memory["memory_fg"], dtype=np.float32)
        bg = np.asarray(memory["memory_bg"], dtype=np.float32)
        grid = int(memory["token_grid"])
    context = {
        "uni": uni,
        "memory_fg": torch.from_numpy(fg).cuda(non_blocking=True),
        "memory_bg": torch.from_numpy(bg).cuda(non_blocking=True),
        "token_grid": grid,
        "reference_resolution": 256,
        "feature_dim": 1024,
        "cfg": cfg,
    }
    for parameter in uni.parameters():
        if parameter.requires_grad:
            raise AssertionError("UNI is not frozen")
    uni.eval()
    return context


def load_models(checkpoint: Path) -> tuple[torch.nn.Module, torch.nn.Module]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_a = get_network("unet", 3, 2).cuda()
    model_b = get_network("unet", 3, 2).cuda()
    model_a.load_state_dict(payload["appearance_state_dict"], strict=True)
    model_b.load_state_dict(payload["structure_state_dict"], strict=True)
    # Evaluation mode freezes BN running statistics and removes stochastic
    # forward behavior, so A/B backward passes see exactly the same state.
    model_a.eval()
    model_b.eval()
    return model_a, model_b


def fixed_reference_targets(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    unsup: dict,
    context: dict,
    args: SimpleNamespace,
) -> dict:
    input_a, input_b = project.get_training_views(unsup)
    with torch.no_grad():
        logits_a = model_a(input_a)[0]
        logits_b = model_b(input_b)[0]
        targets = project._build_native_reference_targets(
            logits_a, logits_b, input_a, context, context["cfg"], args,
        )
        targets["pseudo_a"] = logits_a.argmax(dim=1)
        targets["pseudo_b"] = logits_b.argmax(dim=1)
        targets["opposite_target"] = targets["fine_stats"]["opposite_target_pixel"][:, 0].detach()
        targets["risk"] = targets["fine_stats"]["risk_pixel"][:, 0].detach()
        targets["d_pixel"] = targets["fine_stats"]["d_pixel"][:, 0].detach()
    return targets


def collect_train_positive_d(
    checkpoint: Path,
    unsup_batches: list[dict],
    context: dict,
    args: SimpleNamespace,
) -> np.ndarray:
    """One fixed, no-grad training-side collection for F_train."""
    model_a, model_b = load_models(checkpoint)
    values = []
    with torch.no_grad():
        for batch in unsup_batches:
            unsup = move_tensor_batch(batch)
            targets = fixed_reference_targets(model_a, model_b, unsup, context, args)
            contradiction = targets["contradiction"]
            d = targets["d_positive"]
            positive = d[contradiction].detach().float().cpu().numpy()
            if positive.size:
                values.append(positive)
    del model_a, model_b
    torch.cuda.empty_cache()
    if not values:
        raise RuntimeError("Training-side positive-D collection is empty")
    result = np.sort(np.concatenate(values).astype(np.float64))
    if not np.all(np.isfinite(result)) or np.any(result <= 0):
        raise RuntimeError("Training-side positive-D reference is invalid")
    return result


def empirical_cdf(sorted_positive_d: np.ndarray, d: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(sorted_positive_d, device=d.device, dtype=d.dtype)
    flat = d.reshape(-1)
    ranks = torch.searchsorted(values, flat, right=True)
    return (ranks.to(d.dtype) / float(len(sorted_positive_d))).reshape_as(d).clamp(0.0, 1.0)


def scalar_stats(values: np.ndarray | list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {key: float("nan") for key in ("count", "mean", "std", "p05", "p10", "p25", "p50", "p75", "p90", "p95")}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p05": float(np.percentile(arr, 5)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }


def zero_optimizers(optimizers: list[torch.optim.Optimizer]) -> None:
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)


def group_for(name: str) -> str:
    if name.startswith("Conv_1x1"):
        return "head"
    if name.startswith("Up"):
        return "decoder"
    return "encoder"


def gradient_snapshot(models: tuple[torch.nn.Module, torch.nn.Module]) -> dict:
    result = {}
    for student, model in (("A", models[0]), ("B", models[1])):
        all_values = []
        groups = {"encoder": [], "decoder": [], "head": []}
        max_abs = 0.0
        abs_values = []
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            value = parameter.grad.detach().float()
            flat = value.reshape(-1)
            all_values.append(flat)
            groups[group_for(name)].append(flat)
            max_abs = max(max_abs, float(value.abs().max().item()))
            abs_values.append(value.abs().reshape(-1))
        def norm(chunks):
            return float(torch.cat(chunks).norm().item()) if chunks else 0.0
        result[student] = {
            "total": norm(all_values),
            "encoder": norm(groups["encoder"]),
            "decoder": norm(groups["decoder"]),
            "head": norm(groups["head"]),
            "max_abs": max_abs,
            "mean_abs": float(torch.cat(abs_values).mean().item()) if abs_values else 0.0,
            "finite": all(torch.isfinite(chunk).all().item() for chunk in all_values),
        }
    return result


def gradient_vector(model: torch.nn.Module) -> torch.Tensor:
    chunks = []
    for parameter in model.parameters():
        if parameter.grad is None:
            chunks.append(torch.zeros_like(parameter, dtype=torch.float32).reshape(-1))
        else:
            chunks.append(parameter.grad.detach().float().reshape(-1))
    return torch.cat(chunks)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denominator = a.norm() * b.norm()
    if denominator.item() <= EPS:
        return float("nan")
    return float(torch.dot(a, b).item() / denominator.item())


def loss_components(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    fixed: dict,
    beta: float,
) -> dict:
    pseudo_a = fixed["pseudo_a"]
    pseudo_b = fixed["pseudo_b"]
    risk = fixed["risk"]
    opposite = fixed["opposite_target"]
    if not risk.any().item():
        raise RuntimeError("A fixed diagnostic batch has zero risky pixels")
    nonrisk = ~risk
    current_b = F.cross_entropy(logits_b, pseudo_a, reduction="none")
    current_a = F.cross_entropy(logits_a, pseudo_b, reduction="none")
    opposite_a = F.cross_entropy(logits_a, opposite.long(), reduction="none")
    opposite_b = F.cross_entropy(logits_b, opposite.long(), reduction="none")
    q = fixed["q"]
    risk_mix_a = ((1.0 - q) * current_a + q * opposite_a)[risk]
    risk_mix_b = ((1.0 - q) * current_b + q * opposite_b)[risk]
    loss_cps = 0.5 * (current_a[risk].mean() + current_b[risk].mean())
    loss_opp = 0.5 * (opposite_a[risk].mean() + opposite_b[risk].mean())
    loss_mix = 0.5 * (risk_mix_a.mean() + risk_mix_b.mean())
    loss_risk = float(beta) * loss_mix
    other = logits_a.sum() * 0.0
    if nonrisk.any().item():
        other = 0.5 * (current_a[nonrisk].mean() + current_b[nonrisk].mean())
    return {
        "loss_cps": loss_cps,
        "loss_opp": loss_opp,
        "loss_mix": loss_mix,
        "loss_risk": loss_risk,
        "other_cps": other,
        "current_a": current_a,
        "current_b": current_b,
        "opposite_a": opposite_a,
        "opposite_b": opposite_b,
    }


def evaluate_variant(
    models: tuple[torch.nn.Module, torch.nn.Module],
    optimizers: list[torch.optim.Optimizer],
    sup: dict,
    unsup: dict,
    fixed: dict,
    cfg: dict,
    beta: float,
    epoch: int,
) -> tuple[dict, dict, dict]:
    zero_optimizers(optimizers)
    sup_a, sup_b = project.get_training_views(sup)
    unsup_a, unsup_b = project.get_training_views(unsup)
    sup_mask = sup["mask"]
    sup_logits_a = models[0](sup_a)[0]
    sup_logits_b = models[1](sup_b)[0]
    unsup_logits_a = models[0](unsup_a)[0]
    unsup_logits_b = models[1](unsup_b)[0]
    dice = DiceLoss().cuda()
    sup_loss_a = project.compute_supervised_loss(None, dice, sup_logits_a, sup_mask)
    sup_loss_b = project.compute_supervised_loss(None, dice, sup_logits_b, sup_mask)
    components = loss_components(unsup_logits_a, unsup_logits_b, fixed, beta)
    unsup_weight = 5.0 * float(epoch) / 200.0
    total_loss = sup_loss_a + sup_loss_b + unsup_weight * (
        components["other_cps"] + components["loss_risk"]
    )
    zero_optimizers(optimizers)
    components["loss_risk"].backward(retain_graph=True)
    risk_grad = gradient_snapshot(models)
    risk_vectors = (gradient_vector(models[0]), gradient_vector(models[1]))
    zero_optimizers(optimizers)
    total_loss.backward()
    total_grad = gradient_snapshot(models)
    zero_optimizers(optimizers)
    values = {
        "loss_sup_a": float(sup_loss_a.detach().item()),
        "loss_sup_b": float(sup_loss_b.detach().item()),
        "loss_cps": float(components["loss_cps"].detach().item()),
        "loss_opp": float(components["loss_opp"].detach().item()),
        "loss_mix": float(components["loss_mix"].detach().item()),
        "loss_risk": float(components["loss_risk"].detach().item()),
        "loss_other_cps": float(components["other_cps"].detach().item()),
        "loss_total": float(total_loss.detach().item()),
        "risk_cps_weight_mean": float((beta * (1.0 - fixed["q"])[fixed["risk"]]).mean().item()),
        "risk_opp_weight_mean": float((beta * fixed["q"][fixed["risk"]]).mean().item()),
        "risk_cps_weight_p10": float(torch.quantile(beta * (1.0 - fixed["q"])[fixed["risk"]], 0.10).item()),
        "risk_cps_weight_p50": float(torch.quantile(beta * (1.0 - fixed["q"])[fixed["risk"]], 0.50).item()),
        "risk_cps_weight_p90": float(torch.quantile(beta * (1.0 - fixed["q"])[fixed["risk"]], 0.90).item()),
        "risk_opp_weight_p10": float(torch.quantile(beta * fixed["q"][fixed["risk"]], 0.10).item()),
        "risk_opp_weight_p50": float(torch.quantile(beta * fixed["q"][fixed["risk"]], 0.50).item()),
        "risk_opp_weight_p90": float(torch.quantile(beta * fixed["q"][fixed["risk"]], 0.90).item()),
    }
    return values, risk_grad, total_grad, risk_vectors


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean_rows(rows: list[dict], fields: list[str]) -> dict:
    result = {}
    for field in fields:
        values = [float(row[field]) for row in rows if row.get(field) is not None and math.isfinite(float(row[field]))]
        result[field] = float(np.mean(values)) if values else float("nan")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    args_cli = parser.parse_args()
    out = Path(args_cli.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen UNI/native diagnostic")

    run_dir, checkpoints, memory_path = discover_paths()
    args = make_args(memory_path)
    cfg = dataset_cfg("GlaS")
    sup_batches, unsup_batches = build_training_batches(args, cfg)
    context = load_reference_context(args, cfg)

    # The only q calibration collection: training data, epoch-200 frozen model,
    # four fixed batches, no gradients, no validation/test access.
    train_positive_d = collect_train_positive_d(checkpoints[200], unsup_batches, context, args)
    q_reference_stats = scalar_stats(train_positive_d)

    batch_rows = []
    q_rows = []
    gradient_rows = []
    checkpoint_rows = []
    numerical_flags = set()
    all_q = []
    all_risk = []
    all_weight_a = []
    all_weight_b = []
    isolated_loss_ratios = []
    isolated_grad_ratios = {"A": [], "B": []}
    total_grad_ratios = {"A": [], "B": [], "total": []}
    risk_total_ratios = {"beta05": [], "beta10": []}
    cosines = {"A": [], "B": []}

    for epoch in EPOCHS:
        epoch_rows = []
        for batch_id, (sup_cpu, unsup_cpu) in enumerate(zip(sup_batches, unsup_batches), start=1):
            sup = move_tensor_batch(sup_cpu)
            unsup = move_tensor_batch(unsup_cpu)
            # Fixed mask/q/pseudo targets are computed once from the checkpoint
            # and reused by both intervention budgets.
            target_models = load_models(checkpoints[epoch])
            fixed = fixed_reference_targets(*target_models, unsup, context, args)
            fixed["q"] = empirical_cdf(train_positive_d, fixed["d_pixel"])
            risk = fixed["risk"]
            q_risk = fixed["q"][risk].detach()
            d_risk = fixed["d_pixel"][risk].detach()
            if not risk.any().item():
                raise RuntimeError(f"epoch={epoch} batch={batch_id} has zero risky pixels")
            model_a, model_b = target_models
            optimizer_a = torch.optim.SGD(model_a.parameters(), lr=0.0)
            optimizer_b = torch.optim.SGD(model_b.parameters(), lr=0.0)
            optimizer_pair = [optimizer_a, optimizer_b]
            values_a, risk_a, total_a, vectors_a = evaluate_variant(
                (model_a, model_b), optimizer_pair, sup, unsup, fixed, cfg, 0.5, epoch,
            )
            values_b, risk_b, total_b, vectors_b = evaluate_variant(
                (model_a, model_b), optimizer_pair, sup, unsup, fixed, cfg, 1.0, epoch,
            )
            del optimizer_a, optimizer_b, target_models
            torch.cuda.empty_cache()

            risk_loss_ratio = values_b["loss_risk"] / (values_a["loss_risk"] + EPS)
            risk_grad_ratio_a = risk_b["A"]["total"] / (risk_a["A"]["total"] + EPS)
            risk_grad_ratio_b = risk_b["B"]["total"] / (risk_a["B"]["total"] + EPS)
            total_grad_ratio_a = total_b["A"]["total"] / (total_a["A"]["total"] + EPS)
            total_grad_ratio_b = total_b["B"]["total"] / (total_a["B"]["total"] + EPS)
            total_grad_ratio = math.sqrt(
                total_b["A"]["total"] ** 2 + total_b["B"]["total"] ** 2
            ) / (math.sqrt(total_a["A"]["total"] ** 2 + total_a["B"]["total"] ** 2) + EPS)
            risk_total_a = math.sqrt(risk_a["A"]["total"] ** 2 + risk_a["B"]["total"] ** 2) / (
                math.sqrt(total_a["A"]["total"] ** 2 + total_a["B"]["total"] ** 2) + EPS
            )
            risk_total_b = math.sqrt(risk_b["A"]["total"] ** 2 + risk_b["B"]["total"] ** 2) / (
                math.sqrt(total_b["A"]["total"] ** 2 + total_b["B"]["total"] ** 2) + EPS
            )
            cos_a = cosine(vectors_a[0], vectors_b[0])
            cos_b = cosine(vectors_a[1], vectors_b[1])
            if risk_loss_ratio < 1.8 or risk_loss_ratio > 2.2:
                numerical_flags.add("WARNING_LOSS_SCALING")
            if risk_grad_ratio_a < 1.8 or risk_grad_ratio_a > 2.2 or risk_grad_ratio_b < 1.8 or risk_grad_ratio_b > 2.2:
                numerical_flags.add("WARNING_GRADIENT_SCALING")
            if (math.isfinite(cos_a) and cos_a < 0.999) or (math.isfinite(cos_b) and cos_b < 0.999):
                numerical_flags.add("WARNING_GRADIENT_DIRECTION_CHANGED")
            if total_grad_ratio > 3.0:
                numerical_flags.add("WARNING_TOTAL_GRADIENT_LARGE")
            all_finite = all(
                entry["finite"] for entry in (
                    risk_a["A"], risk_a["B"], total_a["A"], total_a["B"],
                    risk_b["A"], risk_b["B"], total_b["A"], total_b["B"],
                )
            )
            if not all_finite:
                numerical_flags.add("FAIL_NUMERICAL_INSTABILITY")

            risk_fraction = float(risk.float().mean().item())
            row = {
                "epoch": epoch,
                "batch_id": batch_id,
                "risk_pixel_count": int(risk.sum().item()),
                "risk_fraction": risk_fraction,
                "q_mean": float(q_risk.mean().item()),
                "q_std": float(q_risk.std(unbiased=False).item()),
                "q_p10": float(torch.quantile(q_risk, 0.10).item()),
                "q_p50": float(torch.quantile(q_risk, 0.50).item()),
                "q_p90": float(torch.quantile(q_risk, 0.90).item()),
                "loss_cps": values_a["loss_cps"],
                "loss_opp": values_a["loss_opp"],
                "loss_mix": values_a["loss_mix"],
                "loss_risk_beta05": values_a["loss_risk"],
                "loss_risk_beta10": values_b["loss_risk"],
                "risk_grad_norm_A_beta05": risk_a["A"]["total"],
                "risk_grad_norm_B_beta05": risk_a["B"]["total"],
                "risk_grad_norm_A_beta10": risk_b["A"]["total"],
                "risk_grad_norm_B_beta10": risk_b["B"]["total"],
                "risk_grad_ratio_A": risk_grad_ratio_a,
                "risk_grad_ratio_B": risk_grad_ratio_b,
                "risk_grad_cosine_A": cos_a,
                "risk_grad_cosine_B": cos_b,
                "total_grad_norm_A_beta05": total_a["A"]["total"],
                "total_grad_norm_B_beta05": total_a["B"]["total"],
                "total_grad_norm_A_beta10": total_b["A"]["total"],
                "total_grad_norm_B_beta10": total_b["B"]["total"],
                "total_grad_ratio_A": total_grad_ratio_a,
                "total_grad_ratio_B": total_grad_ratio_b,
                "total_grad_ratio": total_grad_ratio,
                "risk_total_grad_ratio_beta05": risk_total_a,
                "risk_total_grad_ratio_beta10": risk_total_b,
                "max_abs_grad_A_beta05": total_a["A"]["max_abs"],
                "max_abs_grad_B_beta05": total_a["B"]["max_abs"],
                "max_abs_grad_A_beta10": total_b["A"]["max_abs"],
                "max_abs_grad_B_beta10": total_b["B"]["max_abs"],
                "loss_B_over_A": risk_loss_ratio,
                "mean_CPS_weight_beta05": values_a["risk_cps_weight_mean"],
                "mean_opp_weight_beta05": values_a["risk_opp_weight_mean"],
                "mean_CPS_weight_beta10": values_b["risk_cps_weight_mean"],
                "mean_opp_weight_beta10": values_b["risk_opp_weight_mean"],
            }
            batch_rows.append(row)
            epoch_rows.append(row)
            q_rows.append({"epoch": epoch, "batch_id": batch_id, **scalar_stats(q_risk.detach().cpu().numpy())})
            for variant, grad, total, values in (("beta0.5", risk_a, total_a, values_a), ("beta1.0", risk_b, total_b, values_b)):
                for student in ("A", "B"):
                    gradient_rows.append({
                        "epoch": epoch,
                        "batch_id": batch_id,
                        "variant": variant,
                        "student": student,
                        "risk_total": grad[student]["total"],
                        "risk_encoder": grad[student]["encoder"],
                        "risk_decoder": grad[student]["decoder"],
                        "risk_head": grad[student]["head"],
                        "risk_max_abs": grad[student]["max_abs"],
                        "total_total": total[student]["total"],
                        "total_encoder": total[student]["encoder"],
                        "total_decoder": total[student]["decoder"],
                        "total_head": total[student]["head"],
                        "total_max_abs": total[student]["max_abs"],
                    })
            q_np = q_risk.detach().cpu().numpy()
            all_q.extend(q_np.tolist())
            all_risk.extend([1.0] * len(q_np))
            all_weight_a.extend((0.5 * (1.0 - q_np)).tolist())
            all_weight_b.extend((1.0 - q_np).tolist())
            isolated_loss_ratios.append(risk_loss_ratio)
            isolated_grad_ratios["A"].append(risk_grad_ratio_a)
            isolated_grad_ratios["B"].append(risk_grad_ratio_b)
            total_grad_ratios["A"].append(total_grad_ratio_a)
            total_grad_ratios["B"].append(total_grad_ratio_b)
            total_grad_ratios["total"].append(total_grad_ratio)
            risk_total_ratios["beta05"].append(risk_total_a)
            risk_total_ratios["beta10"].append(risk_total_b)
            cosines["A"].append(cos_a)
            cosines["B"].append(cos_b)

    q_distribution = scalar_stats(all_q)
    extreme_low = float(np.mean(np.asarray(all_q) < 0.05))
    extreme_high = float(np.mean(np.asarray(all_q) > 0.95))
    low_010 = float(np.mean(np.asarray(all_q) < 0.10))
    high_090 = float(np.mean(np.asarray(all_q) > 0.90))
    warning_q_collapse = extreme_low > 0.80 or extreme_high > 0.80
    if warning_q_collapse:
        numerical_flags.add("WARNING_Q_COLLAPSE")
    if any(not math.isfinite(value) for value in isolated_loss_ratios):
        numerical_flags.add("FAIL_NUMERICAL_INSTABILITY")

    for epoch in EPOCHS:
        rows = [row for row in batch_rows if row["epoch"] == epoch]
        checkpoint_rows.append({
            "epoch": epoch,
            "batches": len(rows),
            "risk_fraction_mean": float(np.mean([row["risk_fraction"] for row in rows])),
            "q_mean": float(np.mean([row["q_mean"] for row in rows])),
            "q_p50_mean": float(np.mean([row["q_p50"] for row in rows])),
            "loss_B_over_A_mean": float(np.mean([row["loss_B_over_A"] for row in rows])),
            "risk_grad_ratio_A_mean": float(np.mean([row["risk_grad_ratio_A"] for row in rows])),
            "risk_grad_ratio_B_mean": float(np.mean([row["risk_grad_ratio_B"] for row in rows])),
            "risk_grad_cosine_A_mean": float(np.mean([row["risk_grad_cosine_A"] for row in rows])),
            "risk_grad_cosine_B_mean": float(np.mean([row["risk_grad_cosine_B"] for row in rows])),
            "total_grad_ratio_mean": float(np.mean([row["total_grad_ratio"] for row in rows])),
            "risk_total_grad_beta05_mean": float(np.mean([row["risk_total_grad_ratio_beta05"] for row in rows])),
            "risk_total_grad_beta10_mean": float(np.mean([row["risk_total_grad_ratio_beta10"] for row in rows])),
        })

    write_csv(out / "batch_level_diagnostic.csv", batch_rows)
    write_csv(out / "checkpoint_summary.csv", checkpoint_rows)
    write_csv(out / "q_distribution.csv", q_rows)
    write_csv(out / "gradient_summary.csv", gradient_rows)

    q_ok = not warning_q_collapse
    scale_ok = all(1.8 <= float(value) <= 2.2 for value in isolated_loss_ratios + isolated_grad_ratios["A"] + isolated_grad_ratios["B"])
    direction_ok = all(float(value) >= 0.999 for value in cosines["A"] + cosines["B"] if math.isfinite(float(value)))
    finite_ok = "FAIL_NUMERICAL_INSTABILITY" not in numerical_flags
    diagnostic = "PASS" if q_ok and scale_ok and direction_ok and finite_ok else "FAIL"
    summary = {
        "protocol": {
            "K": 50,
            "training_performed": False,
            "optimizer_step_performed": False,
            "validation_used_for_q_calibration": False,
            "test_used": False,
            "epochs": list(EPOCHS),
            "batches_per_checkpoint": BATCHES,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "model_mode": "eval (fixed BN state; backward enabled)",
            "checkpoint_run": str(run_dir),
            "memory_path": str(memory_path),
            "q_reference": "epoch-200 frozen model, four fixed unlabeled training batches, positive contradiction-token D only",
            "q_reference_count": int(len(train_positive_d)),
        },
        "q_reference_stats": q_reference_stats,
        "q_risky_stats": q_distribution,
        "risk_fraction": float(np.mean([row["risk_fraction"] for row in batch_rows])),
        "q_extreme_fraction_lt_005": extreme_low,
        "q_extreme_fraction_gt_095": extreme_high,
        "q_fraction_lt_010": low_010,
        "q_fraction_gt_090": high_090,
        "weight_stats_beta_0.5": scalar_stats(all_weight_a),
        "weight_stats_beta_1.0": scalar_stats(all_weight_b),
        "isolated_loss_ratio_B_over_A": scalar_stats(isolated_loss_ratios),
        "isolated_gradient_ratio_A_B": {
            student: scalar_stats(values) for student, values in isolated_grad_ratios.items()
        },
        "isolated_gradient_cosine": {student: scalar_stats(values) for student, values in cosines.items()},
        "total_gradient_ratio_A_B": {key: scalar_stats(values) for key, values in total_grad_ratios.items()},
        "risk_over_total_gradient": {key: scalar_stats(values) for key, values in risk_total_ratios.items()},
        "warnings": sorted(numerical_flags),
        "diagnostic": diagnostic,
        "checks": {
            "q_normal": q_ok,
            "isolated_loss_ratio_1.8_2.2": scale_ok,
            "isolated_gradient_ratio_1.8_2.2": scale_ok,
            "gradient_cosine_ge_0.999": direction_ok,
            "finite_gradients": finite_ok,
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    report = f"""# Anti-Confirmation Budget Gradient Diagnostic

## Protocol

- K=50; UNI frozen; existing k=50 memory; existing native fine-selective risky mask.
- Checkpoints: epoch 40, 120, 200; four fixed training batches each.
- Training performed: false. Optimizer step performed: false.
- Validation/test were not used for q calibration.
- `F_train` was collected once from epoch-200 frozen model on the same four unlabeled training batches, using positive contradiction-token D only.

## Q1 — q distribution

Risk fraction: `{summary['risk_fraction']:.8f}`

| statistic | value |
|---|---:|
| mean | {q_distribution['mean']:.8f} |
| p05 | {q_distribution['p05']:.8f} |
| p10 | {q_distribution['p10']:.8f} |
| p25 | {q_distribution['p25']:.8f} |
| p50 | {q_distribution['p50']:.8f} |
| p75 | {q_distribution['p75']:.8f} |
| p90 | {q_distribution['p90']:.8f} |
| p95 | {q_distribution['p95']:.8f} |
| q < 0.10 | {low_010:.8f} |
| q > 0.90 | {high_090:.8f} |
| q < 0.05 | {extreme_low:.8f} |
| q > 0.95 | {extreme_high:.8f} |

`WARNING_Q_COLLAPSE`: `{warning_q_collapse}`.  q is {'not ' if warning_q_collapse else ''}concentrated at the extremes under the requested 80% rule.

## Q2 — pure intervention-budget scaling

- Isolated risky loss B/A mean: `{np.mean(isolated_loss_ratios):.8f}`.
- Isolated risky gradient norm B/A mean: Student A `{np.mean(isolated_grad_ratios['A']):.8f}`, Student B `{np.mean(isolated_grad_ratios['B']):.8f}`.
- Isolated risky gradient cosine mean: Student A `{np.nanmean(cosines['A']):.8f}`, Student B `{np.nanmean(cosines['B']):.8f}`.

These values test whether beta=1 is only a positive scalar budget change on identical q/risk pixels.

## Q3 — beta=1 numerical behavior

- Full total-gradient B/A mean: Student A `{np.mean(total_grad_ratios['A']):.8f}`, Student B `{np.mean(total_grad_ratios['B']):.8f}`, combined `{np.mean(total_grad_ratios['total']):.8f}`.
- Max-gradient values and batch-level details are in `batch_level_diagnostic.csv` and `gradient_summary.csv`.
- NaN/Inf: `{'yes' if 'FAIL_NUMERICAL_INSTABILITY' in numerical_flags else 'no'}`.
- Flags: `{', '.join(sorted(numerical_flags)) if numerical_flags else 'none'}`.

## Q4 — risky intervention relative scale

- Risk-gradient / total-gradient, beta=0.5: `{np.mean(risk_total_ratios['beta05']):.8f}`.
- Risk-gradient / total-gradient, beta=1.0: `{np.mean(risk_total_ratios['beta10']):.8f}`.

## Q5 — formal-training gate

`{diagnostic}`: {'两个公式实现正确，q 正常，beta=1 无明显数值异常，可以进入正式训练比较。' if diagnostic == 'PASS' else '发现实现/数值问题，需要先修复 diagnostic 中指出的问题。'}

This diagnostic does **not** determine which beta gives better segmentation. A full training comparison is required for final performance judgment.
"""
    (out / "final_report.md").write_text(report, encoding="utf-8")

    print("=== Anti-Confirmation Budget Gradient Diagnostic ===")
    print("\nK = 50")
    print("Training performed: false")
    print("Optimizer step performed: false")
    print("Validation used for q calibration: false")
    print("Test used: false")
    print("\nRisk statistics:")
    print(f"  risk fraction: {summary['risk_fraction']:.8f}")
    print(f"  q mean: {q_distribution['mean']:.8f}")
    print(f"  q p10: {q_distribution['p10']:.8f}")
    print(f"  q p50: {q_distribution['p50']:.8f}")
    print(f"  q p90: {q_distribution['p90']:.8f}")
    print(f"  q<0.10: {low_010:.8f}")
    print(f"  q>0.90: {high_090:.8f}")
    print(f"  q<0.05: {extreme_low:.8f}")
    print(f"  q>0.95: {extreme_high:.8f}")
    print("\nVariant A:")
    print("  beta = 0.5")
    print(f"  mean CPS weight: {np.mean(all_weight_a):.8f}")
    print(f"  mean reverse weight: {np.mean([0.5 * q for q in all_q]):.8f}")
    print("\nVariant B:")
    print("  beta = 1.0")
    print(f"  mean CPS weight: {np.mean(all_weight_b):.8f}")
    print(f"  mean reverse weight: {np.mean(all_q):.8f}")
    print("\nIsolated risky loss:")
    print(f"  B/A ratio: {np.mean(isolated_loss_ratios):.8f}")
    print("\nIsolated risky gradient:")
    print(f"  Student A B/A norm ratio: {np.mean(isolated_grad_ratios['A']):.8f}")
    print(f"  Student B B/A norm ratio: {np.mean(isolated_grad_ratios['B']):.8f}")
    print(f"  Gradient cosine A: {np.nanmean(cosines['A']):.8f}")
    print(f"  Gradient cosine B: {np.nanmean(cosines['B']):.8f}")
    print("\nFull training gradient:")
    print(f"  Student A B/A norm ratio: {np.mean(total_grad_ratios['A']):.8f}")
    print(f"  Student B B/A norm ratio: {np.mean(total_grad_ratios['B']):.8f}")
    print(f"  total B/A norm ratio: {np.mean(total_grad_ratios['total']):.8f}")
    print("\nRisk / total gradient:")
    print(f"  beta=0.5: {np.mean(risk_total_ratios['beta05']):.8f}")
    print(f"  beta=1.0: {np.mean(risk_total_ratios['beta10']):.8f}")
    print("\nNumerical instability:")
    print(f"  NaN: {'yes' if 'FAIL_NUMERICAL_INSTABILITY' in numerical_flags else 'no'}")
    print(f"  Inf: {'yes' if 'FAIL_NUMERICAL_INSTABILITY' in numerical_flags else 'no'}")
    print(f"  abnormal gradient: {', '.join(sorted(numerical_flags)) if numerical_flags else 'no'}")
    print(f"\nDiagnostic: {diagnostic}")
    print("\nNOTE:")
    print("This diagnostic does NOT determine which beta gives better segmentation.")
    print("A full training comparison is required for final performance judgment.")
    print(f"\nOutputs: {out}")


if __name__ == "__main__":
    main()
