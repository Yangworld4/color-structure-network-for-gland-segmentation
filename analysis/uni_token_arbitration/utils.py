"""I/O and project-faithful UNI/CPS helpers for the closed diagnostic."""

from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_ids(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {Path(row["image"]).stem for row in csv.DictReader(handle)}


def load_memory(path: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, dict]:
    with np.load(path, allow_pickle=False) as data:
        required = {"memory_fg", "memory_bg", "feature_dim", "token_grid", "top_k"}
        absent = required.difference(data.files)
        if absent:
            raise RuntimeError(f"Memory missing required keys: {sorted(absent)}")
        meta = {key: (data[key].item() if data[key].ndim == 0 else data[key].tolist()) for key in data.files if key != "memory_fg" and key != "memory_bg"}
        fg, bg = np.asarray(data["memory_fg"], np.float32), np.asarray(data["memory_bg"], np.float32)
    if fg.ndim != 2 or bg.ndim != 2 or fg.shape[1] != bg.shape[1]:
        raise RuntimeError(f"Invalid memory shapes: FG={fg.shape}, BG={bg.shape}")
    if not np.allclose(np.linalg.norm(fg, axis=1), 1.0, atol=1e-4) or not np.allclose(np.linalg.norm(bg, axis=1), 1.0, atol=1e-4):
        raise RuntimeError("Reference memory is not L2 normalized")
    return torch.from_numpy(fg).to(device), torch.from_numpy(bg).to(device), meta


def reconstruct_rgb(image: torch.Tensor, mean: tuple[float, ...], std: tuple[float, ...]) -> torch.Tensor:
    mean_t = torch.tensor(mean, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    return (image * std_t + mean_t).clamp(0, 1)


def uni_tokens_native(uni, rgb_01: torch.Tensor, resolution: int) -> tuple[torch.Tensor, int, int]:
    # Exact preprocessing in main._uni_dense_features, deliberately without any
    # later interpolation of scores or margins.
    mean = torch.tensor((0.485, 0.456, 0.406), device=rgb_01.device, dtype=rgb_01.dtype).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=rgb_01.device, dtype=rgb_01.dtype).view(1, 3, 1, 1)
    x = F.interpolate(rgb_01, size=(resolution, resolution), mode="bilinear", align_corners=False)
    tokens = uni.dense_patch_tokens((x - mean) / std)
    count, dim = int(tokens.shape[1]), int(tokens.shape[2])
    grid = int(math.sqrt(count))
    if grid * grid != count:
        raise RuntimeError(f"UNI returned non-square token count {count}")
    return tokens, grid, dim


def topk_scores(query: torch.Tensor, memory: torch.Tensor, k: int) -> torch.Tensor:
    # Exact main._topk_scores implementation.
    return (query @ memory.t()).topk(min(int(k), int(memory.shape[0])), dim=1, largest=True).values.mean(dim=1)


def score_native(tokens: torch.Tensor, memory_fg: torch.Tensor, memory_bg: torch.Tensor, k: int, grid: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = tokens.reshape(-1, tokens.shape[-1])
    s_fg = topk_scores(flat, memory_fg, k).reshape(tokens.shape[0], grid, grid)
    s_bg = topk_scores(flat, memory_bg, k).reshape(tokens.shape[0], grid, grid)
    margin = s_fg - s_bg
    assert s_fg.shape[-2:] == (grid, grid)
    assert s_bg.shape[-2:] == (grid, grid)
    assert margin.shape[-2:] == (grid, grid)
    return s_fg, s_bg, margin


def load_students(path: Path, device: torch.device):
    from models.getnetwork import get_network
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("appearance_state_dict", "structure_state_dict"):
        if key not in payload:
            raise RuntimeError(f"Milestone lacks {key}: {path}")
    a, b = get_network("unet", 3, 2), get_network("unet", 3, 2)
    a.load_state_dict(payload["appearance_state_dict"], strict=True)
    b.load_state_dict(payload["structure_state_dict"], strict=True)
    return a.to(device).eval(), b.to(device).eval(), payload
