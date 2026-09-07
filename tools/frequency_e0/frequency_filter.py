"""Radial Fourier-band attenuation used by the closed E0 diagnostic."""

from __future__ import annotations

from typing import Iterable, Tuple

import torch


BANDS: Tuple[Tuple[float, float], ...] = (
    (0.00, 0.05),
    (0.05, 0.10),
    (0.10, 0.20),
    (0.20, 0.30),
    (0.30, 0.45),
    (0.45, 0.60),
    (0.60, 0.80),
    (0.80, 1.00),
)


def radial_frequency_map(height: int, width: int, device=None) -> torch.Tensor:
    """Return rho with 0 at the FFT center and 1 at the image corners."""
    y = torch.arange(height, device=device, dtype=torch.float32) - (height // 2)
    x = torch.arange(width, device=device, dtype=torch.float32) - (width // 2)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    rho = torch.sqrt(xx.square() + yy.square())
    return rho / rho.max().clamp_min(torch.finfo(rho.dtype).eps)


def attenuate_radial_band(
    image: torch.Tensor,
    band: Tuple[float, float],
    attenuation: float,
    rho: torch.Tensor | None = None,
) -> torch.Tensor:
    """Attenuate one radial band in each RGB channel, preserving phase.

    ``image`` is expected to be ``[C,H,W]`` or ``[N,C,H,W]`` in the range used
    before model normalization (normally [0, 1]). No spatial clipping is done
    here: clipping would be an additional, non-Fourier perturbation.
    """
    if image.ndim not in (3, 4):
        raise ValueError("image must have shape [C,H,W] or [N,C,H,W]")
    if not 0.0 <= attenuation <= 1.0:
        raise ValueError("attenuation must be in [0, 1]")
    low, high = band
    if not (0.0 <= low < high <= 1.0):
        raise ValueError(f"invalid radial band: {band}")

    squeeze = image.ndim == 3
    batch = image.unsqueeze(0) if squeeze else image
    _, _, height, width = batch.shape
    if rho is None:
        rho = radial_frequency_map(height, width, batch.device)
    mask = ((rho >= low) & (rho < high)).to(batch.dtype)
    # Include the corner at rho == 1.0 in the last requested band.
    if high == 1.0:
        mask = ((rho >= low) & (rho <= high)).to(batch.dtype)

    spectrum = torch.fft.fftshift(torch.fft.fft2(batch, dim=(-2, -1)), dim=(-2, -1))
    amplitude = torch.abs(spectrum)
    phase = torch.angle(spectrum)
    amplitude_new = amplitude * (1.0 - attenuation * mask)
    spectrum_new = torch.polar(amplitude_new, phase)
    reconstructed = torch.fft.ifft2(
        torch.fft.ifftshift(spectrum_new, dim=(-2, -1)), dim=(-2, -1)
    ).real
    if not torch.isfinite(reconstructed).all():
        raise FloatingPointError("non-finite values after Fourier reconstruction")
    return reconstructed.squeeze(0) if squeeze else reconstructed


def check_zero_attenuation(image: torch.Tensor) -> float:
    """Run the exact reconstruction path with zero attenuation."""
    reconstructed = attenuate_radial_band(image, (0.0, 1.0), 0.0)
    return float(torch.max(torch.abs(reconstructed - image)).item())

