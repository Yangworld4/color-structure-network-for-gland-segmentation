"""Default configuration for aligned appearance/structure dataset views."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


_DEFAULT_DUAL_VIEW: Dict[str, Any] = {
    "enabled": True,
    "structure_channels": ["hematoxylin", "gray", "sobel"],
    "color_jitter": {
        "enabled": True,
        "brightness": 0.10,
        "contrast": 0.10,
        "saturation": 0.08,
        "hue": 0.02,
    },
    "stain_augmentation": {
        "enabled": False,
        "probability": 0.5,
    },
    # Produce the strong view on labeled validation data for diagnostics only.
    # Standard validation/test inference still uses image_appearance_weak.
    "generate_strong_eval": True,
    "structure_normalization": "minmax",
    "debug_visualization": False,
}


def dual_view_config() -> Dict[str, Any]:
    """Return an independent copy so callers may safely override values."""
    return deepcopy(_DEFAULT_DUAL_VIEW)
