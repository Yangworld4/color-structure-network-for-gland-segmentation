"""Settings for the frozen UNI reference branch.

Keeping these values outside ``main.py`` makes the training entry point easier
to scan while leaving the reference protocol explicit and centralized.
"""

from pathlib import Path


UNI_WEIGHTS_DEFAULT = (
    '/home/yangorange/.cache/huggingface/hub/models--MahmoodLab--UNI/'
    'snapshots/b55a5ec6cade1a39edfe6534189a9b8ca7a022f0/pytorch_model.bin'
)
UNI_CONFIG_DEFAULT = str(Path(UNI_WEIGHTS_DEFAULT).with_name('config.json'))
UNI_PATCH_SIZE = 16
DEFAULT_REFERENCE_RESOLUTION = 224

# UNI's original ViT-L/16 input contract is fixed at 224x224 (14x14 patches).
SUPPORTED_REFERENCE_RESOLUTIONS = (DEFAULT_REFERENCE_RESOLUTION,)
UNI_MEAN = (0.485, 0.456, 0.406)
UNI_STD = (0.229, 0.224, 0.225)

REFERENCE_MEMORY_ANCHOR_FG = 0.90
REFERENCE_MEMORY_ANCHOR_BG = 0.10
REFERENCE_MEMORY_PER_IMAGE_PER_CLASS = 64
# Cell nuclei are much smaller than the gland regions in GLaS/CRAG. If no
# UNI patch reaches an anchor threshold, retain the most foreground/background
# dominated patches instead of producing an empty class memory.
REFERENCE_MEMORY_FALLBACK_FRACTION = 0.10
REFERENCE_MEMORY_SEED = 2026

EXPECTED_UNI_SHA256 = (
    '56ef09b44a25dc5c7eedc55551b3d47bcd17659a7a33837cf9abc9ec4e2ffb40'
)

