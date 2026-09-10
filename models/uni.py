"""Local UNI backbone construction used by DUAL reference memory.

The pretrained parameters remain an explicit local file supplied by the user;
the model construction and validation live in this project.
"""

import hashlib
import json
from pathlib import Path

import timm
import torch
import torch.nn as nn


UNI_ARCHITECTURE = 'vit_large_patch16_224'
UNI_EMBED_DIM = 1024
UNI_PATCH_SIZE = (16, 16)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class UNIBackbone(nn.Module):
    """Frozen ViT-L/16 UNI encoder with its dense patch-token interface."""

    def __init__(self, weights_path, config_path, expected_sha256=None):
        super().__init__()
        weights_path = Path(weights_path).expanduser().resolve()
        config_path = Path(config_path).expanduser().resolve()
        for path in (weights_path, config_path):
            if not path.is_file():
                raise FileNotFoundError('UNI file not found: {}'.format(path))

        weights_sha256 = _sha256_file(weights_path)
        if expected_sha256 is not None and weights_sha256 != expected_sha256:
            raise RuntimeError(
                'UNI SHA256 mismatch: expected={} actual={}'.format(
                    expected_sha256, weights_sha256
                )
            )
        with open(config_path, encoding='utf-8') as handle:
            config = json.load(handle)
        if config.get('architecture') != UNI_ARCHITECTURE:
            raise RuntimeError(
                'Unexpected UNI architecture: {}'.format(config.get('architecture'))
            )

        encoder = timm.create_model(
            UNI_ARCHITECTURE,
            pretrained=False,
            init_values=1e-5,
            dynamic_img_size=True,
            num_classes=0,
        )
        state_dict = torch.load(
            str(weights_path), map_location='cpu', weights_only=True
        )
        encoder.load_state_dict(state_dict, strict=True)
        self.encoder = encoder.cuda().eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.eval()

        if int(getattr(self.encoder, 'embed_dim', -1)) != UNI_EMBED_DIM:
            raise RuntimeError('Unexpected UNI embedding dimension')
        if tuple(self.encoder.patch_embed.patch_size) != UNI_PATCH_SIZE:
            raise RuntimeError('Unexpected UNI patch size')
        if int(getattr(self.encoder, 'num_prefix_tokens', -1)) != 1:
            raise RuntimeError('Unexpected UNI prefix-token count')
        if int(getattr(self.encoder, 'num_reg_tokens', 0)) != 0:
            raise RuntimeError('Unexpected UNI register-token count')

        self.weights_path = str(weights_path)
        self.config_path = str(config_path)
        self.weights_sha256 = weights_sha256

    def dense_patch_tokens(self, image):
        """Return normalized patch tokens with shape ``[B, 196, 1024]``."""
        x = self.encoder.patch_embed(image)
        x = self.encoder._pos_embed(x)
        for block in self.encoder.blocks:
            x = block(x)
        x = self.encoder.norm(x)
        prefix = int(self.encoder.num_prefix_tokens)
        register_tokens = int(getattr(self.encoder, 'num_reg_tokens', 0))
        end = x.shape[1] - register_tokens if register_tokens else x.shape[1]
        tokens = x[:, prefix:end, :]
        return torch.nn.functional.normalize(tokens, p=2, dim=-1)

    def forward(self, image):
        return self.dense_patch_tokens(image)


def load_uni(weights_path, config_path, expected_sha256=None):
    """Build the local frozen UNI model and return it with file metadata."""
    model = UNIBackbone(weights_path, config_path, expected_sha256)
    return model, Path(model.weights_path), Path(model.config_path)
