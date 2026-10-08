"""Local UNI backbone construction used by DUAL reference memory.

The pretrained parameters remain an explicit local file supplied by the user;
the model construction and validation live in this project.
"""

import hashlib
import json
import os
from functools import partial
from pathlib import Path
from urllib.parse import urlparse

import timm
import torch
import torch.nn as nn


UNI_ARCHITECTURE = 'vit_large_patch16_224'
UNI_EMBED_DIM = 1024
UNI_PATCH_SIZE = (16, 16)
UNI2_H_MODEL_ID = 'MahmoodLab/UNI2-h'
KAIKO_VITB8_ID = 'kaiko-ai/towards_large_pathology_fms:vitb8'


def _hub_model_kwargs(model_id):
    if model_id.lower() != UNI2_H_MODEL_ID.lower():
        return {
            'init_values': 1e-5,
            'dynamic_img_size': True,
            'num_classes': 0,
        }
    # UNI2-h's official feature-extraction setup uses this ViT-H/14 shape.
    # Keep the architecture details inside this adapter so the training and
    # memory-retrieval code only sees dense, normalized patch tokens.
    return {
        'img_size': 224,
        'patch_size': 14,
        'depth': 24,
        'num_heads': 24,
        'init_values': 1e-5,
        'embed_dim': 1536,
        'mlp_ratio': 2.66667 * 2,
        'num_classes': 0,
        'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked,
        'act_layer': torch.nn.SiLU,
        'reg_tokens': 8,
        'dynamic_img_size': True,
    }


def _torchhub_checkpoint_path(model):
    pretrained_cfg = getattr(model, 'pretrained_cfg', None)
    if isinstance(pretrained_cfg, dict):
        url = pretrained_cfg.get('url')
    else:
        url = getattr(pretrained_cfg, 'url', None)
    if not url:
        return 'torch.hub:{}'.format(KAIKO_VITB8_ID)
    filename = os.path.basename(urlparse(url).path)
    return str(Path(torch.hub.get_dir()) / 'checkpoints' / filename)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _create_hf_model_local_first(model_id, model_kwargs):
    """Use a complete local HF cache without creating an HTTP client."""
    from huggingface_hub import hf_hub_download

    config_path = None
    weights_path = None
    safe_weights_path = None
    try:
        config_path = hf_hub_download(
            repo_id=model_id, filename='config.json', local_files_only=True
        )
        try:
            safe_weights_path = hf_hub_download(
                repo_id=model_id, filename='model.safetensors',
                local_files_only=True,
            )
        except Exception:
            safe_weights_path = None
        if safe_weights_path is not None:
            weights_path = safe_weights_path
        else:
            weights_path = hf_hub_download(
                repo_id=model_id, filename='pytorch_model.bin',
                local_files_only=True,
            )
    except Exception:
        # Preserve timm's standard Hub download path when this machine does
        # not yet have a complete local snapshot.
        return None

    import timm.models._hub as timm_hub

    original_download = timm_hub.hf_hub_download
    original_safetensors = timm_hub._has_safetensors
    timm_hub.hf_hub_download = partial(
        original_download, local_files_only=True
    )
    # timm probes for safetensors before its .bin fallback. If only the .bin
    # file is cached, skip that remote existence probe altogether.
    timm_hub._has_safetensors = safe_weights_path is not None
    try:
        encoder = timm.create_model(
            'hf-hub:{}'.format(model_id),
            pretrained=True,
            **model_kwargs
        )
    finally:
        timm_hub.hf_hub_download = original_download
        timm_hub._has_safetensors = original_safetensors
    return encoder, weights_path, config_path


class UNIBackbone(nn.Module):
    """Frozen UNI encoder with a dense patch-token interface."""

    def __init__(
        self, weights_path=None, config_path=None, expected_sha256=None,
        hub_model_id=None, backbone='uni',
    ):
        super().__init__()
        if backbone == 'kaiko-vitb8':
            encoder = torch.hub.load(
                'kaiko-ai/towards_large_pathology_fms',
                'vitb8',
                trust_repo=True,
            )
            weights_path = _torchhub_checkpoint_path(encoder)
            if not os.path.isfile(weights_path):
                raise FileNotFoundError(
                    'Kaiko ViT-B/8 checkpoint was not found in the torch hub cache: {}'.format(
                        weights_path
                    )
                )
            weights_sha256 = _sha256_file(weights_path)
            config_path = 'torch.hub:{}'.format(KAIKO_VITB8_ID)
            source = config_path
        elif hub_model_id:
            # Use timm's official HF Hub integration and default revision.
            from huggingface_hub import hf_hub_download

            cached = _create_hf_model_local_first(
                hub_model_id, _hub_model_kwargs(hub_model_id)
            )
            if cached is None:
                encoder = timm.create_model(
                    'hf-hub:{}'.format(hub_model_id),
                    pretrained=True,
                    **_hub_model_kwargs(hub_model_id)
                )
                # Keep an auditable local checkpoint path and digest in run
                # metadata, while timm remains responsible for model loading.
                weights_path = hf_hub_download(
                    repo_id=hub_model_id,
                    filename='pytorch_model.bin',
                )
                config_path = 'hf-hub:{}'.format(hub_model_id)
            else:
                encoder, weights_path, config_path = cached
            # Keep an auditable local checkpoint path and digest in run
            # metadata, while timm remains responsible for model loading.
            weights_sha256 = _sha256_file(weights_path)
            if expected_sha256 is not None and weights_sha256 != expected_sha256:
                raise RuntimeError(
                    'UNI SHA256 mismatch: expected={} actual={}'.format(
                        expected_sha256, weights_sha256
                    )
                )
            config_path = 'hf-hub:{}'.format(hub_model_id)
            source = config_path
        else:
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
            source = str(weights_path)

        self.encoder = encoder.cuda().eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.eval()

        expected_dims = {
            'uni': UNI_EMBED_DIM,
            'uni2-h': 1536,
            'kaiko-vitb8': 768,
        }
        if int(getattr(self.encoder, 'embed_dim', -1)) != expected_dims[backbone]:
            raise RuntimeError(
                'Unexpected {} embedding dimension: {}'.format(
                    backbone, getattr(self.encoder, 'embed_dim', None)
                )
            )
        patch_size = tuple(int(value) for value in self.encoder.patch_embed.patch_size)
        expected_patch = {
            'uni': UNI_PATCH_SIZE,
            'uni2-h': (14, 14),
            'kaiko-vitb8': (8, 8),
        }[backbone]
        if patch_size != expected_patch:
            raise RuntimeError('Unsupported pathology encoder patch size: {}'.format(patch_size))
        if int(getattr(self.encoder, 'num_prefix_tokens', -1)) < 1:
            raise RuntimeError('Encoder must expose its prefix-token count')
        self.feature_dim = int(self.encoder.embed_dim)
        self.patch_size = int(patch_size[0])
        self.token_grid = None
        self.model_id = {
            'uni': hub_model_id or 'MahmoodLab/UNI',
            'uni2-h': UNI2_H_MODEL_ID,
            'kaiko-vitb8': KAIKO_VITB8_ID,
        }[backbone]
        self.backbone = backbone
        if backbone == 'kaiko-vitb8':
            self.input_mean = (0.5, 0.5, 0.5)
            self.input_std = (0.5, 0.5, 0.5)
        else:
            self.input_mean = (0.485, 0.456, 0.406)
            self.input_std = (0.229, 0.224, 0.225)

        self.weights_path = str(weights_path)
        self.config_path = str(config_path)
        self.weights_sha256 = weights_sha256
        self.source = source

    def dense_patch_tokens(self, image):
        """Return normalized patch tokens with shape ``[B, grid**2, dim]``."""
        x = self.encoder.forward_features(image)
        if x.ndim != 3:
            raise RuntimeError(
                'UNI forward_features must return patch tokens [B,N,C], got {}'.format(
                    tuple(x.shape)
                )
            )
        prefix = int(self.encoder.num_prefix_tokens)
        tokens = x[:, prefix:, :]
        grid = int(round(tokens.shape[1] ** 0.5))
        if grid * grid != tokens.shape[1]:
            raise RuntimeError('Encoder patch-token count is not square: {}'.format(tokens.shape[1]))
        self.token_grid = grid
        return torch.nn.functional.normalize(tokens, p=2, dim=-1)

    def forward(self, image):
        return self.dense_patch_tokens(image)


def load_uni(
    weights_path=None, config_path=None, expected_sha256=None,
    loader='local', hub_model_id='MahmoodLab/UNI', backbone='uni',
):
    """Build frozen UNI from local files or the official timm HF Hub source."""
    if loader not in ('local', 'hf-hub'):
        raise ValueError('UNI loader must be local or hf-hub, got {}'.format(loader))
    if backbone not in ('uni', 'uni2-h', 'kaiko-vitb8'):
        raise ValueError('Unknown pathology backbone: {}'.format(backbone))
    if backbone == 'uni2-h':
        loader = 'hf-hub'
        hub_model_id = UNI2_H_MODEL_ID
    elif backbone == 'kaiko-vitb8':
        loader = 'torch-hub'
    if loader not in ('local', 'hf-hub', 'torch-hub'):
        raise ValueError('Unsupported pathology model loader: {}'.format(loader))
    model = UNIBackbone(
        weights_path=weights_path,
        config_path=config_path,
        expected_sha256=(expected_sha256 if backbone == 'uni' else None),
        hub_model_id=hub_model_id if loader == 'hf-hub' else None,
        backbone=backbone,
    )
    return model, model.weights_path, model.config_path
