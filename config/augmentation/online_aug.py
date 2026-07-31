import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchio import transforms as T
import torchio as tio
import torch
import torch.nn.functional as F

def data_transform_2d():
    data_transforms = {
        'train': A.Compose([
            # 论文复现修正：论文明确使用 256×256 输入。
            A.Resize(256, 256, p=1),
            A.Flip(p=0.75),
            A.Transpose(p=0.5),
            A.RandomRotate90(p=1),
        ],
            additional_targets={'image2': 'image', 'mask2': 'mask'}
        ),
        'val': A.Compose([
            A.Resize(256, 256, p=1),
        ],
            additional_targets={'image2': 'image', 'mask2': 'mask'}
        ),
        'test': A.Compose([
            A.Resize(256, 256, p=1),
        ],
            additional_targets={'image2': 'image', 'mask2': 'mask'}
        )
    }
    return data_transforms

def data_normalize_2d(mean, std):
    data_normalize = A.Compose([
            A.Normalize(mean, std),
            ToTensorV2()
        ],
            additional_targets={'image2': 'image', 'mask2': 'mask'}
    )
    return data_normalize


def weak_augmentation(x):
    """弱光度增强，不改变空间位置。"""
    x = x.clone()

    # 每张图像独立亮度变化
    brightness = torch.empty(
        x.size(0), 1, 1, 1, device=x.device
    ).uniform_(0.95, 1.05)
    x = x * brightness

    # 每张图像独立对比度变化
    contrast = torch.empty(
        x.size(0), 1, 1, 1, device=x.device
    ).uniform_(0.95, 1.05)
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * contrast + mean

    return x.clamp(0.0, 1.0)


def strong_augmentation(x):
    """强光度增强，不改变空间位置。"""
    x = x.clone()

    # 更强的亮度变化
    brightness = torch.empty(
        x.size(0), 1, 1, 1, device=x.device
    ).uniform_(0.75, 1.25)
    x = x * brightness

    # 更强的对比度变化
    contrast = torch.empty(
        x.size(0), 1, 1, 1, device=x.device
    ).uniform_(0.70, 1.30)
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * contrast + mean

    # 饱和度变化
    saturation = torch.empty(
        x.size(0), 1, 1, 1, device=x.device
    ).uniform_(0.65, 1.35)

    gray = (
        0.299 * x[:, 0:1]
        + 0.587 * x[:, 1:2]
        + 0.114 * x[:, 2:3]
    )
    x = gray + saturation * (x - gray)

    # Gamma变化
    gamma = torch.empty(
        x.size(0), 1, 1, 1, device=x.device
    ).uniform_(0.7, 1.4)
    x = x.clamp(1e-6, 1.0).pow(gamma)

    # 高斯噪声
    x = x + torch.randn_like(x) * 0.03

    # 轻度模糊
    if torch.rand(1, device=x.device).item() < 0.3:
        x = F.avg_pool2d(
            x,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    return x.clamp(0.0, 1.0)