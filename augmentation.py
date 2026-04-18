"""
Data augmentation for tear microscopy images (torchvision.transforms.v2).

Design rules for this domain:
  - Microscopy has NO preferred orientation -> full rotation + flips are free wins.
  - `scale` (50 / 92 µm) is a feature column, so DO NOT resize, zoom, or
    random-resized-crop: that would silently change the effective micrometer
    scale and contradict the feature.
  - Color can carry diagnostic info; keep color jitter mild, avoid hue shifts.
"""

import torch
from torchvision.transforms import v2


# ---------- Core augmentation pipeline ----------
def build_train_transform(img_size: int = 224):
    return v2.Compose([
        v2.ToImage(),                                   # HWC uint8/ndarray -> Image tensor
        v2.ToDtype(torch.float32, scale=True),          # [0,1]

        # --- required: full rotation + translation invariance ---
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomAffine(
            degrees=180,                                # ±180°, any direction
            translate=(0.5, 0.5),                       # up to ±50% in x and y
            scale=None,                                 # keep native scale (scale is a feature)
            shear=0,
            fill=0,                                     # black fill; use 'reflect' via padding if preferred
            interpolation=v2.InterpolationMode.BILINEAR,
        ),

        # --- sensor / focus realism ---
        v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.3),
        v2.RandomApply([v2.GaussianNoise(mean=0.0, sigma=0.02)], p=0.3),

        # --- occlusion regularizer ---
        v2.RandomErasing(p=0.25, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),

        # --- ImageNet normalization (matches pretrained ResNet50 feature extractor) ---
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_eval_transform():
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ---------- Quick usage example ----------
if __name__ == "__main__":
    import numpy as np
    tfm = build_train_transform()
    fake_img = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    out = tfm(fake_img)
    print("Output:", out.shape, out.dtype, out.min().item(), out.max().item())