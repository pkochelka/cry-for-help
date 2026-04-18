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

FINAL_SIZE = 256

def build_train_transform():
    post_crop = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomAffine(degrees=180, translate=(0.5, 0.5), scale=None, shear=0,
                        fill=0, interpolation=v2.InterpolationMode.BILINEAR),
        v2.RandomApply([v2.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.3),
        v2.RandomApply([v2.GaussianNoise(mean=0.0, sigma=0.02)], p=0.3),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0),

        v2.RandomErasing(p=0.25, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def transform(img):
        img = v2.ToImage()(img)
        img = v2.ToDtype(torch.float32, scale=True)(img)
        img = v2.Resize(FINAL_SIZE)(img)
        return post_crop(img)

    return transform


def build_eval_transform():
    def transform(img):
        img = v2.ToImage()(img)
        img = v2.ToDtype(torch.float32, scale=True)(img)
        img = v2.Resize(FINAL_SIZE)(img)
        return v2.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])(img)
    return transform