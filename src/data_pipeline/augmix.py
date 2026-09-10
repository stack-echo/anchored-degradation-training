"""Faithful AugMix image transform (Hendrycks et al., ICLR 2020).

Structure follows the reference implementation:
  - mixture of `width` augmentation chains, each of random depth in [1, 3]
  - per-chain ops drawn uniformly from the 12-op augmentations_all set
  - chain weights ~ Dirichlet(alpha, ..., alpha)
  - clean-image mixing weight ~ Beta(alpha, alpha)
Canonical configuration: severity fixed at 3, width 3, depth -1 (random 1-3),
alpha 1.0.

Domain adaptation (documented): geometric ops (rotate / shear / translate)
use white fill (255) instead of black, matching the white background of
molecular structure depictions.
"""
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import albumentations as A
from albumentations.pytorch import ToTensorV2


def _int_parameter(level, maxval):
    return int(level * maxval / 10)


def _float_parameter(level, maxval):
    return float(level) * maxval / 10.0


def _autocontrast(pil_img, _):
    return ImageOps.autocontrast(pil_img)


def _equalize(pil_img, _):
    return ImageOps.equalize(pil_img)


def _posterize(pil_img, level):
    level = _int_parameter(level, 4)
    return ImageOps.posterize(pil_img, max(4 - level, 1))


def _rotate(pil_img, level):
    degrees = _int_parameter(level, 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR, fillcolor=(255, 255, 255))


def _solarize(pil_img, level):
    level = _int_parameter(level, 256)
    return ImageOps.solarize(pil_img, 256 - level)


def _shear_x(pil_img, level):
    level = _float_parameter(level, 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, level, 0, 0, 1, 0),
                             resample=Image.BILINEAR, fillcolor=(255, 255, 255))


def _shear_y(pil_img, level):
    level = _float_parameter(level, 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, 0, level, 1, 0),
                             resample=Image.BILINEAR, fillcolor=(255, 255, 255))


def _translate_x(pil_img, level):
    level = _int_parameter(level, 10)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, level, 0, 1, 0),
                             resample=Image.BILINEAR, fillcolor=(255, 255, 255))


def _translate_y(pil_img, level):
    level = _int_parameter(level, 10)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, 0, level, 1, 0),
                             resample=Image.BILINEAR, fillcolor=(255, 255, 255))


def _color(pil_img, level):
    level = _float_parameter(level, 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


def _brightness(pil_img, level):
    level = _float_parameter(level, 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


def _contrast(pil_img, level):
    level = _float_parameter(level, 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


def _sharpness(pil_img, level):
    level = _float_parameter(level, 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


AUGMENTATIONS_ALL = [
    _autocontrast, _equalize, _posterize, _rotate, _solarize,
    _shear_x, _shear_y, _translate_x, _translate_y,
    _color, _brightness, _contrast, _sharpness,
]


def augment_and_mix(image_np, severity=3, width=3, depth=-1, alpha=1.0):
    """Core AugMix mixing on a numpy RGB uint8 image. Returns numpy uint8."""
    ws = np.float32(np.random.dirichlet([alpha] * width))
    m = np.float32(np.random.beta(alpha, alpha))

    pil_img = Image.fromarray(image_np)
    mix = np.zeros((image_np.shape[0], image_np.shape[1], 3), dtype=np.float32)

    for i in range(width):
        image_aug = pil_img.copy()
        d = depth if depth > 0 else np.random.randint(1, 4)
        for _ in range(d):
            op = AUGMENTATIONS_ALL[np.random.randint(len(AUGMENTATIONS_ALL))]
            image_aug = op(image_aug, severity)
        mix += ws[i] * np.asarray(image_aug, dtype=np.float32)

    mixed = (1 - m) * np.asarray(pil_img, dtype=np.float32) + m * mix
    return np.clip(mixed, 0, 255).astype(np.uint8)


class AugMixOp(A.ImageOnlyTransform):
    """Albumentations wrapper so AugMix composes with the standard tail."""

    def __init__(self, severity=3, width=3, depth=-1, alpha=1.0, always_apply=True, p=1.0):
        super().__init__(always_apply=always_apply, p=p)
        self.severity = severity
        self.width = width
        self.depth = depth
        self.alpha = alpha

    def apply(self, img, **params):
        return augment_and_mix(img, self.severity, self.width, self.depth, self.alpha)


def get_augmix_transform(for_saving=False, severity=3, width=3, depth=-1, alpha=1.0):
    """Train-time pipeline: faithful AugMix followed by the standard resize/normalize tail."""
    transforms = [AugMixOp(severity=severity, width=width, depth=depth, alpha=alpha)]
    transforms.append(A.Resize(300, 300))
    if not for_saving:
        transforms.extend([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    return A.Compose(transforms)
