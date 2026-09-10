import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# =====================================================================
# Five-level degradation operator suite (see paper Table 2)
# =====================================================================
def get_degradation_transform(level=0, for_saving=False):
    """Return the degradation pipeline for a severity level.

    :param level: 0 (clean) to 4 (extreme)
    :param for_saving: if True, skip Normalize/ToTensorV2 so the output can
        be written to disk with cv2.imwrite (used by generate_benchmarks.py)
    """
    transforms = []

    if level == 1:
        # L1 (mild): slight blur, moderate JPEG compression
        transforms.extend([
            A.GaussianBlur(blur_limit=(3, 5), p=0.8),
            A.ImageCompression(quality_lower=50, quality_upper=80, p=0.5),
        ])
    elif level == 2:
        # L2 (moderate): blur + light morphological change + JPEG
        transforms.extend([
            A.GaussianBlur(blur_limit=(3, 7), p=0.8),
            A.OneOf([
                A.Morphological(scale=(1, 2), operation='dilation', p=0.5),
                A.Morphological(scale=(1, 2), operation='erosion', p=0.5),
            ], p=0.7),
            A.ImageCompression(quality_lower=30, quality_upper=60, p=0.7),
        ])
    elif level == 3:
        # L3 (severe): downscale + small occlusions + elastic warp + blur
        transforms.extend([
            A.Downscale(scale_min=0.4, scale_max=0.6, p=0.8),
            A.CoarseDropout(max_holes=3, max_height=20, max_width=20,
                            fill_value=255, p=0.7),
            A.ElasticTransform(alpha=1, sigma=10, alpha_affine=10, p=0.5),
            A.GaussianBlur(blur_limit=(5, 9), p=0.8),
        ])
    elif level == 4:
        # L4 (extreme): strong downscale, large/slit occlusions, morphology,
        # elastic warp, aggressive JPEG
        transforms.extend([
            A.Downscale(scale_min=0.3, scale_max=0.5, p=0.9),
            A.OneOf([
                A.CoarseDropout(max_holes=5, max_height=35, max_width=35,
                                fill_value=255, p=0.8),
                A.CoarseDropout(max_holes=2, max_height=120, max_width=15,
                                fill_value=255, p=0.8),
            ], p=0.85),
            A.OneOf([
                A.Morphological(scale=(2, 3), operation='dilation', p=0.5),
                A.Morphological(scale=(1, 2), operation='erosion', p=0.5),
            ], p=0.7),
            A.ElasticTransform(alpha=1, sigma=15, alpha_affine=15, p=0.6),
            A.ImageCompression(quality_lower=10, quality_upper=30, p=0.9),
        ])

    transforms.append(A.Resize(300, 300))

    if not for_saving:
        transforms.extend([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    return A.Compose(transforms)


def get_random_erasing_transform(p=0.7, max_holes=3, max_height=40, max_width=40):
    """Random-erasing baseline: clean images with random white occlusions.

    This is the training-time transform of the "random-erasing baseline" row
    in the paper: occlusion exposure without degradation-level anchoring.
    """
    return A.Compose([
        A.CoarseDropout(max_holes=max_holes, max_height=max_height,
                        max_width=max_width, fill_value=255, p=p),
        A.Resize(300, 300),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# =====================================================================
# Core dataset
# =====================================================================
class MoleculeRetrievalDataset(Dataset):
    """Molecule--image pairs with precomputed graph features.

    :param img_dir: a single image root or a list of equivalent roots
        (e.g. default renders + style-diverse renders); a random root is
        picked per access so every epoch mixes depiction styles.
    :param transform: a single Albumentations pipeline, or a list of
        pipelines together with `level_probs` for per-sample severity
        sampling (the MixTrain exposure-matched schedule).
    """

    def __init__(self, df, img_dir, fp_path,
                 graph_path="data/processed/fastrp_embeddings.npz",
                 transform=None, level_probs=None):
        self.df = df
        self.img_dirs = list(img_dir) if isinstance(img_dir, (list, tuple)) else [img_dir]
        self.level_probs = level_probs
        if isinstance(transform, (list, tuple)):
            assert level_probs is not None and len(transform) == len(level_probs)
            self.transforms = list(transform)
            self.transform = None
        else:
            self.transform = transform
            self.transforms = None

        print("Loading feature stores...")
        loaded_fps = np.load(fp_path)
        self.fp_dict = {key: loaded_fps[key] for key in loaded_fps.files}

        loaded_fastrp = np.load(graph_path)
        self.graph_dict = {}
        for k in loaded_fastrp.files:
            clean_key = k.replace("MOL_", "").replace("mol_", "").replace(".npy", "")
            self.graph_dict[clean_key] = loaded_fastrp[k]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        chembl_id = str(row['chembl_id'])

        img_path = os.path.join(random.choice(self.img_dirs), f"{chembl_id}.png")
        image = cv2.imread(img_path)
        if image is None:
            image = np.full((300, 300, 3), 255, dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            tf = random.choices(self.transforms, weights=self.level_probs, k=1)[0]
            image_tensor = tf(image=image)['image']
        elif self.transform is not None:
            image_tensor = self.transform(image=image)['image']
        else:
            image_tensor = image  # raw numpy image (no tensor conversion)

        fp_array = self.fp_dict.get(chembl_id, np.zeros(2048, dtype=np.float32))
        fp_tensor = torch.tensor(fp_array, dtype=torch.float32)

        smiles_key = str(row['canonical_smiles'])
        graph_array = self.graph_dict.get(smiles_key, np.zeros(128, dtype=np.float32))
        graph_tensor = torch.tensor(graph_array, dtype=torch.float32)

        return image_tensor, graph_tensor, fp_tensor, chembl_id
