#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CPU latency benchmark for the ADT retrieval pipeline (deployment section).

Per-query wall time: PNG decode -> letterbox preprocess -> image-encoder
forward pass -> nearest-neighbour search over precomputed gallery embeddings.
Reports mean / p50 / p95 / min / max per stage.

Usage:
  python bench_latency.py --ckpt checkpoints/convnext_base_adt_l4_seed42_best.pth \
      --gallery_emb results_real/gallery_embed_..._morgan.npy \
      --gallery_keys data/processed/morgan_fingerprints.npz \
      --img_dir data/real_benchmarks/curated/Staker/images \
      --threads 4 --n 100 --warmup 15
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.models.adt_net import ADTNet


class PadToSquare(A.ImageOnlyTransform):
    def __init__(self, always_apply=True, p=1.0):
        super().__init__(p=p)

    def apply(self, img, **params):
        h, w = img.shape[:2]
        if h == w:
            return img
        s = max(h, w)
        top = (s - h) // 2
        left = (s - w) // 2
        return cv2.copyMakeBorder(img, top, s - h - top, left, s - w - left,
                                  cv2.BORDER_CONSTANT, value=(255, 255, 255))


def build_transform():
    return A.Compose([PadToSquare(), A.Resize(300, 300),
                      A.Normalize(mean=(0.485, 0.456, 0.406),
                                  std=(0.229, 0.224, 0.225)),
                      ToTensorV2()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--gallery_emb', required=True,
                    help='.npy of precomputed gallery embeddings (N, 512)')
    ap.add_argument('--gallery_keys', default='data/processed/morgan_fingerprints.npz')
    ap.add_argument('--img_dir', required=True, help='query images (PNG)')
    ap.add_argument('--graph_type', default='morgan', choices=['fastrp', 'morgan'])
    ap.add_argument('--vision_backbone', default='convnext_base')
    ap.add_argument('--embed_dim', type=int, default=512)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--warmup', type=int, default=15)
    ap.add_argument('--out', default='results_real/latency_cpu.json')
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    model = ADTNet(vision_backbone=a.vision_backbone, graph_type=a.graph_type,
                   embed_dim=a.embed_dim)
    model.load_state_dict(torch.load(a.ckpt, map_location='cpu'))
    model.eval()

    gallery = torch.from_numpy(np.load(a.gallery_emb)).float()  # (N, 512)
    gallery = F.normalize(gallery, dim=-1)

    tf = build_transform()
    imgs = sorted([f for f in os.listdir(a.img_dir) if f.endswith('.png')])[: a.n + a.warmup]
    assert len(imgs) >= a.n, f'not enough images: {len(imgs)}'

    rows = []
    with torch.no_grad():
        for i, fn in enumerate(imgs):
            t0 = time.perf_counter()
            img = np.array(Image.open(os.path.join(a.img_dir, fn)).convert('RGB'))
            t1 = time.perf_counter()
            x = tf(image=img)['image'].unsqueeze(0)
            t2 = time.perf_counter()
            emb = model.vision_encoder(x)
            if emb.dim() > 2:
                emb = emb.view(emb.size(0), -1)
            emb = F.normalize(emb.float(), dim=-1)
            t3 = time.perf_counter()
            sims = gallery @ emb[0]
            _ = torch.topk(sims, 5).indices.tolist()
            t4 = time.perf_counter()
            if i < a.warmup:
                continue
            rows.append({'decode_img_ms': (t1 - t0) * 1e3,
                         'preprocess_ms': (t2 - t1) * 1e3,
                         'encoder_ms': (t3 - t2) * 1e3,
                         'search_ms': (t4 - t3) * 1e3,
                         'total_ms': (t4 - t0) * 1e3})
    df = pd.DataFrame(rows)
    stats = {}
    for c in df.columns:
        s = df[c]
        stats[c] = {k: round(float(v), 2) for k, v in
                    [('mean', s.mean()), ('p50', s.quantile(.5)),
                     ('p95', s.quantile(.95)), ('min', s.min()), ('max', s.max())]}
    out = {'threads': a.threads, 'n_queries': len(df),
           'gallery_size': int(gallery.shape[0]),
           'cpu': os.popen('lscpu 2>/dev/null | grep "Model name" || sysctl -n machdep.cpu.brand_string').read().strip(),
           'stats': stats}
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
