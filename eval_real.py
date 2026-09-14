#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Retrieval evaluation on real-world OCSR benchmarks
(CLEF / UOB / USPTO / JPO / ACS / Staker / MolRecBenchWild).

Two gallery representations:
- fastrp: the 200k ChEMBL FastRP embedding space (in-distribution for the
  main models). Only ground-truth structures covered by that space can be
  scored (coverage is reported).
- morgan: deterministic Morgan fingerprints (r=2, 2048 bits); the gallery is
  the 200k ChEMBL fingerprints plus fingerprints of benchmark ground truths
  not already present. This mirrors deployment against a curated structure
  gallery and makes every valid ground truth evaluable (see paper
  Limitations).

Metrics: I2G R@1 / R@5 (stereo-stripped canonical SMILES match), ring / atom /
bond / aromatic-ring exact match of the top-1 retrieval, mean Tanimoto
(Morgan r=2, 2048 bits).

Usage:
  python eval_real.py --ckpt checkpoints/XXX.pth --graph_repr morgan \
      --vision_backbone convnext_base --benches CLEF --out_dir results_real
"""
import os
import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, DataStructs, rdFingerprintGenerator
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

from src.models.adt_net import ADTNet

_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--graph_repr', default='morgan', choices=['fastrp', 'morgan'])
    p.add_argument('--vision_backbone', default='convnext_base')
    p.add_argument('--graph_type', default=None,
                   help='default: same as graph_repr')
    p.add_argument('--embed_dim', type=int, default=512)
    p.add_argument('--real_root', default='data/real_benchmarks',
                   help='root with curated/<bench>/{records.csv,images/}')
    p.add_argument('--benches', default='CLEF,UOB,USPTO,JPO,ACS,Staker,MolRecBenchWild')
    p.add_argument('--csv_200k', default='data/my_200k_dataset.csv')
    p.add_argument('--fp_npz', default='data/processed/morgan_fingerprints.npz')
    p.add_argument('--graph_npz', default='data/processed/fastrp_embeddings.npz')
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out_dir', default='results_real')
    p.add_argument('--max_queries', type=int, default=0,
                   help='smoke-test cap per benchmark (0 = full)')
    return p.parse_args()


def fp_of(smiles):
    m = Chem.MolFromSmiles(str(smiles))
    if m is None:
        return None
    arr = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(_MORGAN.GetFingerprint(m), arr)
    return arr


def mol_stats(smiles):
    m = Chem.MolFromSmiles(str(smiles))
    if m is None:
        return None
    return {'mol': m,
            'rings': m.GetRingInfo().NumRings(),
            'atoms': m.GetNumAtoms(),
            'bonds': m.GetNumBonds(),
            'aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(m)}


def tanimoto(s1, s2):
    fp1 = rdMolDescriptors.GetMorganFingerprintAsBitVect(s1['mol'], 2, nBits=2048)
    fp2 = rdMolDescriptors.GetMorganFingerprintAsBitVect(s2['mol'], 2, nBits=2048)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


class _PadToSquare(A.ImageOnlyTransform):
    """Pad to square with white (letterbox), centered."""

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


class RealImageDataset(Dataset):
    """Real-world query images; letterbox padding preserves bond geometry."""

    def __init__(self, records, img_dir):
        self.records = records.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = A.Compose([
            _PadToSquare(),
            A.Resize(300, 300),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.records)

    def resolve(self, image_id):
        stem = os.path.splitext(str(image_id))[0]
        for ext in ('.png', '.jpg', '.jpeg'):
            path = os.path.join(self.img_dir, stem + ext)
            if os.path.exists(path):
                return path
        return None

    def __getitem__(self, idx):
        r = self.records.iloc[idx]
        path = self.resolve(r['image_id'])
        img = np.array(Image.open(path).convert('RGB'))
        return self.transform(image=img)['image']


def nostereo(smiles):
    m = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(m, isomericSmiles=False) if m is not None else None


def build_gallery(args, extra_smiles):
    """Return (keys, matrix): isomeric canonical SMILES and an (N, d) array."""
    if args.graph_repr == 'fastrp':
        z = np.load(args.graph_npz)
        keys = z.files
        mat = np.stack([z[k] for k in keys]).astype(np.float32)
        return keys, mat
    # Morgan gallery: 200k ChEMBL fingerprints + extra ground-truth fingerprints
    df = pd.read_csv(args.csv_200k)
    z = np.load(args.fp_npz)
    keys, rows = [], []
    for _, r in tqdm(df.iterrows(), total=len(df), desc='load ChEMBL fps'):
        cid = str(r['chembl_id'])
        if cid in z.files:
            keys.append(str(r['canonical_smiles']))
            rows.append(z[cid].astype(np.float32))
    existing = set(keys)
    for s in tqdm(extra_smiles, desc='extra GT fps'):
        if s not in existing:
            a = fp_of(s)
            if a is not None:
                keys.append(s)
                rows.append(a)
                existing.add(s)
    return keys, np.stack(rows)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tag = os.path.splitext(os.path.basename(args.ckpt))[0]
    graph_type = args.graph_type or ('morgan' if args.graph_repr == 'morgan' else 'fastrp')
    amp_on = str(args.device).startswith('cuda')

    # ---------- collect benchmark ground truths for gallery extension ----------
    bench_recs = {}
    for bench in args.benches.split(','):
        rec = pd.read_csv(f'{args.real_root}/curated/{bench}/records.csv')
        bench_recs[bench] = rec
    extra = sorted({s for rec in bench_recs.values()
                    for s in rec.loc[rec.valid, 'gt_smiles_iso'].dropna()})

    g_keys, g_mat = build_gallery(args, extra)
    g_ns = [nostereo(k) for k in tqdm(g_keys, desc='gallery nostereo')]
    print(f'gallery: {len(g_keys)} structures ({args.graph_repr})')

    # ---------- model ----------
    model = ADTNet(vision_backbone=args.vision_backbone, graph_type=graph_type,
                   embed_dim=args.embed_dim).to(args.device)
    model.load_state_dict(torch.load(args.ckpt, map_location=args.device))
    model.eval()
    print(f'checkpoint loaded: {args.ckpt} (graph tower: {graph_type})')

    cache = f'{args.out_dir}/gallery_embed_{tag}_{args.graph_repr}_{len(g_keys)}.npy'
    if os.path.exists(cache):
        gallery = torch.from_numpy(np.load(cache)).to(args.device)
    else:
        g_list = []
        with torch.no_grad():
            for i in tqdm(range(0, len(g_keys), 4096), desc='gallery embed'):
                gb = torch.from_numpy(g_mat[i:i + 4096]).to(args.device)
                with torch.amp.autocast(device_type='cuda' if amp_on else 'cpu',
                                        dtype=torch.bfloat16, enabled=amp_on):
                    ge = model.graph_encoder(gb)
                g_list.append(ge.float().cpu())
        gallery = F.normalize(torch.cat(g_list, 0), dim=-1)
        np.save(cache, gallery.numpy())
        gallery = gallery.to(args.device)

    # ---------- per-benchmark ----------
    summary = []
    for bench, rec in bench_recs.items():
        n_total, n_valid = len(rec), int(rec.valid.sum())
        if args.graph_repr == 'morgan':
            ev = rec[rec.valid].reset_index(drop=True)
        else:
            ev = rec[(rec.valid) & (rec.in_gallery)].reset_index(drop=True)
        if args.max_queries > 0:
            ev = ev.iloc[:args.max_queries]
        _img_dir = f'{args.real_root}/curated/{bench}/images'
        _probe = RealImageDataset(ev, _img_dir)
        _keep = [i for i, r in ev.iterrows() if _probe.resolve(r['image_id'])]
        if len(_keep) < len(ev):
            print(f'[{bench}] skipped {len(ev) - len(_keep)} records with missing images')
        ev = ev.loc[_keep].reset_index(drop=True)
        ds = RealImageDataset(ev, _img_dir)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

        img_list = []
        with torch.no_grad():
            for images in tqdm(loader, desc=f'{bench} query embed', leave=False):
                images = images.to(args.device)
                with torch.amp.autocast(device_type='cuda' if amp_on else 'cpu',
                                        dtype=torch.bfloat16, enabled=amp_on):
                    ie = model.vision_encoder(images)
                img_list.append(ie.float().cpu())
        q = F.normalize(torch.cat(img_list, 0), dim=-1).to(args.device)

        sims = q @ gallery.t()
        top5 = sims.topk(5, dim=1).indices.cpu().numpy()

        gt_stats = [mol_stats(s) for s in ev['gt_smiles_iso']]
        rows = []
        r1 = r5 = 0
        ring_m, atom_m, bond_m, arom_m, tani = [], [], [], [], []
        for qi in range(len(ev)):
            gt_ns = ev.iloc[qi]['gt_smiles_nostereo']
            cand = top5[qi]
            hit1 = g_ns[cand[0]] == gt_ns
            hit5 = any(g_ns[c] == gt_ns for c in cand)
            r1 += hit1
            r5 += hit5
            rt = mol_stats(g_keys[cand[0]])
            gt = gt_stats[qi]
            if gt is not None and rt is not None:
                ring_m.append(float(gt['rings'] == rt['rings']))
                atom_m.append(float(gt['atoms'] == rt['atoms']))
                bond_m.append(float(gt['bonds'] == rt['bonds']))
                arom_m.append(float(gt['aromatic_rings'] == rt['aromatic_rings']))
                tani.append(tanimoto(gt, rt))
            rows.append({'image_id': ev.iloc[qi]['image_id'], 'r@1': bool(hit1),
                         'r@5': bool(hit5),
                         'hardcase_labels': ev.iloc[qi].get('hardcase_labels', ''),
                         'evaluation_subset': ev.iloc[qi].get('evaluation_subset', '')})
        m = len(ev)
        s = {'ckpt': tag, 'graph_repr': args.graph_repr, 'bench': bench,
             'n_total': n_total, 'n_valid': n_valid, 'n_eval': m,
             'coverage': (m / n_valid) if n_valid else 0.0,
             'R@1': r1 / max(m, 1) * 100, 'R@5': r5 / max(m, 1) * 100,
             'ring_match': float(np.mean(ring_m)) if ring_m else np.nan,
             'atom_match': float(np.mean(atom_m)) if atom_m else np.nan,
             'bond_match': float(np.mean(bond_m)) if bond_m else np.nan,
             'aromatic_ring_match': float(np.mean(arom_m)) if arom_m else np.nan,
             'mean_tanimoto': float(np.mean(tani)) if tani else np.nan}
        summary.append(s)
        pd.DataFrame(rows).to_csv(f'{args.out_dir}/per_sample_{tag}_{bench}.csv',
                                  index=False)
        print(f"[{bench}] eval={m}/{n_valid} (coverage {s['coverage'] * 100:.1f}%) "
              f"R@1={s['R@1']:.2f}% R@5={s['R@5']:.2f}% ring={s['ring_match']:.4f} "
              f"tani={s['mean_tanimoto']:.4f}")

    out_csv = f'{args.out_dir}/summary_{tag}_{args.graph_repr}.csv'
    pd.DataFrame(summary).to_csv(out_csv, index=False)
    print(f'saved: {out_csv}')


if __name__ == '__main__':
    main()
