#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Open-set probe: hold out a fraction of each benchmark's ground-truth structures
from the retrieval gallery, then measure (i) separability of covered vs uncovered
queries by top-1 similarity (AUROC) and (ii) a precision-coverage curve obtained by
thresholding top-1 similarity (abstention).

Run from the repository root (cwd must contain data/ and eval_real.py).

Usage:
  python eval_openset.py --ckpt checkpoints/style_adt_morgan_seed_42_best.pth \
      --graph_repr morgan --vision_backbone convnext_base --holdout_frac 0.5 --seed 42
"""
import os, argparse, random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_real import build_gallery, nostereo, RealImageDataset
from src.models.adt_net import ADTNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--graph_repr', default='morgan', choices=['morgan'])
    p.add_argument('--vision_backbone', default='convnext_base')
    p.add_argument('--embed_dim', type=int, default=512)
    p.add_argument('--real_root', default='data/real_benchmarks',
                   help='root with curated/<bench>/{records.csv,images/}')
    p.add_argument('--csv_200k', default='data/my_200k_dataset.csv')
    p.add_argument('--fp_npz', default='data/processed/morgan_fingerprints.npz')
    p.add_argument('--benches', default='CLEF,UOB,USPTO,JPO,ACS,Staker,MolRecBenchWild')
    p.add_argument('--holdout_frac', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out_dir', default='results_openset')
    p.add_argument('--max_queries', type=int, default=0)
    return p.parse_args()


def auroc(scores, labels):
    """Rank-based AUROC with tie handling. labels: 1=covered(positive), 0=uncovered."""
    s = pd.Series(scores)
    ranks = s.rank(method='average').to_numpy()
    labels = np.asarray(labels)
    P = int(labels.sum()); N = int(len(labels) - P)
    if P == 0 or N == 0:
        return float('nan')
    return float((ranks[labels == 1].sum() - P * (P + 1) / 2) / (P * N))


def precision_coverage(sim, correct):
    """Sort by similarity desc; return arrays (coverage, precision) at each prefix,
    plus precision at target coverage levels."""
    order = np.argsort(-np.asarray(sim))
    c = np.asarray(correct)[order].astype(float)
    n = len(c)
    cum = np.cumsum(c)
    kept = np.arange(1, n + 1)
    prec = cum / kept
    cov = kept / n
    targets = {}
    for t in (1.0, 0.95, 0.9, 0.8, 0.7, 0.5):
        k = int(round(t * n))
        k = max(k, 1)
        targets[f'precision_at_cov{int(t*100)}'] = float(prec[k - 1])
    return cov, prec, targets


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tag = os.path.splitext(os.path.basename(args.ckpt))[0]

    bench_recs = {}
    for bench in args.benches.split(','):
        bench_recs[bench] = pd.read_csv(f'{args.real_root}/curated/{bench}/records.csv')

    # per-benchmark holdout split of unique GT structures
    rng = random.Random(args.seed)
    holdout = set()
    bench_holdout = {}
    for bench, rec in bench_recs.items():
        gts = sorted({s for s in rec.loc[rec.valid, 'gt_smiles_iso'].dropna()})
        k = int(round(len(gts) * args.holdout_frac))
        h = set(rng.sample(gts, k))
        bench_holdout[bench] = h
        holdout |= h
    extra_all = sorted({s for rec in bench_recs.values()
                        for s in rec.loc[rec.valid, 'gt_smiles_iso'].dropna()})
    extra_kept = [s for s in extra_all if s not in holdout]
    print(f'GT structures total={len(extra_all)}, held out={len(holdout)}, kept={len(extra_kept)}')

    g_keys, g_mat = build_gallery(args, extra_kept)
    g_ns = [nostereo(k) for k in tqdm(g_keys, desc='gallery nostereo')]
    g_ns_set = set(g_ns)
    print(f'gallery (holdout applied): {len(g_keys)} structures')

    model = ADTNet(vision_backbone=args.vision_backbone, graph_type='morgan',
                   embed_dim=args.embed_dim).to(args.device)
    model.load_state_dict(torch.load(args.ckpt, map_location=args.device))
    model.eval()
    print(f'checkpoint loaded: {args.ckpt}')

    cache = f'{args.out_dir}/gallery_embed_{tag}_{len(g_keys)}.npy'
    if os.path.exists(cache):
        gallery = torch.from_numpy(np.load(cache)).to(args.device)
    else:
        g_list = []
        with torch.no_grad():
            for i in tqdm(range(0, len(g_keys), 4096), desc='gallery embed'):
                gb = torch.from_numpy(g_mat[i:i + 4096]).to(args.device)
                if args.device.startswith('cuda'):
                    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                        ge = model.graph_encoder(gb)
                else:
                    ge = model.graph_encoder(gb)
                g_list.append(ge.float().cpu())
        gallery = F.normalize(torch.cat(g_list, 0), dim=-1)
        np.save(cache, gallery.numpy())
        gallery = gallery.to(args.device)

    summary = []
    for bench, rec in bench_recs.items():
        ev = rec[rec.valid].reset_index(drop=True)
        if args.max_queries > 0:
            ev = ev.iloc[:args.max_queries]
        _img_dir = f'{args.real_root}/curated/{bench}/images'
        _probe = RealImageDataset(ev, _img_dir)
        _keep = [i for i, r in ev.iterrows() if _probe.resolve(r['image_id'])]
        ev = ev.loc[_keep].reset_index(drop=True)
        ds = RealImageDataset(ev, _img_dir)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
        img_list = []
        with torch.no_grad():
            for images in tqdm(loader, desc=f'{bench} query embed', leave=False):
                images = images.to(args.device)
                if args.device.startswith('cuda'):
                    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                        ie = model.vision_encoder(images)
                else:
                    ie = model.vision_encoder(images)
                img_list.append(ie.float().cpu())
        q = F.normalize(torch.cat(img_list, 0), dim=-1).to(args.device)
        ts_list, ti_list = [], []
        with torch.no_grad():
            for i in range(0, q.shape[0], 512):
                s = q[i:i + 512] @ gallery.t()
                ts, ti = s.max(dim=1)
                ts_list.append(ts.float().cpu())
                ti_list.append(ti.cpu())
        top1_sim = torch.cat(ts_list).numpy()
        top1_idx = torch.cat(ti_list).numpy()

        covered = np.array([ns in g_ns_set for ns in ev['gt_smiles_nostereo']], dtype=bool)
        correct = np.array([g_ns[top1_idx[i]] == ev.iloc[i]['gt_smiles_nostereo']
                            for i in range(len(ev))], dtype=bool)

        a = auroc(top1_sim, covered.astype(int))
        cov_curve, prec_curve, targets = precision_coverage(top1_sim, correct)
        r1_covered = float(correct[covered].mean()) if covered.any() else float('nan')
        sim_cov = top1_sim[covered]; sim_unc = top1_sim[~covered]

        pd.DataFrame({'image_id': ev['image_id'], 'top1_sim': top1_sim,
                      'covered': covered, 'correct': correct}).to_csv(
            f'{args.out_dir}/per_query_{tag}_{bench}.csv', index=False)
        pd.DataFrame({'coverage': cov_curve, 'precision': prec_curve}).to_csv(
            f'{args.out_dir}/pc_curve_{tag}_{bench}.csv', index=False)

        s = {'bench': bench, 'n_eval': len(ev),
             'n_covered': int(covered.sum()), 'n_uncovered': int((~covered).sum()),
             'auroc_top1sim': a, 'r1_covered': r1_covered,
             'sim_covered_mean': float(sim_cov.mean()) if covered.any() else float('nan'),
             'sim_uncovered_mean': float(sim_unc.mean()) if (~covered).any() else float('nan'),
             **targets}
        summary.append(s)
        print(f"[{bench}] n={len(ev)} covered={s['n_covered']} uncovered={s['n_uncovered']} "
              f"AUROC={a:.3f} R@1|covered={r1_covered*100:.2f}% "
              f"simCov={s['sim_covered_mean']:.3f} simUnc={s['sim_uncovered_mean']:.3f} "
              f"P@cov90={targets['precision_at_cov90']*100:.1f}%")

    out_csv = f'{args.out_dir}/summary_{tag}_holdout{args.holdout_frac}_seed{args.seed}.csv'
    pd.DataFrame(summary).to_csv(out_csv, index=False)
    print(f'saved: {out_csv}')


if __name__ == '__main__':
    main()
