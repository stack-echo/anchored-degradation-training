#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Quantify exact and near-duplicate overlap between the 200k training-pool
structures (superset of the 193k training split) and each real-world benchmark's
ground-truth structures.

Exact overlap: stereo-stripped canonical SMILES identity.
Near-duplicate: max Tanimoto (Morgan r2, 2048 bits) against any training structure >= 0.9.

Run from the repository root. Pure CPU/RDKit.

Usage:
  python overlap_quant.py --real_root data/real_benchmarks \
      --csv_200k data/my_200k_dataset.csv
"""
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog('rdApp.*')

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_real import nostereo

NEAR_DUP_T = 0.9

_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--real_root', default='data/real_benchmarks',
                   help='root with curated/<bench>/{records.csv,images/}')
    p.add_argument('--csv_200k', default='data/my_200k_dataset.csv')
    p.add_argument('--benches', default='CLEF,UOB,USPTO,JPO,ACS,Staker,MolRecBenchWild')
    p.add_argument('--out_csv', default='results_real/overlap_quant.csv')
    return p.parse_args()


def morgan_fp(smiles):
    m = Chem.MolFromSmiles(str(smiles))
    return _MORGAN.GetFingerprint(m) if m is not None else None


def main():
    args = parse_args()
    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(args.csv_200k)
    train_smiles = [str(s) for s in df['canonical_smiles']]
    train_ns = set()
    train_fps = []
    for s in tqdm(train_smiles, desc='train fps'):
        ns = nostereo(s)
        fp = morgan_fp(s)
        if ns is not None and fp is not None:
            train_ns.add(ns)
            train_fps.append(fp)
    print(f'train pool: {len(train_fps)} valid structures')

    rows = []
    for bench in args.benches.split(','):
        rec = pd.read_csv(f'{args.real_root}/curated/{bench}/records.csv')
        gts = sorted({s for s in rec.loc[rec.valid, 'gt_smiles_iso'].dropna()})
        n_exact = 0
        n_near = 0
        n_ok = 0
        for s in tqdm(gts, desc=bench, leave=False):
            ns = nostereo(s)
            fp = morgan_fp(s)
            if ns is None or fp is None:
                continue
            n_ok += 1
            if ns in train_ns:
                n_exact += 1
                n_near += 1  # exact match is also >= 0.9
                continue
            sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
            if max(sims) >= NEAR_DUP_T:
                n_near += 1
        rows.append({'bench': bench, 'n_gt': len(gts), 'n_valid_gt': n_ok,
                     'exact_overlap': n_exact, 'exact_pct': 100 * n_exact / max(n_ok, 1),
                     'neardup_0.9': n_near, 'neardup_pct': 100 * n_near / max(n_ok, 1)})
        r = rows[-1]
        print(f"[{bench}] gt={r['n_gt']} exact={n_exact} ({r['exact_pct']:.1f}%) "
              f"near-dup>=0.9: {n_near} ({r['neardup_pct']:.1f}%)")

    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f'saved: {args.out_csv}')


if __name__ == '__main__':
    main()
