#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Topology-preservation probe: direct structural-invariant metrics.

For each query image, retrieve the nearest molecular graph and compare
graph invariants (ring count, atom count, bond count, aromatic rings,
Morgan Tanimoto) between the retrieved and ground-truth molecules.
This directly tests whether topology is preserved under degradation.

No training required. Needs rdkit. Run from the repository root.

Example:
    python eval_topology.py \
        --ckpt checkpoints/convnext_base_adt_l4_seed42_best.pth \
        --vision_backbone convnext_base \
        --levels 0,3,4 \
        --csv data/my_200k_dataset.csv \
        --img_dir data/pretrain_images \
        --fp data/processed/morgan_fingerprints.npz \
        --graph data/processed/fastrp_embeddings.npz \
        --device mps
"""
import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import DataStructs, rdMolDescriptors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.adt_net import ADTNet
from src.data_pipeline.dataloader import MoleculeRetrievalDataset, get_degradation_transform
from src.utils.metrics import calculate_retrieval_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--vision_backbone', type=str, default='convnext_base')
    p.add_argument('--graph_type', type=str, default='fastrp')
    p.add_argument('--embed_dim', type=int, default=512)
    p.add_argument('--levels', type=str, default='0,3,4',
                   help="Comma-separated test degradation levels from the paper's 5-level schedule")
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--device', type=str,
                   default='cuda' if torch.cuda.is_available() else
                           ('mps' if torch.backends.mps.is_available() else 'cpu'))
    p.add_argument('--csv', type=str, required=True)
    p.add_argument('--img_dir', type=str, required=True)
    p.add_argument('--fp', type=str, required=True)
    p.add_argument('--graph', type=str, required=True)
    p.add_argument('--out_dir', type=str, default='results_topology')
    p.add_argument('--max_samples', type=int, default=0,
                   help="If >0, use only the first N test samples (smoke test).")
    return p.parse_args()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    img_list, graph_list = [], []
    use_amp = device.startswith('cuda')
    for images, graph_feats, fps, _ in tqdm(loader, desc="  eval", leave=False):
        images = images.to(device)
        graph_feats = graph_feats.to(device)
        if use_amp:
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                img_embed, graph_embed, _ = model(images, graph_feats)
        else:
            img_embed, graph_embed, _ = model(images, graph_feats)
        img_list.append(img_embed.float().cpu())
        graph_list.append(graph_embed.float().cpu())

    img_tensor = F.normalize(torch.cat(img_list, dim=0), dim=-1)
    graph_tensor = F.normalize(torch.cat(graph_list, dim=0), dim=-1)
    return calculate_retrieval_metrics(img_tensor, graph_tensor)


def mol_stats(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        'mol': mol,
        'rings': mol.GetRingInfo().NumRings(),
        'atoms': mol.GetNumAtoms(),
        'bonds': mol.GetNumBonds(),
        'aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
    }


def tanimoto(s1, s2):
    fp1 = rdMolDescriptors.GetMorganFingerprintAsBitVect(s1['mol'], 2, nBits=2048)
    fp2 = rdMolDescriptors.GetMorganFingerprintAsBitVect(s2['mol'], 2, nBits=2048)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    test_df = df.iloc[-5000:].reset_index(drop=True)
    if args.max_samples > 0:
        test_df = test_df.iloc[:args.max_samples].reset_index(drop=True)
        print(f"SMOKE TEST: using only {len(test_df)} samples")
    smiles_list = [str(s) for s in test_df['canonical_smiles']]
    print("Pre-parsing ground-truth molecules with RDKit...")
    gt_stats = [mol_stats(s) for s in tqdm(smiles_list)]

    model = ADTNet(vision_backbone=args.vision_backbone,
                   graph_type=args.graph_type,
                   embed_dim=args.embed_dim).to(args.device)
    model.load_state_dict(torch.load(args.ckpt, map_location=args.device))
    print(f"Checkpoint loaded: {args.ckpt}")

    tag = os.path.splitext(os.path.basename(args.ckpt))[0]
    summary_rows = []

    for level in [int(x) for x in args.levels.split(',')]:
        set_seed(42)
        transform = get_degradation_transform(level=level, for_saving=False)
        dataset = MoleculeRetrievalDataset(test_df, args.img_dir, args.fp,
                                           graph_path=args.graph, transform=transform)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
        metrics = evaluate(model, loader, args.device)

        # Retrieval: nearest graph for each image
        img_list, graph_list = [], []
        # re-run embedding extraction is wasteful; evaluate() already computed
        # retrieval metrics, but we need the indices, so recompute embeddings here
        model.eval()
        with torch.no_grad():
            for images, graph_feats, fps, _ in tqdm(loader, desc=f"  embed L{level}", leave=False):
                images = images.to(args.device)
                graph_feats = graph_feats.to(args.device)
                if args.device.startswith('cuda'):
                    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                        ie, ge, _ = model(images, graph_feats)
                else:
                    ie, ge, _ = model(images, graph_feats)
                img_list.append(ie.float().cpu())
                graph_list.append(ge.float().cpu())
        img_tensor = F.normalize(torch.cat(img_list, dim=0), dim=-1)
        graph_tensor = F.normalize(torch.cat(graph_list, dim=0), dim=-1)
        nearest = (img_tensor @ graph_tensor.t()).argmax(dim=1).numpy()

        ring_m, atom_m, bond_m, arom_m, tani = [], [], [], [], []
        for qi, ri in enumerate(nearest):
            gt, rt = gt_stats[qi], gt_stats[ri]
            if gt is None or rt is None:
                continue
            ring_m.append(float(gt['rings'] == rt['rings']))
            atom_m.append(float(gt['atoms'] == rt['atoms']))
            bond_m.append(float(gt['bonds'] == rt['bonds']))
            arom_m.append(float(gt['aromatic_rings'] == rt['aromatic_rings']))
            tani.append(tanimoto(gt, rt))

        row = {
            'ckpt': tag, 'test_level': level,
            'I2G_R@1': metrics['I2G_R@1'],
            'ring_match': np.mean(ring_m), 'atom_match': np.mean(atom_m),
            'bond_match': np.mean(bond_m), 'aromatic_ring_match': np.mean(arom_m),
            'mean_tanimoto': np.mean(tani), 'n': len(tani),
        }
        summary_rows.append(row)
        pd.DataFrame({'ring_match': ring_m, 'atom_match': atom_m,
                      'bond_match': bond_m, 'aromatic_ring_match': arom_m,
                      'tanimoto': tani}).to_csv(
            os.path.join(args.out_dir, f"topology_persample_{tag}_L{level}.csv"),
            index=False)
        print(f"\n[L{level}] R@1={row['I2G_R@1']:.2f}% | ring={row['ring_match']:.4f} "
              f"atom={row['atom_match']:.4f} bond={row['bond_match']:.4f} "
              f"arom={row['aromatic_ring_match']:.4f} tanimoto={row['mean_tanimoto']:.4f}")

    out_path = os.path.join(args.out_dir, f"topology_{tag}.csv")
    pd.DataFrame(summary_rows).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
