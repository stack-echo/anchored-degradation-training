import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.models.adt_net import ADTNet
from src.data_pipeline.dataloader import (MoleculeRetrievalDataset,
                                          get_degradation_transform)
from train import evaluate  # shared evaluation logic with the training script


def parse_args():
    parser = argparse.ArgumentParser(
        description="Five-level degradation benchmark evaluation.")

    # Core evaluation parameters
    parser.add_argument('--ckpt', type=str, required=True,
                        help="Path to the trained model weights")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=16)

    # Architecture synchronization (must match the checkpoint)
    parser.add_argument('--vision_backbone', type=str, default='resnet50')
    parser.add_argument('--graph_type', type=str, default='fastrp',
                        choices=['fastrp', 'morgan'])
    parser.add_argument('--embed_dim', type=int, default=512)

    # Mode synchronization (tensor-side interceptors must match training)
    parser.add_argument('--mode', type=str, default='adt',
                        choices=['adt', 'clean', 'mae_mask', 'random_erasing',
                                 'augmix', 'mixtrain',
                                 'random_graph', 'symmetric_noise'])

    parser.add_argument('--toy_mode', action='store_true',
                        help="Quick sanity check on the toy dataset.")
    parser.add_argument('--csv_path', type=str, default='data/my_200k_dataset.csv')

    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 85)
    print("Five-Level Degradation Benchmark Evaluation".center(85))
    print(f"Checkpoint: {args.ckpt}".center(85))
    print(f"Backbone: {args.vision_backbone.upper()} | Mode: {args.mode.upper()}".center(85))
    print("=" * 85 + "\n")

    if args.toy_mode:
        test_df = pd.read_csv("data/toy_dataset/toy_split.csv")
        fp_path = "data/toy_dataset/toy_fingerprints.npz"
        base_img_dir = "data/toy_dataset/images"
        graph_path = "data/toy_dataset/toy_fastrp_embeddings.npz"
    else:
        # The last 5,000 rows form the held-out test split
        df = pd.read_csv(args.csv_path)
        test_df = df.iloc[-5000:]
        fp_path = "data/processed/morgan_fingerprints.npz"
        graph_path = "data/processed/fastrp_embeddings.npz"

    model = ADTNet(vision_backbone=args.vision_backbone,
                   graph_type=args.graph_type,
                   embed_dim=args.embed_dim).to(args.device)
    model.load_state_dict(torch.load(args.ckpt, map_location=args.device))
    print(f"Successfully loaded weights: {args.ckpt}\n")

    results = {}
    for level in range(5):
        if args.toy_mode:
            # On-the-fly degradation for the toy sanity check
            print(f"Evaluating level {level} (online degradation)...")
            eval_transform = get_degradation_transform(level=level, for_saving=False)
            current_img_dir = base_img_dir
        else:
            # Static pre-degraded benchmark: every model sees byte-identical inputs
            test_dir = f"data/benchmark/level_{level}"
            print(f"Evaluating level {level} benchmark ({test_dir})...")
            eval_transform = get_degradation_transform(level=0, for_saving=False)
            current_img_dir = test_dir

        dataset = MoleculeRetrievalDataset(test_df, current_img_dir, fp_path,
                                           graph_path=graph_path,
                                           transform=eval_transform)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

        metrics = evaluate(model, loader, args.device, args)
        results[f"Level_{level}"] = metrics

    print("\n" + "Degradation-Axis Report".center(85))
    print("-" * 85)
    print(f"| {'Metric':<10} | {'L0 (Clean)':<10} | {'L1 (Mild)':<10} | "
          f"{'L2 (Mod)':<10} | {'L3 (Sev)':<10} | {'L4 (Ext)':<10} |")
    print("-" * 85)
    for k in ['I2G_R@1', 'I2G_R@5', 'G2I_R@1']:
        row_str = f"| {k:<10} | "
        for level in range(5):
            row_str += f"{results[f'Level_{level}'][k]:.2f}%".ljust(10) + " | "
        print(row_str)
    print("-" * 85 + "\n")


if __name__ == "__main__":
    main()
