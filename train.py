import os
import argparse
import random

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from src.models.adt_net import ADTNet
from src.loss.contrastive import HardNegativeInfoNCE
from src.data_pipeline.dataloader import (MoleculeRetrievalDataset,
                                          get_degradation_transform,
                                          get_random_erasing_transform)
from src.utils.metrics import calculate_retrieval_metrics


def set_seed(seed):
    """Set seeds for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training script for Anchored Degradation Training (ADT) "
                    "and all baseline / ablation configurations.")

    # Basic configuration
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=2.5e-5)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)

    # Architecture settings
    parser.add_argument('--vision_backbone', type=str, default='resnet50')
    parser.add_argument('--graph_type', type=str, default='fastrp',
                        choices=['fastrp', 'morgan'])
    parser.add_argument('--img_dirs', type=str, default='data/pretrain_images',
                        help='Comma-separated image roots; a random root is '
                             'drawn per sample (style-diverse training).')
    parser.add_argument('--embed_dim', type=int, default=512)

    # Experimental modes
    parser.add_argument('--mode', type=str, default='adt',
                        choices=['adt', 'clean', 'mae_mask', 'random_erasing',
                                 'augmix', 'mixtrain',
                                 'random_graph', 'symmetric_noise'],
                        help="Training configuration: ADT, baselines, or ablations.")

    # Hyperparameters
    parser.add_argument('--anchor_level', type=int, default=4,
                        help="Degradation level anchoring ADT training (0-4).")
    parser.add_argument('--hard_neg_margin', type=float, default=1.1)
    parser.add_argument('--push_weight', type=float, default=1.0)

    return parser.parse_args()


def build_train_transform(args):
    """Select the training-time image pipeline for the chosen mode."""
    if args.mode == 'adt':
        return get_degradation_transform(level=args.anchor_level, for_saving=False)
    if args.mode == 'augmix':
        # Faithful AugMix augmentation pipeline (train-time only)
        from src.data_pipeline.augmix import get_augmix_transform
        return get_augmix_transform()
    if args.mode == 'random_erasing':
        return get_random_erasing_transform()
    if args.mode == 'mixtrain':
        # Exposure-matched schedule: severity level resampled per image
        return [get_degradation_transform(level=lv, for_saving=False) for lv in range(5)]
    # clean / mae_mask / random_graph / symmetric_noise train on clean inputs
    # (mae_mask applies patch masking on the tensor side instead)
    return get_degradation_transform(level=0, for_saving=False)


def apply_mode_transform(images, graph_feats, fps, mode, device):
    """Tensor-side modifications for baselines and graph-side ablations."""
    if mode == 'random_graph':
        # Graph-anchor necessity ablation: replace structure with noise
        graph_feats = torch.randn_like(graph_feats)
    elif mode == 'symmetric_noise':
        # Symmetric-degradation ablation: heavy dropout on the graph side
        graph_feats = F.dropout(graph_feats, p=0.8, training=True)
    elif mode == 'mae_mask':
        # Masked-patch baseline: 75% random patch masking of the input image
        B, _, H, W = images.shape
        patch_size = 16
        num_patches_h, num_patches_w = H // patch_size, W // patch_size
        num_patches = num_patches_h * num_patches_w
        num_mask = int(0.75 * num_patches)

        mask = torch.ones(B, num_patches, device=device)
        for i in range(B):
            perm = torch.randperm(num_patches, device=device)
            mask[i, perm[:num_mask]] = 0

        mask = mask.view(B, 1, num_patches_h, num_patches_w)
        mask = F.interpolate(mask, size=(H, W), mode='nearest')
        images = images * mask

    return images, graph_feats


def amp_args(device):
    """AMP settings that also work on CPU (for local smoke tests)."""
    dev_type = 'cuda' if str(device).startswith('cuda') else 'cpu'
    return dev_type, (dev_type == 'cuda')


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, args):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch}", disable=False)
    dev_type, amp_on = amp_args(device)

    for images, graph_feats, fps, _ in pbar:
        images, graph_feats, fps = [x.to(device, non_blocking=True)
                                    for x in [images, graph_feats, fps]]

        images, graph_feats = apply_mode_transform(images, graph_feats, fps,
                                                   args.mode, device)

        optimizer.zero_grad()
        with autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=amp_on):
            img_embed, graph_embed, logit_scale = model(images, graph_feats)
            criterion.temperature = 1.0 / logit_scale.clamp(max=100)
            loss = criterion(img_embed, graph_embed, fps)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    return running_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, args):
    model.eval()
    img_list, graph_list = [], []
    dev_type, amp_on = amp_args(device)

    for images, graph_feats, fps, _ in tqdm(loader, desc="Evaluation"):
        images, graph_feats, fps = [x.to(device, non_blocking=True)
                                    for x in [images, graph_feats, fps]]

        images, graph_feats = apply_mode_transform(images, graph_feats, fps,
                                                   args.mode, device)

        with autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=amp_on):
            img_embed, graph_embed, _ = model(images, graph_feats)

        img_list.append(img_embed.float())
        graph_list.append(graph_embed.float())

    img_tensor = F.normalize(torch.cat(img_list, dim=0), dim=-1)
    graph_tensor = F.normalize(torch.cat(graph_list, dim=0), dim=-1)

    return calculate_retrieval_metrics(img_tensor, graph_tensor)


def main():
    args = parse_args()
    set_seed(args.seed)

    print(f"Starting experiment: {args.mode.upper()} with {args.vision_backbone}")

    train_transform = build_train_transform(args)
    val_transform = get_degradation_transform(level=4, for_saving=False)

    df = pd.read_csv("data/my_200k_dataset.csv")
    train_dataset = MoleculeRetrievalDataset(
        df=df.iloc[:-7000],
        img_dir=[d.strip() for d in args.img_dirs.split(',')],
        fp_path="data/processed/morgan_fingerprints.npz",
        graph_path="data/processed/fastrp_embeddings.npz",
        transform=train_transform,
        level_probs=[0.2] * 5 if args.mode == 'mixtrain' else None,
    )
    val_dataset = MoleculeRetrievalDataset(
        df=df.iloc[-7000:-5000],
        img_dir="data/pretrain_images",
        fp_path="data/processed/morgan_fingerprints.npz",
        graph_path="data/processed/fastrp_embeddings.npz",
        transform=val_transform,
    )

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True,
                              worker_init_fn=worker_init_fn, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)

    model = ADTNet(vision_backbone=args.vision_backbone,
                   graph_type=args.graph_type,
                   embed_dim=args.embed_dim).to(args.device)
    criterion = HardNegativeInfoNCE(hard_neg_margin=args.hard_neg_margin,
                                    push_weight=args.push_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(device='cuda', enabled=str(args.device).startswith('cuda'))

    os.makedirs("checkpoints", exist_ok=True)
    exp_name = f"{args.vision_backbone}_{args.mode}_l{args.anchor_level}_seed{args.seed}"

    best_r1 = 0.0
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler,
                               args.device, epoch, args)
        scheduler.step()
        metrics = evaluate(model, val_loader, args.device, args)

        current_r1 = metrics['I2G_R@1']
        print(f"Epoch {epoch}: Loss {loss:.4f} | Val R@1 {current_r1:.2f}%")

        if current_r1 > best_r1:
            best_r1 = current_r1
            torch.save(model.state_dict(), f"checkpoints/{exp_name}_best.pth")


if __name__ == "__main__":
    main()
