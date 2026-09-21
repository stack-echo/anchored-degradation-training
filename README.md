# Anchored Degradation Training for Robust Molecular Image Recognition

Official code for **"Anchored Degradation Training for Robust Molecular Image Recognition via Graph-Anchored Retrieval"** (under review).

**Anchored Degradation Training (ADT)** is a training strategy for retrieval-based molecular image recognition (OCSR). The image encoder is trained along a controlled degradation axis while the molecular graph encoder remains anchored to clean structural ground truth. Recognition is formulated as retrieval against a validated gallery of molecular graphs rather than autoregressive decoding, so outputs are deterministic and chemically valid by construction.

<p align="center">
  <b>ConvNeXt-Base, I2G R@1 (%), extreme test degradation (L<sub>4</sub>), mean over 3 seeds</b><br>
  clean-trained 5.93 &nbsp;·&nbsp; masked-patch 8.92 &nbsp;·&nbsp; random-erasing 8.71 &nbsp;·&nbsp; <b>ADT 88.52</b>
</p>

## Installation

```bash
conda create -n adt python=3.10 -y && conda activate adt
pip install -r requirements.txt
```

Tested with PyTorch 2.x on a single NVIDIA GPU (mixed precision, bf16). CPU works for smoke tests.

## Data preparation

The training collection is 200,000 molecule–image pairs derived from [ChEMBL](https://www.ebi.ac.uk/chembl/) (canonical SMILES + rendered 2D depictions), split into 193,000 train / 2,000 validation / 5,000 test.

```
data/
├── my_200k_dataset.csv            # chembl_id, canonical_smiles (ordered: train | val | test)
├── pretrain_images/<id>.png       # rendered 2D depictions
├── style_images/<id>.png          # style-diverse re-renders (render_styles.py)
├── processed/
│   ├── morgan_fingerprints.npz    # 2048-bit Morgan fingerprints per chembl_id
│   └── fastrp_embeddings.npz      # 128-d 3-order FastRP embeddings per SMILES
└── benchmark/level_{0..4}/        # static five-level degraded test sets
```

Pipeline:

```bash
python src/data_pipeline/graph_builder.py               # molecule–scaffold graphs
python src/data_pipeline/generate_fastrp.py             # FastRP embeddings
python src/data_pipeline/generate_fingerprints_mp.py    # Morgan fingerprints
python render_styles.py                                 # style-diverse re-rendering
python generate_benchmarks.py                           # static L0–L4 test sets
```

## Training

```bash
# ADT (ours), anchored at the extreme level L4
python train.py --mode adt --anchor_level 4 --vision_backbone convnext_base --seed 42

# Baselines
python train.py --mode clean           --vision_backbone convnext_base --seed 42
python train.py --mode mae_mask        --vision_backbone convnext_base --seed 42
python train.py --mode random_erasing  --vision_backbone convnext_base --seed 42
python train.py --mode augmix          --vision_backbone convnext_base --seed 42
python train.py --mode mixtrain        --vision_backbone convnext_base --seed 42

# Style-diverse training (Section: real-world evaluation)
python train.py --mode adt --anchor_level 4 --graph_type morgan \
    --img_dirs data/pretrain_images,data/style_images --vision_backbone convnext_base --seed 42
```

All models share the same optimization schedule: AdamW (lr 2.5e-5, weight decay 1e-4), cosine annealing, batch size 256, 20 epochs, symmetric InfoNCE with a learnable logit scale (initial temperature 0.07). Anchor-level ablations: `--mode adt --anchor_level {0,1,2,3,4}`. Graph-side ablations: `--mode random_graph`, `--mode symmetric_noise`.

## Evaluation

```bash
# Five-level static degradation benchmark (main table)
python eval.py --ckpt checkpoints/convnext_base_adt_l4_seed42_best.pth \
    --vision_backbone convnext_base --mode adt

# Real-world OCSR benchmarks (CLEF/UOB/USPTO/JPO/ACS/Staker/MolRecBenchWild)
python eval_real.py --ckpt checkpoints/XXX.pth --graph_repr morgan \
    --vision_backbone convnext_base --real_root data/real_benchmarks

# CPU latency benchmark (deployment section)
python bench_latency.py --ckpt checkpoints/XXX.pth \
    --gallery_emb results_real/gallery_embed_XXX_morgan.npy \
    --img_dir data/real_benchmarks/curated/Staker/images --threads 4

# Topology-preservation probe (Section 4.5: ring/atom/bond/aromatic-ring
# exact match and mean Tanimoto of the top-1 retrieval, per degradation level)
python eval_topology.py --ckpt checkpoints/XXX.pth \
    --vision_backbone convnext_base --levels 0,3,4 \
    --csv data/my_200k_dataset.csv --img_dir data/pretrain_images \
    --fp data/processed/morgan_fingerprints.npz \
    --graph data/processed/fastrp_embeddings.npz

# Open-set probe (Section 4.8: covered vs uncovered separability AUROC and
# precision-coverage abstention curves after holding out part of the gallery)
python eval_openset.py --ckpt checkpoints/XXX.pth --graph_repr morgan \
    --vision_backbone convnext_base --holdout_frac 0.5 --seed 42

# Training-set / benchmark overlap quantification (Limitations; Appendix F.4:
# exact and near-duplicate Tanimoto >= 0.9 overlap against the 200k pool)
python overlap_quant.py --real_root data/real_benchmarks \
    --csv_200k data/my_200k_dataset.csv
```

## Repository layout

```
train.py                  training: ADT + all baselines/ablations
eval.py                   five-level degradation benchmark evaluation
eval_real.py              real-world OCSR benchmark evaluation
eval_topology.py          topology-preservation probe (Section 4.5)
eval_openset.py           open-set / abstention probe (Section 4.8)
overlap_quant.py          train-benchmark overlap quantification (Appendix F.4)
bench_latency.py          CPU latency benchmark
generate_benchmarks.py    static five-level test-set generation
render_styles.py          style-diverse re-rendering of training images
src/models/               ADTNet two-tower model, vision/graph encoders
src/loss/                 symmetric InfoNCE with optional hard-negative weighting
src/data_pipeline/        dataset, degradation suite, AugMix, data prep
src/utils/                retrieval metrics
```

## Checkpoint compatibility

Checkpoints released with the paper (`.pth` state dicts) load directly into `ADTNet`; the module tree (`vision_encoder`, `graph_encoder`, `logit_scale`) is unchanged from the internal training code.

## License and citation

Code is released under the MIT License (see `LICENSE`). If you use this code, please cite the paper (see `CITATION.cff`).
