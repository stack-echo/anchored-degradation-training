#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Style-diverse re-rendering of the 200k ChEMBL training set.
Randomizes RDKit drawing options per molecule: monochrome vs colored palette,
bond line width, label font size, bond spacing/offset, padding, plus mild
rotation/scale via PIL. Output: data/style_images/<chembl_id>.png (224x224),
same naming as data/pretrain_images so datasets can mix sources.
"""
import os, random
import numpy as np
import pandas as pd
import multiprocessing as mp
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from PIL import Image
import io

CSV = 'data/my_200k_dataset.csv'
OUT = 'data/style_images'
os.makedirs(OUT, exist_ok=True)


def render_one(task):
    chembl_id, smi, seed = task
    rng = random.Random(seed)
    out_path = os.path.join(OUT, f'{chembl_id}.png')
    if os.path.exists(out_path):
        return None
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None

    drawer = rdMolDraw2D.MolDraw2DCairo(224, 224)
    opts = drawer.drawOptions()
    # monochrome (black) vs colored atom palette
    if rng.random() < 0.65:
        opts.useBWAtomPalette()
    # line/label geometry
    opts.bondLineWidth = rng.choice([1.0, 1.5, 2.0, 2.5, 3.0])
    opts.multipleBondOffset = rng.uniform(0.10, 0.22)
    opts.padding = rng.uniform(0.02, 0.12)
    opts.minFontSize = rng.randint(8, 14)
    opts.maxFontSize = rng.randint(14, 22)
    # bond double-line style
    opts.multipleBondOffset = rng.uniform(0.08, 0.24)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    img = Image.open(io.BytesIO(drawer.GetDrawingText())).convert('RGB')

    # mild geometric jitter (rotation / slight non-uniform scale)
    if rng.random() < 0.5:
        img = img.rotate(rng.uniform(-12, 12), resample=Image.BILINEAR,
                         fillcolor=(255, 255, 255))
    img.save(out_path)
    return None


def main():
    df = pd.read_csv(CSV)
    tasks = [(str(r['chembl_id']), str(r['canonical_smiles']), i)
             for i, (_, r) in enumerate(df.iterrows())]
    print(f'rendering {len(tasks)} molecules with style diversity...')
    with mp.Pool(max(1, mp.cpu_count() - 4)) as pool:
        for i, _ in enumerate(pool.imap_unordered(render_one, tasks, chunksize=64)):
            if (i + 1) % 10000 == 0:
                print(f'{i + 1}/{len(tasks)}', flush=True)
    print('done')


if __name__ == '__main__':
    main()
