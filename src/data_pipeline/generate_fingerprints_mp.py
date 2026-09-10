import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import multiprocessing as mp
import os
from rdkit.Chem import rdFingerprintGenerator

# 生成 2048 维的 Morgan 指纹
FP_SIZE = 2048
FP_RADIUS = 2

morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_SIZE)


def process_single_fp(data_tuple):
    idx, smi, chembl_id = data_tuple
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return idx, chembl_id, None
    # 生成指纹
    fp = morgan_generator.GetFingerprint(mol)
    fp_arr = np.zeros((0,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, fp_arr)
    return idx, chembl_id, fp_arr


def build_fingerprints(df, num_workers=None):
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 2)

    print(f"🚀 启动指纹计算引擎 ({num_workers} Workers)...")

    tasks = [(idx, row['canonical_smiles'], row['chembl_id']) for idx, row in df.iterrows()]

    fp_dict = {}
    with mp.Pool(num_workers) as pool:
        chunk_size = max(1, len(tasks) // (num_workers * 4))
        for res in tqdm(pool.imap_unordered(process_single_fp, tasks, chunksize=chunk_size),
                        total=len(tasks), desc="🧬 Morgan 指纹计算"):
            idx, chembl_id, fp_arr = res
            if fp_arr is not None:
                fp_dict[str(chembl_id)] = fp_arr

    return fp_dict

if __name__ == "__main__":
    df = pd.read_csv("../../data/my_200k_dataset.csv")

    fp_dict = build_fingerprints(df)

    # 保存为高度压缩的 .npz 格式 (PyTorch 加载极快，20万条大概只需几百MB)
    os.makedirs("../../data/processed", exist_ok=True)
    save_path = "../../data/processed/morgan_fingerprints.npz"
    print(f"\n💾 正在保存至 {save_path} ...")
    np.savez_compressed(save_path, **fp_dict)
    print("🎉 指纹库建立完成！")