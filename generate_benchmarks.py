import os

import cv2
import pandas as pd
from tqdm import tqdm

from src.data_pipeline.dataloader import get_degradation_transform


def generate_test_sets():
    """Materialize the static five-level benchmark.

    The held-out 5,000 test images are degraded once, offline, at each level,
    so every evaluated model sees byte-identical inputs.
    """
    df = pd.read_csv("data/my_200k_dataset.csv")
    test_df = df.iloc[-5000:]  # the last 5,000 rows form the held-out test split
    src_dir = "data/pretrain_images"

    print("Generating the five-level static degradation benchmark...")

    for level in range(5):
        # for_saving=True returns numpy images (no tensor conversion)
        transform = get_degradation_transform(level=level, for_saving=True)
        target_dir = f"data/benchmark/level_{level}"
        os.makedirs(target_dir, exist_ok=True)

        print(f"\nLevel {level} -> {target_dir}")
        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            chembl_id = str(row['chembl_id'])
            src_path = os.path.join(src_dir, f"{chembl_id}.png")

            img = cv2.imread(src_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                degraded_img = transform(image=img)['image']
                degraded_img = cv2.cvtColor(degraded_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(target_dir, f"{chembl_id}.png"), degraded_img)


if __name__ == "__main__":
    generate_test_sets()
