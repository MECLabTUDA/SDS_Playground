import os
import time

import torch
import numpy as np
from tqdm import tqdm
from torchvision.io import write_png
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
from concurrent.futures import ThreadPoolExecutor, as_completed

from sds_playground.datasets.cholecseg8k.cholecseg8k_dataset import CholecSeg8kDataset

TGT_SIZE = [128, 128]
# TGT_SIZE = [256, 256]


def process_sample(img, mask, filename, img_target, mask_target):
    try:
        img = resize(img, size=TGT_SIZE, interpolation=InterpolationMode.BILINEAR)
        img = (img * 255).to(torch.uint8)
        mask = resize(mask.unsqueeze(0), size=TGT_SIZE, interpolation=InterpolationMode.NEAREST)
        mask = mask.to(torch.uint8)
        write_png(img, filename=os.path.join(img_target, filename))
        write_png(mask, filename=os.path.join(mask_target, filename.replace('.png', '_mask.png')))
        np.save(os.path.join(mask_target,filename.replace('.png', '_mask.npy')), mask.squeeze().cpu().numpy())
    except Exception as e:
        print(f"Error processing {filename}: {e}")


def process_split(ds, split, target):
    print(f"{split}-split contains {len(ds)} samples")
    img_target = os.path.join(target, f"{split}_images")
    os.makedirs(img_target, exist_ok=True)
    mask_target = os.path.join(target, f"{split}_masks")
    os.makedirs(mask_target, exist_ok=True)

    time.sleep(0.1)

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = []
        for i in tqdm(range(len(ds)), desc=f"Converting {split} split"):
            try:
                img, mask, filename, _ = ds[i]
            except Exception as e:
                print(f"Error loading sample {i}: {e}")
            try:
                futures.append(executor.submit(process_sample, img, mask, filename, img_target, mask_target))
            except Exception as e:
                print(f"Error loading sample {filename}: {e}")

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {split} split"):
            try:
                future.result()  # We call result to raise any exceptions caught during execution
            except Exception as e:
                print(f"Error in future task: {e}")


if __name__ == "__main__":

    root = '/home/yfrisch_locale/DATA/CholecSeg8k/'
    target = f'/home/yfrisch_locale/DATA/CholecSeg8k_{TGT_SIZE[0]}pix_datadir/'
    assert os.path.isdir(root)
    os.makedirs(target, exist_ok=True)

    train_ds = CholecSeg8kDataset(
        root=root,
        mode='train',
        sample_img=True,
        sample_mask=True
    )

    val_ds = CholecSeg8kDataset(
        root=root,
        mode='val',
        sample_img=True,
        sample_mask=True
    )

    test_ds = CholecSeg8kDataset(
        root=root,
        mode='test',
        sample_img=True,
        sample_mask=True
    )

    for ds, split in zip([train_ds, val_ds, test_ds], ['train', 'val', 'test']):
        process_split(ds, split, target)

