""" Converts CholecSeg8k hex color masks to integer id masks """

import multiprocessing
import os
from glob import glob

import torch
import numpy as np
from torchvision.io import write_png
from natsort import natsorted
from tqdm import tqdm
from PIL import Image

from .cholecseg8k_dataset import rgb_hex_to_class_id, inverse_lookup_dict


if __name__ == "__main__":

    source = '/local/scratch/CholecSeg8k/'
    assert os.path.isdir(source)

    mask_paths = natsorted(glob(source + f"/*/*/*endo_watershed_mask.png"))

    for mask_file in tqdm(mask_paths, desc='Converting masks'):
        mask_file_name = mask_file.split('/')[-1]
        video_id_path = mask_file.removesuffix(mask_file_name)
        frame_id = mask_file_name.split('_')[1]
        target_file_name = f"frame_{frame_id}_endo_id_mask.png"
        target_file_path = os.path.join(video_id_path, target_file_name)

        mask = np.array(Image.open(mask_file).convert("RGB"))
        id_mask = torch.from_numpy(mask)
        # Convert RGB / watershed masks to integer class masks
        id_mask = rgb_hex_to_class_id(id_mask.unsqueeze(0), inverse_lookup_dict)  # .squeeze(0)
        # Setting all ignore labels to "0" / black background
        id_mask[id_mask < 0] = 0
        id_mask = id_mask.to(torch.uint8)
        write_png(input=id_mask, filename=target_file_path)

