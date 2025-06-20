""" (Using different conda env. for nnUNet) """

import glob
import shutil
import random
import multiprocessing

import numpy as np
import torch
from torch.nn.functional import interpolate
from skimage import io
from natsort import natsorted
from tqdm import tqdm

from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.paths import nnUNet_raw
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

from sds_playground.datasets.cholecseg8k.cholecseg8k_dataset import CLASSES


def load_and_convert_case(input_image_path: str,
                          input_mask_path: str,
                          output_image_path: str,
                          output_mask_path: str,
                          target_shape: tuple = (128, 128)):
    try:
        mask = io.imread(input_mask_path)
        mask = torch.from_numpy(mask).unsqueeze(0)
        # Reshape to (256, 256)
        mask = interpolate(mask.unsqueeze(0), target_shape, mode='nearest').squeeze(0).squeeze(0)
        # Setting all ignore labels to "0" / black background
        mask[mask == -1] = 0
        mask[mask == -2] = 0
        mask[mask == -3] = 0
        mask = mask.numpy().astype(np.uint8)
        assert np.min(mask) >= 0
        assert np.max(mask) < 255
        io.imsave(output_mask_path, mask, check_contrast=False)

    except Exception as e:
        # Skipping failure cases...
        print(f"Skipped {input_image_path} due to exception: \n {e}")
        return

    img = io.imread(input_image_path)
    img = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2)
    img = interpolate(img, target_shape, mode='bilinear').squeeze(0)
    io.imsave(output_image_path, img.permute(1, 2, 0).numpy())


def get_out_file_id(path: str) -> str:
    """ E.g. /local/scratch/CholecSeg8k/video01/video01_00080/frame_80_endo.png """
    video_id = path.split('/')[-3]
    frame_id = path.split('/')[-1].split('_')[0:2]
    frame_id = frame_id[0] + frame_id[1]
    return video_id + "_" + frame_id


if __name__ == "__main__":

    source = '/local/scratch/CholecSeg8k/'

    ds_name = 'Dataset777_CholecSeg8k'

    images_tr = join(nnUNet_raw, ds_name, 'imagesTr')
    labels_tr = join(nnUNet_raw, ds_name, 'labelsTr')
    images_ts = join(nnUNet_raw, ds_name, 'imagesTs')
    labels_ts = join(nnUNet_raw, ds_name, 'labelsTs')
    maybe_mkdir_p(images_tr)
    maybe_mkdir_p(images_ts)
    maybe_mkdir_p(labels_tr)
    maybe_mkdir_p(labels_ts)

    with multiprocessing.get_context("spawn").Pool(8) as p:

        img_file_list = natsorted(glob.glob(source + f"video*/**/*endo.png"))
        label_file_list = natsorted(glob.glob(source + f"video*/**/*endo_id_mask.png"))

        combined_file_list = list(zip(img_file_list, label_file_list))

        # Seed the random number generator for reproducibility
        random.seed(42)

        # Shuffle the combined list
        random.shuffle(combined_file_list)

        # Unzip the shuffled list back into images and labels
        shuffled_image_files, shuffled_label_files = zip(*combined_file_list)

        # Specify the percentages for the split
        train_percent = 0.7
        val_percent = 0.15
        test_percent = 1.0 - train_percent - val_percent

        # Calculate the split indices
        total_items = len(shuffled_image_files)
        train_end = int(train_percent * total_items)
        val_end = train_end + int(val_percent * total_items)

        #
        # Training data
        #

        tr_image_paths, tr_label_paths = shuffled_image_files[:val_end], shuffled_label_files[:val_end]

        """
        tr_image_paths = [glob.glob(source + f"/video{idn}/**/*endo.png")
                          for idn in
                          ['01', '09', '12', '17', '18', '20', '24', '25', '26', '27', '28', '35', '43', '52']]
        tr_image_paths = natsorted([path for paths in tr_image_paths for path in paths])
        tr_label_paths = [glob.glob(source + f"/video{idn}/**/*endo_watershed_mask.png")
                          for idn in
                          ['01', '09', '12', '17', '18', '20', '24', '25', '26', '27', '28', '35', '43', '52']]
        tr_label_paths = natsorted([path for _paths in tr_label_paths for path in _paths])
        """

        print(f"Converting {len(tr_image_paths)} training samples.")

        r = []

        for sample_id in range(len(tr_image_paths)):
            r.append(
                p.starmap_async(
                    load_and_convert_case,
                    ((
                         tr_image_paths[sample_id],
                         tr_label_paths[sample_id],
                         # Output paths
                         join(images_tr, get_out_file_id(tr_image_paths[sample_id]) + '_0000.png'),
                         join(labels_tr, get_out_file_id(tr_label_paths[sample_id]) + '.png')
                     ),)
                )

            )

        #
        # Testing data
        #

        """
        ts_image_paths = [glob.glob(source + f"/video{idn}/**/*endo.png")
                          for idn in ['37', '48', '55']]
        ts_image_paths = natsorted([path for _paths in ts_image_paths for path in _paths])
        ts_label_paths = [glob.glob(source + f"/video{idn}/**/*endo_watershed_mask.png")
                          for idn in ['37', '48', '55']]
        ts_label_paths = natsorted([path for _paths in ts_label_paths for path in _paths])
        """

        ts_image_paths, ts_label_paths = shuffled_image_files[val_end:], shuffled_label_files[val_end:]

        print(f"Converting {len(ts_image_paths)} testing samples.")

        for sample_id in range(len(ts_image_paths)):
            r.append(
                p.starmap_async(
                    load_and_convert_case,
                    ((
                         ts_image_paths[sample_id],
                         ts_label_paths[sample_id],
                         # Output paths
                         join(images_ts, get_out_file_id(ts_image_paths[sample_id]) + '_0000.png'),
                         join(labels_ts, get_out_file_id(ts_label_paths[sample_id]) + '.png')
                     ),)
                )

            )

        _ = [i.get() for i in tqdm(r)]

    generate_dataset_json(join(nnUNet_raw, ds_name),
                          {0: 'R', 1: 'G', 2: 'B'},
                          # {'background': 0, 'road': 1},
                          {**{'background': 0}, **{v['name']: k for k, v in CLASSES.items() if 255 > k > 0}},
                          # Inverted dict
                          len(tr_image_paths),
                          '.png',
                          dataset_name=ds_name)
