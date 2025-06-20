""" (Using different conda env. for nnUNet) """

import glob
import random
import multiprocessing

import torch
import numpy as np
from torch.nn.functional import interpolate
from skimage import io
from natsort import natsorted
from tqdm import tqdm

from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.paths import nnUNet_raw
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

from sds_playground.datasets.cadisv2.cadisv2_experiments import EXP1, EXP2, EXP3


def remap_mask(mask: np.ndarray, exp_dict: dict, ignore_label: int = 0) -> np.ndarray:
    classes = []
    class_remapping = exp_dict["LABEL"]
    for key, val in class_remapping.items():
        for cls in val:
            classes.append(cls)
    assert len(classes) == len(set(classes))

    N = max(len(classes), np.max(mask) + 1)
    remap_array = np.full(N, ignore_label, dtype=np.uint8)
    for key, val in class_remapping.items():
        for v in val:
            # remap_array[v] = key
            remap_array[v] = key + 1
    mask = mask.astype(int)
    remap_mask = remap_array[mask]
    return remap_mask


def load_and_convert_case(input_image_path: str,
                          input_mask_path: str,
                          output_image_path: str,
                          output_mask_path: str,
                          target_shape: tuple = (128, 128)):

    try:
        mask = io.imread(input_mask_path)
        mask = remap_mask(mask, EXP2, ignore_label=0)
        mask = torch.from_numpy(mask).unsqueeze(0)
        mask = interpolate(mask.unsqueeze(0), target_shape, mode='nearest').squeeze(0).squeeze(0)
        mask = mask.numpy().astype(np.uint8)
        assert np.min(mask) >= 0
        assert np.max(mask) < 255
        io.imsave(output_mask_path, mask, check_contrast=False)

    except Exception as e:
        # Skipping failure cases...
        print(e)
        return

    img = io.imread(input_image_path)
    img = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2)
    img = interpolate(img, target_shape, mode='bilinear').squeeze(0)
    io.imsave(output_image_path, img.permute(1, 2, 0).numpy())

    # shutil.copy(input_image_path, output_image_path)


if __name__ == "__main__":

    source = '/local/scratch/CaDISv2/'

    # TODO: Filter mislabeled?

    # ds_name = 'Dataset666_CaDISv2EXP1'
    ds_name = 'Dataset666_CaDISv2EXP2'
    # ds_name = 'Dataset666_CaDISv2EXP3'

    images_tr = join(nnUNet_raw, ds_name, 'imagesTr')
    labels_tr = join(nnUNet_raw, ds_name, 'labelsTr')
    images_ts = join(nnUNet_raw, ds_name, 'imagesTs')
    labels_ts = join(nnUNet_raw, ds_name, 'labelsTs')
    maybe_mkdir_p(images_tr)
    maybe_mkdir_p(images_ts)
    maybe_mkdir_p(labels_tr)
    maybe_mkdir_p(labels_ts)

    with multiprocessing.get_context("spawn").Pool(8) as p:

        """
        img_file_list = natsorted(glob.glob(source + f"/Video*/Images/*.png"))
        label_file_list = natsorted(glob.glob(source + f"/Video*/Labels/*.png"))
        combined_file_list = list(zip(img_file_list, label_file_list))

        # Seed the random number generator for reproducibility
        random.seed(42)

        # Shuffle the combined list
        random.shuffle(combined_file_list)

        # Unzip the shuffled list back into images and labels
        shuffled_image_files, shuffled_label_files = zip(*combined_file_list)

        # Specify the percentages for the split
        train_percent = 0.9
        val_percent = 0.05
        test_percent = 1.0 - train_percent - val_percent

        # Calculate the split indices
        total_items = len(shuffled_image_files)
        train_end = int(train_percent * total_items)
        val_end = train_end + int(val_percent * total_items)
        
        """

        #
        # Training data
        #


        tr_image_paths = [glob.glob(source + f"/Video{idn}/Images/*.png")
                          for idn in ['01', '03', '04', '06', '08', '09', '10', '11', '13', '14', '15',
                                      '17', '18', '19', '20', '21', '23', '24'] + ['05', '07', '16']]
        tr_image_paths = natsorted([i for sl in tr_image_paths for i in sl])

        tr_label_paths = [glob.glob(source + f"/Video{idn}/Labels/*.png")
                          for idn in ['01', '03', '04', '06', '08', '09', '10', '11', '13', '14', '15',
                                      '17', '18', '19', '20', '21', '23', '24'] + ['05', '07', '16']]
        tr_label_paths = natsorted([i for sl in tr_label_paths for i in sl])


        # tr_image_paths, tr_label_paths = shuffled_image_files[:val_end], shuffled_label_files[:val_end]


        print(f"Converting {len(tr_image_paths)} training samples.")

        r = []

        for sample_id in range(len(tr_image_paths)):
            r.append(
                p.starmap_async(
                    load_and_convert_case,
                    ((
                         tr_image_paths[sample_id],
                         tr_label_paths[sample_id],
                         join(images_tr, tr_image_paths[sample_id].split('/')[-1][:-4] + '_0000.png'),
                         join(labels_tr, tr_label_paths[sample_id].split('/')[-1])
                     ),)
                )

            )

        #
        # Testing data
        #


        ts_image_paths = [glob.glob(source + f"/Video{idn}/Images/*.png")
                          for idn in ['02', '12', '22']]
        ts_image_paths = natsorted([i for sl in ts_image_paths for i in sl])

        ts_label_paths = [glob.glob(source + f"/Video{idn}/Labels/*.png")
                          for idn in ['02', '12', '22']]
        ts_label_paths = natsorted([i for sl in ts_label_paths for i in sl])

        # ts_image_paths, ts_label_paths = shuffled_image_files[val_end:], shuffled_label_files[val_end:]

        print(f"Converting {len(ts_image_paths)} test samples.")

        for sample_id in range(len(ts_image_paths)):
            r.append(
                p.starmap_async(
                    load_and_convert_case,
                    ((
                         ts_image_paths[sample_id],
                         ts_label_paths[sample_id],
                         join(images_ts, ts_image_paths[sample_id].split('/')[-1][:-4] + '_0000.png'),
                         join(labels_ts, ts_label_paths[sample_id].split('/')[-1])
                     ),)
                )

            )

        _ = [i.get() for i in tqdm(r)]

    generate_dataset_json(join(nnUNet_raw, ds_name),
                          {0: 'R', 1: 'G', 2: 'B'},
                          # {'background': 0, 'road': 1},
                          {**{'background': 0}, **{v: k + 1 for k, v in EXP2['CLASS'].items() if k < 255}},
                          # Inverted dict
                          len(tr_image_paths),
                          '.png',
                          dataset_name=ds_name)
