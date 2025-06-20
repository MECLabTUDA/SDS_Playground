""" (Using different conda env. for nnUNet) """

import glob
import shutil
import random
import multiprocessing

import torch
import numpy as np
import SimpleITK as sitk
import nibabel as nib
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
            remap_array[v] = key
    mask = mask.astype(int)
    remap_mask = remap_array[mask]

    binary_tool_mask = np.where(remap_mask == 7, 1.0, 0.0)

    return binary_tool_mask


def load_and_convert_case(input_image_path: str,
                          input_mask_path: str,
                          output_image_path: str,
                          output_mask_path: str,
                          target_shape: tuple = (256, 256)):

    try:
        mask = io.imread(input_mask_path)
        mask = remap_mask(mask, EXP1, ignore_label=0)
        mask = torch.from_numpy(mask).unsqueeze(0)
        mask = interpolate(mask.unsqueeze(0), target_shape, mode='nearest').squeeze(0).squeeze(0)  # H, W
        mask = mask.numpy().astype(np.uint8)
        # mask = np.rot90(mask, k=1, axes=(0, 1))
        #assert np.min(mask) >= 0
        assert np.min(mask) == 0
        assert np.max(mask) == 1
        #assert np.max(mask) < 255
        # io.imsave(output_mask_path, mask, check_contrast=False)
        #mask = sitk.GetImageFromArray(mask)
        #sitk.WriteImage(mask, output_mask_path)
        #nifti_mask = nib.Nifti1Image(mask[..., np.newaxis, np.newaxis], affine=np.eye(4))  # H, W, Z(1=), 1
        #nib.save(nifti_mask, output_mask_path)
        sitk_mask = sitk.GetImageFromArray(mask[..., np.newaxis, np.newaxis])
        sitk.WriteImage(sitk_mask, output_mask_path)

    except Exception as e:
        # Skipping failure cases...
        print(e)
        return

    img = io.imread(input_image_path)
    img = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2)  # 1, C, H, W
    img = interpolate(img, target_shape, mode='bilinear')
    img = img.permute(2, 3, 0, 1).numpy()  # H, W, Z(=1), 3

    sitk_img = sitk.GetImageFromArray(img)
    # sitk_img = sitk.Rotate(sitk_img, angle=-90, interpolationMethod=sitk.sitkBilinear)
    sitk.WriteImage(sitk_img, output_image_path, True)

    #img = np.rot90(img, k=1, axes=(0, 1))
    # nifti_img = nib.Nifti1Image(img, affine=np.eye(4))
    # nib.save(nifti_img, output_image_path)
    # img = sitk.GetImageFromArray(img)
    # sitk.WriteImage(img, output_image_path)
    #io.imsave(output_image_path, img.permute(1, 2, 0).numpy())

    # shutil.copy(input_image_path, output_image_path)


if __name__ == "__main__":

    source = '/local/scratch/CaDISv2/'

    # TODO: Filter mislabeled?

    # ds_name = 'Dataset666_CaDISv2EXP1'
    ds_name = 'Dataset666_CaDISv2EXP1_sam_whitepaper'
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
        train_percent = 0.6
        val_percent = 0.2
        test_percent = 1.0 - train_percent - val_percent

        # Calculate the split indices
        total_items = len(shuffled_image_files)
        train_end = int(train_percent * total_items)
        val_end = train_end + int(val_percent * total_items)

        #
        # Training data
        #

        """
        tr_image_paths = [glob.glob(source + f"/Video{idn}/Images/*.png")
                          for idn in ['01', '03', '04', '06', '08', '09', '10', '11', '13', '14', '15',
                                      '17', '18', '19', '20', '21', '23', '24'] + ['05', '07', '16']]
        tr_image_paths = natsorted([i for sl in tr_image_paths for i in sl])
        tr_label_paths = [glob.glob(source + f"/Video{idn}/Labels/*.png")
                          for idn in ['01', '03', '04', '06', '08', '09', '10', '11', '13', '14', '15',
                                      '17', '18', '19', '20', '21', '23', '24'] + ['05', '07', '16']]
        tr_label_paths = natsorted([i for sl in tr_label_paths for i in sl])
        """

        tr_image_paths, tr_label_paths = shuffled_image_files[:val_end], shuffled_label_files[:val_end]

        print(f"Converting {len(tr_image_paths)} training samples.")

        r = []

        for sample_id in range(len(tr_image_paths)):
            r.append(
                p.starmap_async(
                    load_and_convert_case,
                    ((
                         tr_image_paths[sample_id],
                         tr_label_paths[sample_id],
                         join(images_tr, tr_image_paths[sample_id].split('/')[-1][:-4] + '.nii.gz'),
                         join(labels_tr, tr_label_paths[sample_id].split('/')[-1][:-4] + '.nii.gz')
                     ),)
                )

            )

        #
        # Testing data
        #

        """
        ts_image_paths = [glob.glob(source + f"/Video{idn}/Images/*.png")
                          for idn in ['02', '12', '22']]
        ts_image_paths = natsorted([i for sl in ts_image_paths for i in sl])
        ts_label_paths = [glob.glob(source + f"/Video{idn}/Labels/*.png")
                          for idn in ['02', '12', '22']]
        ts_label_paths = natsorted([i for sl in ts_label_paths for i in sl])
        """

        ts_image_paths, ts_label_paths = shuffled_image_files[val_end:], shuffled_label_files[val_end:]

        print(f"Converting {len(ts_image_paths)} test samples.")

        for sample_id in range(len(ts_image_paths)):
            r.append(
                p.starmap_async(
                    load_and_convert_case,
                    ((
                         ts_image_paths[sample_id],
                         ts_label_paths[sample_id],
                         join(images_ts, ts_image_paths[sample_id].split('/')[-1][:-4] + '.nii.gz'),
                         join(labels_ts, ts_label_paths[sample_id].split('/')[-1][:-4] + '.nii.gz')
                     ),)
                )

            )

        _ = [i.get() for i in tqdm(r)]

    generate_dataset_json(join(nnUNet_raw, ds_name),
                          {0: 'R', 1: 'G', 2: 'B'},
                          {'background': 0, 'tool': 1},
                          len(tr_image_paths),
                          '.nii.gz',
                          dataset_name=ds_name)
