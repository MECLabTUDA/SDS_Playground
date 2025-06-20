import multiprocessing
import glob
import os.path

import torch
import numpy as np
from natsort import natsorted
from tqdm import tqdm
from skimage import io
from torch.nn.functional import interpolate


from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.paths import nnUNet_raw
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

from sds_playground.datasets.cataract1k.cataract1ksegm_visualisation import CLASSES


def load_and_convert_case(input_image_path: str,
                          input_mask_path: str,
                          output_image_path: str,
                          output_mask_path: str,
                          target_shape: tuple = (128, 128)):

    try:
        mask = io.imread(input_mask_path)
        mask = torch.from_numpy(mask).unsqueeze(0)
        mask = interpolate(mask.unsqueeze(0), target_shape, mode='nearest').squeeze(0).squeeze(0)
        mask = mask.numpy().astype(np.uint8)
        assert np.min(mask) >= 0
        assert np.max(mask) < 14
        io.imsave(output_mask_path, mask, check_contrast=False)

    except Exception as e:
        # Skipping failure cases...
        print(e)
        return

    img = io.imread(input_image_path)
    img = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2)
    img = interpolate(img, target_shape, mode='bilinear').squeeze(0)
    io.imsave(output_image_path, img.permute(1, 2, 0).numpy())


if __name__ == "__main__":
    source = '/local/scratch/Cataract1kSegmentation/'

    ds_name = 'Dataset888_Cataracts1kSegm'

    images_tr = join(nnUNet_raw, ds_name, 'imagesTr')
    labels_tr = join(nnUNet_raw, ds_name, 'labelsTr')
    images_ts = join(nnUNet_raw, ds_name, 'imagesTs')
    labels_ts = join(nnUNet_raw, ds_name, 'labelsTs')
    maybe_mkdir_p(images_tr)
    maybe_mkdir_p(images_ts)
    maybe_mkdir_p(labels_tr)
    maybe_mkdir_p(labels_ts)

    with multiprocessing.get_context("spawn").Pool(8) as p:

        #
        # Training data
        #

        tr_image_paths = [glob.glob(os.path.join(source,
                                                 f"Segmentation_dataset/Annotations/Images-and-Supervisely-Annotations/"
                                                 f"case_{idn}/img/*.png"))
                          for idn in [5013, 5014, 5015, 5016, 5017,
                                      5032, 5051, 5057, 5058, 5063,
                                      5072, 5104, 5108, 5299, 5300,
                                      5301, 5303, 5304, 5305, 5309,
                                      5315, 5316, 5317, 5319, 5325]]
        tr_image_paths = natsorted([i for sl in tr_image_paths for i in sl])

        tr_label_paths = [glob.glob(os.path.join(source,
                                                 f"Segmentation_dataset/Annotations/Images-and-Supervisely-Annotations/"
                                                 f"case_{idn}/mask/*.png"))
                          for idn in [5013, 5014, 5015, 5016, 5017,
                                      5032, 5051, 5057, 5058, 5063,
                                      5072, 5104, 5108, 5299, 5300,
                                      5301, 5303, 5304, 5305, 5309,
                                      5315, 5316, 5317, 5319, 5325]]
        tr_label_paths = natsorted([i for sl in tr_label_paths for i in sl])

        print(f"Converting {len(tr_image_paths)} training samples.")

        r = []

        for sample_id in range(len(tr_image_paths)):
        # for sample_id in range(5):
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

        ts_image_paths = [glob.glob(os.path.join(source,
                                                 f"Segmentation_dataset/Annotations/Images-and-Supervisely-Annotations/"
                                                 f"case_{idn}/img/*.png"))
                          for idn in [5329, 5334, 5335, 5340, 5353]]
        ts_image_paths = natsorted([i for sl in ts_image_paths for i in sl])

        ts_label_paths = [glob.glob(os.path.join(source,
                                                 f"Segmentation_dataset/Annotations/Images-and-Supervisely-Annotations/"
                                                 f"case_{idn}/mask/*.png"))
                          for idn in [5329, 5334, 5335, 5340, 5353]]
        ts_label_paths = natsorted([i for sl in ts_label_paths for i in sl])

        print(f"Converting {len(ts_image_paths)} test samples.")

        for sample_id in range(len(ts_image_paths)):
        # for sample_id in range(5):
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
                          {**{v["name"]: k for k, v in CLASSES.items()}},
                          # Inverted dict
                          len(tr_image_paths),
                          '.png',
                          dataset_name=ds_name)
