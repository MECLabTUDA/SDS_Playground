import os
import glob
import inspect
from pathlib import Path

import torch
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torchvision import transforms
from albumentations.pytorch.functional import img_to_tensor
from PIL import Image
from natsort import natsorted
from tqdm import tqdm

from .cadisv2_experiments import EXP1, EXP2, EXP3
from .cadisv2_visualisation import get_cadis_float_cmap

TT = transforms.ToTensor()

MAX_N_LABELS = 25


def remap_mask(mask: torch.Tensor, exp_dict: dict, ignore_label: int = 255) -> torch.Tensor:
    classes = []
    class_remapping = exp_dict["LABEL"]
    for key, val in class_remapping.items():
        for cls in val:
            classes.append(cls)
    assert len(classes) == len(set(classes))

    N = max(len(classes), mask.max() + 1)
    # remap_array = np.full(N, ignore_label, dtype=np.uint8)
    remap_array = torch.full([N], ignore_label, dtype=torch.int64)
    for key, val in class_remapping.items():
        for v in val:
            remap_array[v] = key
    # mask = mask.to(torch.int64)
    remap_mask = remap_array[mask]
    # remap_mask_tensor = torch.from_numpy(remap_mask)
    # return remap_mask_tensor
    return remap_mask


def mask_to_one_hot(mask: torch.Tensor, max_n_labels: int = MAX_N_LABELS) -> torch.Tensor:
    """ Obtains one-hot labels from a given mask.

    :param mask: (H, W) tensor in [0, K]
    :return: (K,) tensor in {0, 1}
    """
    labels = torch.unique(mask)
    if 255 in labels:
        labels = labels[:-1]
    one_hot = F.one_hot(labels, num_classes=max_n_labels).sum(0).float()
    return one_hot


class CaDISv2_Dataset(data.Dataset):

    original_shape = (3, 540, 960)
    display_shape = (3, 270, 480)

    def __init__(self,
                 root: Path,
                 spatial_transform=None,
                 img_normalization=None,
                 exp: int = 1,
                 mode: str = "train",
                 filter_mislabeled: bool = False,
                 compute_unique_classes_per_sample: bool = False,
                 sample_img: bool = True,
                 sample_mask: bool = True,
                 sample_phase_label: bool = False,
                 sample_sem_label: bool = True):

        """ Creates class instance.

        :param root: Path to dataset
        :param spatial_transform: Transformations applied to images AND label masks
        :param img_normalization: Normalization transformations, only applied to images
        :param exp: Experiment abstraction level
        :param mode: train/val/test
        """
        assert root.is_dir(), f"{root} not a directory"
        assert (root / 'phase_annotations/').is_dir(), \
            "Please put CATARACTS phase annotations into CaDIS root dir"
        self.sample_list = []
        self.exp_id = exp
        self.mode = mode
        self.sample_img = sample_img
        self.sample_mask = sample_mask
        self.sample_phase_label = sample_phase_label
        self.sample_sem_label = sample_sem_label
        if exp == 0:
            self.num_classes = 36
            self.ignore_index = None
        elif exp == 1:
            self.num_classes = 9  # TODO: 8?
            self.ignore_index = None
            self._EXP = EXP1
        elif exp == 2:
            self.num_classes = 18
            self.ignore_index = 255
            self._EXP = EXP2
        else:
            self.num_classes = 26
            self.ignore_index = 255
            self._EXP = EXP3
        self.normalization = img_normalization
        self.spatial_transform = spatial_transform

        self.compute_unique_classes_per_sample = compute_unique_classes_per_sample
        self.filter_mislabeled = filter_mislabeled
        self.mislabeled_samples = []
        if filter_mislabeled:
            # TODO: Can this import be simplified?
            cf = inspect.getfile(self.__class__)
            with open(os.path.join("/", *cf.split("/")[:-1], 'cadisv2_mislabeled.txt'), 'r') as f:
                for line in f.readlines():
                    self.mislabeled_samples.append(line.replace('\n', ''))

        img_file_list = []
        label_file_list = []
        annotation_list = []
        if mode == 'train':
            img_file_list = [glob.glob(str(root / f"Video{idn}/Images/*.png"))
                             for idn in ['01', '03', '04', '06', '08', '09', '10', '11', '13', '14', '15',
                                         '17', '18', '19', '20', '21', '23', '24', '25']]  # Remove '25'?
            label_file_list = [glob.glob(str(root / f"Video{idn}/Labels/*.png"))
                               for idn in ['01', '03', '04', '06', '08', '09', '10', '11', '13', '14', '15',
                                           '17', '18', '19', '20', '21', '23', '24', '25']]  # Remove '25'?
        elif mode == 'val':
            img_file_list = [glob.glob(str(root / f"Video{idn}/Images/*.png"))
                             for idn in ['05', '07', '16']]
            label_file_list = [glob.glob(str(root / f"Video{idn}/Labels/*.png"))
                               for idn in ['05', '07', '16']]
        elif mode == 'test':
            img_file_list = [glob.glob(str(root / f"Video{idn}/Images/*.png"))
                             for idn in ['02', '12', '22']]
            label_file_list = [glob.glob(str(root / f"Video{idn}/Labels/*.png"))
                               for idn in ['02', '12', '22']]
        elif mode == 'full':
            img_file_list = [glob.glob(str(root / f"Video{idn}/Images/*.png"))
                             for idn in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12',
                                         '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24',
                                         '25']]  # Remove '25'?
            label_file_list = [glob.glob(str(root / f"Video{idn}/Labels/*.png"))
                               for idn in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12',
                                           '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24',
                                           '25']]

        assert len(img_file_list) > 0
        assert len(img_file_list) == len(label_file_list)  # == len(graph_file_list)

        img_file_list = natsorted([i for sl in img_file_list for i in sl])
        label_file_list = natsorted([i for sl in label_file_list for i in sl])

        # Read CATARACTS annotations into pd_frames
        annotation_frames = {}
        for path in natsorted(glob.glob(str(root / 'phase_annotations/*.csv'))):
            vid_id = path.split('/')[-1].replace('.csv', '')
            pd_frame = pd.read_csv(path, index_col='Frame')
            annotation_frames[vid_id] = pd_frame

        # Filter list of image file paths and read CATARACTS phase annotations
        # TODO: Phase annotations are not read if filter_mislabeled=False ...
        self.unique_classes_per_sample = []
        if filter_mislabeled:
            _img_file_list = []
            _label_file_list = []
            annotation_list = []

            for img_file, label_file in tqdm(zip(img_file_list, label_file_list),
                                             total=len(img_file_list),
                                             desc='Loading CaDISv2'):
                clean = True
                for mislabeled_sample in self.mislabeled_samples:
                    if mislabeled_sample in img_file:
                        clean = False
                if clean:
                    _img_file_list.append(img_file)
                    _label_file_list.append(label_file)

                    vid_id = img_file.split('/')[-1].split('_')[0].replace('Video', '').zfill(2)
                    vid_id = 'train' + vid_id
                    frame_id = img_file.split('/')[-1].split('_')[-1].replace('frame', '').replace('.png', '')
                    frame_id = frame_id.lstrip('0')
                    phase = annotation_frames[vid_id]['Steps'].values[int(frame_id) - 1]
                    annotation_list.append(phase)

                    if self.compute_unique_classes_per_sample:
                        # Extract unique classes per mask
                        # TODO: Not the most efficient to do the load & convert twice...
                        # TODO: For bigger datasets this should be pre-computed and saved/loaded
                        mask = np.array(Image.open(label_file))
                        mask = torch.from_numpy(mask).to(torch.int64)
                        if not self.exp_id == 0:
                            mask = self.remap_mask(mask)
                        mask = mask.squeeze(0)
                        mask_classes = np.setdiff1d(np.unique(mask.numpy()), np.array([255]))
                        self.unique_classes_per_sample.append(mask_classes)

            img_file_list = _img_file_list
            label_file_list = _label_file_list

            # Initial weights for weighted_sampling
            initial_sampling_weights = [1.0 / len(img_file_list)] * len(img_file_list)
            self.sampling_weights = initial_sampling_weights

        for i in range(len(img_file_list)):
            img_name = img_file_list[i].split('/')[-1]
            tmp_dict = {'img': img_file_list[i],
                        'mask': label_file_list[i],
                        'name': img_name,
                        'phase': annotation_list[i]}
            self.sample_list.append(tmp_dict)

    def remap_mask(self, mask) -> torch.Tensor:
        return remap_mask(mask, self._EXP, ignore_label=255)

    def update_sampling_weights(self, new_weights: list):
        assert len(new_weights) == len(self.sampling_weights)
        self.sampling_weights = new_weights

    def get_cmap(self):
        return get_cadis_float_cmap()

    def get_class_names(self) -> list:
        if self.exp_id <= 1:
            return list(self._EXP['CLASS'].values())
        return list(self._EXP['CLASS'].values())[:-1]  # w/o ignore

    def __getitem__(self, index: int) -> (torch.Tensor, torch.Tensor, str):
        sample = self.sample_list[index]
        img = np.array(Image.open(sample['img'])) if self.sample_img else torch.empty([])
        mask = np.array(Image.open(sample['mask'])) if self.sample_mask else torch.empty([])
        file_name = sample['name']
        phase = sample['phase']

        if self.spatial_transform is not None and (self.sample_img or self.sample_mask):
            if not self.sample_mask:
                data = self.spatial_transform(image=img)
                img = data['image']
            elif not self.sample_img:
                data = self.spatial_transform(image=mask)
                mask = data['image']
            else:
                # To ensure the same transformation is applied to img + mask
                data = self.spatial_transform(image=img, mask=mask)
                img = data['image']
                mask = data['mask']

        if self.normalization is not None and self.sample_img:
            img = self.normalization(image=img)['image']

        if self.sample_img:
            img = img_to_tensor(img)

        if self.sample_mask:
            mask = torch.from_numpy(mask).to(torch.int64)
            if not self.exp_id == 0:
                mask = self.remap_mask(mask)
            mask = mask.squeeze(0)

        label = []
        if self.sample_phase_label:
            phase_label = torch.tensor([phase])
            label.append(phase_label)

        if self.sample_sem_label:
            # Tools / seg. masks to one-hot
            if self.exp_id == 0:
                sem_label = mask_to_one_hot(mask, max_n_labels=self.num_classes)
            else:
                sem_label = mask_to_one_hot(mask, max_n_labels=self.num_classes - 1)
            label.append(sem_label)

        if self.sample_phase_label or self.sample_sem_label:
            label = torch.cat(label, dim=-1)
        else:
            label = torch.empty([])

        if self.exp_id in [2, 3] and self.sample_sem_label:
            # Add dummy label for ignore class
            label = torch.cat([label, torch.zeros(1)])

        # return img, mask, graph, file_name, label
        return img, mask, file_name, label

    def __len__(self) -> int:
        return len(self.sample_list)

    def img_size(self) -> tuple:
        return self[0][0].shape
