import os
import glob
import random
from pathlib import Path

import torch
import torch.utils.data as data
import numpy as np
from albumentations.pytorch.functional import img_to_tensor
from natsort import natsorted
from PIL import Image
from tqdm import tqdm

from .cholecseg8k_visualisation import get_cholecseg8k_float_cmap, CLASSES


inverse_lookup_dict = {tuple(v["rgb-hex"]): k for k, v in CLASSES.items()}


def rgb_hex_to_class_id(rgb_hex_mask: torch.Tensor, inverse_dict: dict) -> torch.Tensor:
    """
    Converts RGB hex mask tensors (watershed) to class ID tensors.
    Assumes rgb_hex_mask is in (N, H, W, 3)
    """
    # Flatten the tensor and convert it to a list of tuples
    flat_tensor_list = [tuple(x) for x in rgb_hex_mask.reshape(-1, 3).tolist()]

    # Convert each RGB tuple to its corresponding class ID
    label_list = [inverse_dict[color] for color in flat_tensor_list]

    # Convert the list back to tensor shape
    label_tensor = torch.tensor(label_list, dtype=torch.long).reshape_as(rgb_hex_mask[:, :, :, 0])

    return label_tensor


def class_id_to_rgb(class_id_mask: torch.Tensor, label_dict: dict) -> torch.Tensor:
    """
        Converts class ID tensors to RGB tensors.
        Assumes tensor is of shape (N, H, W)
    """
    # Get the shape details
    N, H, W = class_id_mask.shape

    # Convert tensor to a list
    class_id_list = class_id_mask.reshape(-1).tolist()

    # Convert each class ID to its RGB tuple
    rgb_list = [label_dict[class_id]["color"] for class_id in class_id_list]

    # Convert the list back to tensor shape
    rgb_tensor = torch.tensor(rgb_list, dtype=torch.uint8).reshape(N, H, W, 3)

    return rgb_tensor


def id_mask_to_one_hot(mask, num_labels):
    # Create a zero-filled vector where the length is the number of labels
    one_hot = torch.zeros(num_labels)

    # Find the unique labels present in the mask
    unique_labels = torch.unique(mask)

    # Use the unique labels as indices to set the corresponding positions in the one-hot vector to 1
    one_hot[unique_labels.tolist()] = 1

    return one_hot


class CholecSeg8kDataset(data.Dataset):

    original_shape = (3, 480, 854)
    display_shape = (3, 270, 480)
    num_classes = 13
    ignore_index = None

    def __init__(self,
                 root: Path,
                 spatial_transform=None,
                 img_normalization=None,
                 mode: str = 'train',
                 compute_unique_classes_per_sample: bool = False,
                 sample_img: bool = True,
                 sample_mask: bool = True):

        """ Creates class instance.

        :param root: Path to dataset
        :param spatial_transform: Transformations applied to images AND segmentation masks
        :param img_normalization: Normalization transformations, only applied to images
        """

        assert root.is_dir()

        self.sample_list = []
        self.mode = mode
        self.normalization = img_normalization
        self.spatial_transform = spatial_transform
        self.sample_img = sample_img
        self.sample_mask = sample_mask

        self.compute_unique_classes_per_sample = compute_unique_classes_per_sample

        img_file_list = natsorted(glob.glob(str(root / f"video*/**/*endo.png")))
        label_file_list = natsorted(glob.glob(str(root / f"video*/**/*endo_id_mask.png")))
        combined_file_list = list(zip(img_file_list, label_file_list))

        # Seed the random number generator for reproducibility
        random.seed(42)

        # Shuffle the combined list
        random.shuffle(combined_file_list)

        # Reset random seed
        random.seed(None)

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

        if mode == 'train':
            img_file_list, label_file_list = shuffled_image_files[:train_end], shuffled_label_files[:train_end]
        elif mode == 'val':
            img_file_list, label_file_list = (shuffled_image_files[train_end:val_end],
                                              shuffled_label_files[train_end:val_end])
        elif mode == 'test':
            img_file_list, label_file_list = shuffled_image_files[val_end:], shuffled_label_files[val_end:]
        elif mode == 'full':
            img_file_list, label_file_list = shuffled_image_files, shuffled_label_files
        else:
            raise ValueError(f"Unknown mode '{mode}'")

        assert len(img_file_list) > 0
        assert len(img_file_list) == len(label_file_list)

        if self.compute_unique_classes_per_sample:
            # Sampling weights / list of unique class ids per sample
            self.unique_classes_per_sample = []
            for img_file, label_file in tqdm(zip(img_file_list, label_file_list),
                                             total=len(img_file_list),
                                             desc='Loading CholecSeg8k classes'):
                mask = np.array(Image.open(label_file))
                mask = torch.from_numpy(mask).long()
                mask_classes = np.setdiff1d(np.unique(mask.numpy()), np.array([-3, -2, -1]))
                self.unique_classes_per_sample.append(mask_classes)

        # Initial weights for weighted_sampling
        initial_sampling_weights = [1.0 / len(img_file_list)] * len(img_file_list)
        self.sampling_weights = initial_sampling_weights

        for i in range(len(img_file_list)):
            img_name = os.path.join(*(img_file_list[i].split('/')[-2:])).replace('/', '_')
            tmp_dict = {'img': img_file_list[i],
                        'mask': label_file_list[i],
                        'name': img_name}
            self.sample_list.append(tmp_dict)

    def __len__(self) -> int:
        return len(self.sample_list)

    def img_size(self) -> tuple:
        return self[0][0].shape

    def get_cmap(self):
        return get_cholecseg8k_float_cmap()

    def get_class_names(self):
        return [class_dict["name"] for class_dict in CLASSES.values()][3:]

    def __getitem__(self, index: int) -> (torch.Tensor, torch.Tensor, str):
        sample = self.sample_list[index]
        img = np.array(Image.open(sample['img']).convert("RGB")) if self.sample_img else torch.empty([])
        mask = np.array(Image.open(sample['mask'])).squeeze() if self.sample_mask else torch.empty([])
        file_name = sample['name']

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
            label = id_mask_to_one_hot(mask, self.num_classes)
        else:
            label = torch.tensor([])

        return img, mask, file_name, label
