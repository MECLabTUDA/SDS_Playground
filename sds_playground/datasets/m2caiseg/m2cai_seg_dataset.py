import os
import glob

import torch
import torch.utils.data as data
import numpy as np
from albumentations.pytorch.functional import img_to_tensor
from natsort import natsorted
from PIL import Image


class M2CaiSegDataset(data.Dataset):

    original_shape = (3, 434, 774)
    display_shape = (3, 270, 480)

    def __init__(self,
                 root: str,
                 spatial_transform=None,
                 img_normalization=None,
                 mode: str = 'train',
                 sample_img: bool = True,
                 sample_mask: bool = True):

        """ Creates class instance.

        :param root: Path to dataset
        :param spatial_transform: Transformations applied to images AND segmentation masks
        :param img_normalization: Normalization transformations, only applied to images
        """

        assert os.path.isdir(root)

        self.sample_list = []
        self.mode = mode
        self.normalization = img_normalization
        self.spatial_transform = spatial_transform
        self.sample_img = sample_img
        self.sample_mask = sample_mask

        img_file_list = []
        mask_file_list = []

        if mode == 'train':
            img_file_list = [glob.glob(root + f"/train/images/*.jpg")]
            mask_file_list = [glob.glob(root + f"/train/groundtruth/*.png")]
        elif mode == 'val':
            img_file_list = [glob.glob(root + f"/trainval/images/*.jpg")]
            mask_file_list = [glob.glob(root + f"/trainval/groundtruth/*.png")]
        elif mode == 'test':
            img_file_list = [glob.glob(root + f"/test/images/*.jpg")]
            mask_file_list = [glob.glob(root + f"/test/groundtruth/*.png")]

        assert len(img_file_list) > 0
        img_file_list = natsorted([i for sl in img_file_list for i in sl])
        mask_file_list = natsorted([i for sl in mask_file_list for i in sl])

        for i in range(len(img_file_list)):
            img_name = mode + "/" + img_file_list[i].split('/')[-1]
            tmp_dict = {'img': img_file_list[i],
                        'mask': mask_file_list[i],
                        'name': img_name}
            self.sample_list.append(tmp_dict)

    def __len__(self) -> int:
        return len(self.sample_list)

    def img_size(self) -> tuple:
        return self[0][0].shape

    def __getitem__(self, index: int) -> (torch.Tensor, torch.Tensor, str):
        sample = self.sample_list[index]
        img = np.array(Image.open(sample['img'])) if self.sample_img else torch.empty([])
        mask = np.array(Image.open(sample['mask'])) if self.sample_mask else torch.empty([])
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
            mask = torch.from_numpy(mask)

        return img, mask, file_name
