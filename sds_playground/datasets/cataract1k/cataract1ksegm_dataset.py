from pathlib import Path
from glob import glob
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from albumentations.pytorch.functional import img_to_tensor
from natsort import natsorted

from sds_playground.datasets.cataract1k.cataract1ksegm_visualisation import get_cataract1k_float_cmap


class Cataract1kSegmentationDataset(Dataset):

    original_shape = (3, 768, 1024)
    display_shape = (3, 270, 480)
    ignore_index = 0
    num_classes = 14

    def __init__(self,
                 root: Path,
                 spatial_transform=None,
                 img_normalization=None,
                 mode: str = "train",
                 sample_img: bool = True,
                 sample_mask: bool = True,
                 sample_sem_label: bool = False):

        assert root.exists(), f"{root} not a directory"
        assert (root / "Segmentation_dataset/Annotations/Images-and-Supervisely-Annotations").exists(), \
            f"{root} does not contain all directories"

        self.sample_list = []
        self.mode = mode
        self.sample_img = sample_img
        self.sample_mask = sample_mask
        self.sample_sem_label = sample_sem_label
        self.spatial_transform = spatial_transform
        self.normalization = img_normalization

        case_list = []
        img_file_list = []
        mask_file_list = []

        for case_path in (root / "Segmentation_dataset/Annotations/Images-and-Supervisely-Annotations").iterdir():

            if not (case_path.is_dir() and case_path.name.startswith("case_")):
                continue

            case_list.append(case_path.name)

            if self.mode == "train" and int(case_path.name[-4:]) in [5013, 5014, 5015, 5016, 5017,
                                                                     5032, 5051, 5057, 5058, 5063,
                                                                     5072, 5104, 5108, 5299, 5300,
                                                                     5301, 5303, 5304, 5305, 5309]:
                img_file_list.extend(natsorted(glob(str(case_path / "img" / "*.png"))))
                mask_file_list.extend(natsorted(glob(str(case_path / "mask" / "*.png"))))
            elif self.mode == "val" and int(case_path.name[-4:]) in [5315, 5316, 5317, 5319, 5325]:
                img_file_list.extend(natsorted(glob(str(case_path / "img" / "*.png"))))
                mask_file_list.extend(natsorted(glob(str(case_path / "mask" / "*.png"))))
            elif self.mode == "test" and int(case_path.name[-4:]) in [5329, 5334, 5335, 5340, 5353]:
                img_file_list.extend(natsorted(glob(str(case_path / "img" / "*.png"))))
                mask_file_list.extend(natsorted(glob(str(case_path / "mask" / "*.png"))))
            elif self.mode == "full":
                img_file_list.extend(natsorted(glob(str(case_path / "img" / "*.png"))))
                mask_file_list.extend(natsorted(glob(str(case_path / "mask" / "*.png"))))
            else:
                continue

        assert len(img_file_list) > 0
        assert len(img_file_list) == len(mask_file_list)

        for i in range(len(img_file_list)):
            img_name = img_file_list[i].split('/')[-1]
            tmp_dict = {'img': img_file_list[i],
                        'mask': mask_file_list[i],
                        'name': img_name}
            self.sample_list.append(tmp_dict)

        assert len(self.sample_list) > 0

    def get_cmap(self):
        return get_cataract1k_float_cmap()

    def __len__(self):
        return len(self.sample_list)

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
            mask = torch.from_numpy(mask).to(torch.int64)

        if self.sample_sem_label:
            raise NotImplementedError

        return img, mask, file_name, torch.empty([])


if __name__ == "__main__":

    train_ds = Cataract1kSegmentationDataset(root=Path("/home/yfrisch_locale/DATA/Cataract-1k/"), mode="train")
    val_ds = Cataract1kSegmentationDataset(root=Path("/home/yfrisch_locale/DATA/Cataract-1k/"), mode="val")
    test_ds = Cataract1kSegmentationDataset(root=Path("/home/yfrisch_locale/DATA/Cataract-1k/"), mode="test")
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    img, mask, file_name, _ = train_ds[np.random.randint(0, len(train_ds))]
    print(img.shape, mask.shape, file_name)
    print(torch.unique(mask))
