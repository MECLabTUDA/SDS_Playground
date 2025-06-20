from pathlib import Path

import matplotlib.pyplot as plt
import torch.nn.functional as F
import albumentations as A
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from sds_playground.datasets.cadisv2.cadisv2_dataset import CaDISv2_Dataset
from sds_playground.utils.utils import denormalize, convert_mask_to_RGB, convert_to_integer_mask, convert_to_binary_mask

ds = CaDISv2_Dataset(
    root=Path('/local/scratch/CaDISv2/'),
    spatial_transform=A.Compose([
        # A.RandomCrop(height=540, width=540),
        A.Resize(512, 512),
    ]),
    img_normalization=A.Normalize(0.5, 0.5),
    exp=2,
    mode='train',
    filter_mislabeled=True,
    sample_img=True,
    sample_mask=True,
    sample_sem_label=True,
    sample_phase_label=False
)

dl = DataLoader(ds, batch_size=8, num_workers=1, shuffle=True)

sample, label, _, _ = next(iter(dl))
sample = denormalize(sample, .5, .5)
sample = F.interpolate(sample, ds.display_shape[1:], mode='bilinear')
label = F.interpolate(label.unsqueeze(1).float(), ds.display_shape[1:], mode='nearest').squeeze(1).long()
rgb_label = convert_mask_to_RGB(mask=label,
                                palette=ds.get_cmap(),
                                ignore_index=ds.ignore_index)

N = sample.shape[0]
fig, ax = plt.subplots(2, N, figsize=(N*3, 6))
for n in range(N):
    ax[0, n].imshow(sample[n].permute(1, 2, 0))
    ax[0, n].axis('off')
    ax[1, n].imshow(rgb_label[n].permute(1, 2, 0))
    ax[1, n].axis('off')

    #save_image(tensor=sample[n], fp=f'CaDISv2_example_{n}_img.png')
    #save_image(tensor=rgb_label[n], fp=f'CaDISv2_example_{n}_mask.png')

plt.tight_layout()
plt.show()

binary_label = convert_to_binary_mask(mask=label,
                                      num_classes=ds.num_classes,
                                      ignore_index=ds.ignore_index,
                                      keep_ignore_index=False)

print(f"{binary_label.shape=}")
