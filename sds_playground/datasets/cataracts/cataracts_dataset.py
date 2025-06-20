import os
import glob

import torch
from torch.utils.data import Dataset
import albumentations as A
import torchvision.transforms.functional as TF
from natsort import natsorted
from PIL import Image
import jpeg4py as jpeg
import pandas as pd
import numpy as np
from tqdm import tqdm

from .cataracts_visualisation import TOOL_LABEL_NAMES, PHASE_LABEL_NAMES


class CATARACTSDataset(Dataset):
    original_shape = (3, 1080, 1920)
    display_shape = (3, 270, 480)

    tool_label_names = TOOL_LABEL_NAMES
    num_tool_classes = len(tool_label_names)

    phase_label_names = PHASE_LABEL_NAMES
    num_phases_classes = len(phase_label_names)

    def __init__(self,
                 root: str,
                 resize_shape: tuple,
                 crop_shape: tuple = None,
                 normalize: tuple = None,
                 random_hflip: bool = False,
                 random_brightness_contrast: bool = False,
                 mode: str = "train",
                 sample_img: bool = True,
                 sample_tool_labels: bool = True,
                 sample_phase_labels: bool = True,
                 frame_step: int = 1,
                 n_seq_frames: int = 1,
                 overlapping_seq_chunks: bool = False,
                 remove_idle: bool = False,
                 train_data_only: bool = True
                 ):

        """ Creates class instance.


        """

        super(CATARACTSDataset, self).__init__()

        assert os.path.isdir(root)
        assert os.path.isdir(os.path.join(root,'phase_annotations/')), \
            "Please put CATARACTS phase annotations into root dir"
        assert os.path.isdir(os.path.join(root, 'tool_annotations/')), \
            "Please put CATARACTS tool annotations into root dir"
        self.root = root
        self.sample_list = []
        self.mode = mode
        self.sample_img = sample_img
        self.sample_phase_labels = sample_phase_labels
        self.sample_tool_labels = sample_tool_labels
        assert len(resize_shape) == 2
        self.resize_shape = resize_shape
        self.crop_shape = crop_shape
        self.normalize = normalize
        self.random_hflip = random_hflip
        self.random_brightness_contrast = random_brightness_contrast
        self.n_seq_frames = n_seq_frames
        self.remove_idle = remove_idle

        self.tool_counts = [0] * self.num_tool_classes
        self.tool_counts = np.array(self.tool_counts, dtype=float)

        self.phase_counts = [0] * self.num_phases_classes
        self.phase_counts = np.array(self.phase_counts, dtype=float)

        self.num_classes = 0
        if self.sample_phase_labels:
            self.num_classes += self.num_phases_classes
        if self.sample_tool_labels:
            self.num_classes += self.num_tool_classes

        # TODO: Can we split alternatively randomly based on percentage of video clips instead of whole videos?
        if mode == 'train':
            if train_data_only:
                img_file_list = [glob.glob(os.path.join(root, f"train{idn}/*.jpg"))
                                 for idn in ['01', '03', '05', '06', '07', '08', '09', '10', '11', '13',
                                             '14', '15', '16', '17', '18', '19', '22', '23', '24', '25']]
                img_file_list = natsorted([i for sl in img_file_list for i in sl])
            else:
                img_file_list = natsorted(glob.glob(os.path.join(root, f"train*/*.jpg")))
        elif mode == 'val':
            if train_data_only:
                img_file_list = [glob.glob(os.path.join(root, f"train{idn}/*.jpg"))
                                 for idn in ['04', '12', '21']]
            else:
                img_file_list = [glob.glob(os.path.join(root, f"test{idn}/*.jpg"))
                                 for idn in ['01', '07', '14', '16', '19']]
            img_file_list = natsorted([i for sl in img_file_list for i in sl])
        elif mode == 'test':
            if train_data_only:
                img_file_list = [glob.glob(os.path.join(root, f"test{idn}/*.jpg"))
                                 for idn in ['02', '03', '04', '05', '06', '08', '09', '10', '11', '12', '13',
                                             '15', '17', '18', '20', '21', '22', '23', '24', '25']]
            else:
                img_file_list = [glob.glob(os.path.join(root, f"train{idn}/*.jpg"))
                                 for idn in ['02', '20']]
            img_file_list = natsorted([i for sl in img_file_list for i in sl])
        else:
            raise ValueError("'mode' should be one of ['train', 'val', 'test']")

        # Read annotations into pandas dataframes
        self.phase_annotation_frames = {}
        self.tool_annotation_frames = {}
        for path in natsorted(glob.glob(os.path.join(root, 'phase_annotations/*.csv'))):
            vid_id = path.split('/')[-1].replace('.csv', '')
            pd_frame = pd.read_csv(path, index_col='Frame')
            self.phase_annotation_frames[vid_id] = pd_frame
        for path in natsorted(glob.glob(os.path.join(root, 'tool_annotations/*.csv'))):
            vid_id = path.split('/')[-1].replace('.csv', '')
            pd_frame = pd.read_csv(path, index_col='Frame')
            to_be_evaluated_tool_df = pd_frame[pd_frame['to_be_evaluated'] == 1]
            self.tool_annotation_frames[vid_id] = to_be_evaluated_tool_df

        if overlapping_seq_chunks:
            chunk_dist = 1
        else:
            chunk_dist = n_seq_frames
        for img_file_chunk in [img_file_list[::frame_step][i:i + n_seq_frames]
                               for i in range(0, len(img_file_list) // frame_step - (n_seq_frames - 1), chunk_dist)]:

            img_files = []
            names = []
            phase_labels = []

            for img_file in img_file_chunk:
                img_files.append(img_file)

                vid_id = img_file.split('/')[-2]
                frame_nr = img_file.split('/')[-1].removesuffix('.jpg').removeprefix('frame')

                # Integer phase labels
                phase_label = self.phase_annotation_frames[vid_id]['Steps'].values[int(frame_nr) - 1]
                phase_labels.append(phase_label)
                self.phase_counts[phase_label] += 1.0

                # One-hot tool labels
                tool_label = self.tool_annotation_frames[vid_id].values[int(frame_nr) - 1][1:]
                self.tool_counts += tool_label

                name = img_file.split('/')[-2] + "_" + img_file.split('/')[-1].removesuffix('.jpg')
                names.append(name)

            # Skipping idle samples if needed
            if 0 in phase_labels and remove_idle:
                continue

            tmp_dict = {
                'img': img_files,
                'name': names,
                'phase': phase_labels,
                # 'tools': tool_labels
            }
            self.sample_list.append(tmp_dict)

        self.dataset = self

    def get_phase_sample_weights(self) -> (np.array, np.array, np.array):
        """ Returns sample weights based on their phase label, the
            distribution of phase labels over all samples
            and the corresponding weight per class.
        """

        n_samples = sum(self.phase_counts)
        dist = self.phase_counts / n_samples
        weight_per_class = n_samples / self.phase_counts

        if self.remove_idle:
            weight_per_class[0] = 0.
        weights = [0] * len(self.sample_list)
        print("Getting phase-based sample weights ... ")
        for i, sample in enumerate(tqdm(self.sample_list)):
            weights[i] = weight_per_class[sample['phase']]
        weights = np.array(weights)

        return weights, dist, weight_per_class

    def get_tool_sample_weights(self) -> (np.array, np.array, np.array):
        """ Returns sample weights based on their tool label, the
            distribution of tool labels over all samples
            and the corresponding weight per class.
        """

        NO_TOOL_WEIGHT = 100 / 30

        n_samples = sum(self.tool_counts)
        dist = self.tool_counts / n_samples
        weight_per_class = n_samples / self.tool_counts
        if not os.path.isfile(os.path.join(self.root, 'tool_sample_weights.npy')):
            weights = [0] * len(self.sample_list)
            print("Getting tool-based sample weights ... ")
            for i, sample in enumerate(tqdm(self.sample_list)):
                img_file = sample['img'][-1]
                vid_id = img_file.split('/')[-2]
                frame_nr = img_file.split('/')[-1].removesuffix('.jpg').removeprefix('frame')
                tools = self.tool_annotation_frames[vid_id].values[int(frame_nr) - 1][1:]
                # CATARACTS tool labels contain 0.5 values, these are mapped to 1
                tools[tools > 0] = 1
                n_tools = tools[tools > 0].sum()
                if n_tools == 0:
                    weights[i] = NO_TOOL_WEIGHT
                else:
                    weights[i] = (weight_per_class * tools).sum() / n_tools
            weights = np.array(weights)
            np.save(os.path.join(self.root, 'tool_sample_weights.npy'), weights)
        else:
            weights = np.load(os.path.join(self.root, 'tool_sample_weights.npy'))

        return weights, dist, weight_per_class

    def get_fold(self, idx: int, n_folds: int = 5):
        all_ids = np.arange(0, len(self))
        splits = np.array_split(all_ids, indices_or_sections=n_folds)
        val_ids = splits[idx]
        train_ids = np.concatenate([splits[i] for i in np.delete(np.arange(0, n_folds), idx)])
        return train_ids, val_ids

    def transform(self, img_list: list) -> torch.Tensor:

        aug_list = [A.Resize(*self.resize_shape)]

        if self.crop_shape is not None:
            aug_list.append(A.RandomCrop(*self.crop_shape, always_apply=True))

        if self.random_hflip:
            aug_list.append(A.HorizontalFlip(p=.3))

        if self.random_brightness_contrast:
            aug_list.append(A.RandomBrightnessContrast(p=.3))

        transf = A.Compose(aug_list,
                           additional_targets={'image' + str(key): 'image' for key in range(0, len(img_list) - 1)})

        kwargs = {}
        kwargs['image'] = img_list[0]
        for key in range(0, len(img_list) - 1):
            kwargs['image' + str(key)] = img_list[key + 1]

        data = transf(**kwargs)

        img_tensor = []
        for str_key in kwargs.keys():
            img = data[str_key]
            img = torch.from_numpy(img) / 255.0
            if self.normalize is not None:
                TF.normalize(img, mean=self.normalize[0], std=self.normalize[1], inplace=True)
            img_tensor.append(img.permute(2, 0, 1).unsqueeze(0))  # T, C, H, W
        img_tensor = torch.cat(img_tensor, dim=0)

        return img_tensor

    def __getitem__(self, index: int):
        sample = self.sample_list[index]
        img_files = sample['img']
        file_names = sample['name']
        phase_labels = sample['phase']

        img_list = []
        file_name_list = []
        phase_label_tensor = []
        tool_label_tensor = []

        for img_file, file_name, phase_label in zip(img_files, file_names, phase_labels):

            if self.sample_img:
                try:
                    img = jpeg.JPEG(img_file).decode()
                except Exception:
                    img = Image.open(img_file).convert('RGB')

                img = np.array(img) if self.sample_img else np.zeros(1)

            vid_id = img_file.split('/')[-2]
            frame_nr = img_file.split('/')[-1].removesuffix('.jpg').removeprefix('frame')

            tools = self.tool_annotation_frames[vid_id].values[int(frame_nr) - 1][1:]

            # CATARACTS tool labels contain 0.5 values, these are mapped to 1
            tools[tools > 0] = 1

            img_list.append(img)
            file_name_list.append(file_name)

            # Phase to integer
            phase_label_tensor.append(torch.tensor([phase_label]).unsqueeze(0))  # T, 1

            # Tool to one-hot
            tool_label_tensor.append(torch.tensor(tools).unsqueeze(0))  # T, K

        # Tensors are squeezed for nt = 1
        if self.sample_img:
            img_tensor = self.transform(img_list).squeeze(0)
        else:
            img_tensor = torch.zeros(size=(self.n_seq_frames, 1))
        phase_label_tensor = torch.cat(phase_label_tensor, dim=0).squeeze(0)
        tool_label_tensor = torch.cat(tool_label_tensor, dim=0).squeeze(0)

        label = torch.cat([phase_label_tensor, tool_label_tensor], dim=-1)  # T, K + 1

        return img_tensor, torch.zeros(size=(self.n_seq_frames, 1)), \
            torch.zeros(size=(self.n_seq_frames, 1)), label

    def __len__(self):
        return len(self.sample_list)

    def __dataset__(self):
        return self
