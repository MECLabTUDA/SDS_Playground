from typing import Union

import torch
from torch.utils.data import Dataset
import pickle
from tqdm import tqdm

from sds_playground.datasets import CaDISv2_Dataset


def load_weights(weights_file):
    with open(weights_file, 'rb') as f:
        weights = pickle.load(f)
    return torch.tensor(weights, dtype=torch.double)


def get_sample_weights(dataset: Dataset, power: int = 1, save_path: Union[str, None] = None):
    """ Function to calculate weights for each sample in a dataset """

    label_counts = {}
    for _, _, _, label_tensor in tqdm(dataset, desc=f"Counting {type(dataset).__name__} labels"):
        # Assuming label_tensor is a 1D tensor of binary labels
        labels = label_tensor.nonzero().flatten().tolist()  # Get indices of positive labels
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1

    # Compute weights for each label
    total_samples = len(dataset)
    label_weights = {label: total_samples / (len(label_counts) * count)**power for label, count in label_counts.items()}

    # Assign average weight to each sample based on its labels
    sample_weights = []
    for _, _, _, label_tensor in tqdm(dataset, desc=f"Computing {type(dataset).__name__} weights"):
        sample_labels = label_tensor.nonzero().flatten().tolist()
        if sample_labels:  # If there are positive labels
            sample_weight = sum(label_weights[label] for label in sample_labels) / len(sample_labels)
        else:  # Handle case with no positive labels, if necessary
            sample_weight = 1.0  # Default weight, adjust as needed
        sample_weights.append(sample_weight)

    # Save weights to a file, if specified
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(sample_weights, f)

    return torch.tensor(sample_weights, dtype=torch.double)


if __name__ == "__main__":

    ds = CaDISv2_Dataset(
        root='/local/scratch/CaDISv2/',
        exp=2,
        mode='train',
        filter_mislabeled=True
    )

    target_weights_file = '../../cadisv2_exp2_sample_weights.pkl'

    _ = get_sample_weights(dataset=ds, power=2, save_path=target_weights_file)


