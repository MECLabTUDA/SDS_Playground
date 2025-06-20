from typing import Union

import torch
import torch.nn.functional as F


def normalize(x: torch.Tensor, mean: Union[torch.Tensor, float], std: Union[torch.Tensor, float]) -> torch.Tensor:
    return (x - mean) / std


def denormalize(x: torch.Tensor, mean: Union[torch.Tensor, float], std: Union[torch.Tensor, float]) -> torch.Tensor:
    return (x * std) + mean


def convert_to_binary_mask(mask: torch.Tensor,
                           num_classes: int,
                           ignore_index: Union[int, None],
                           keep_ignore_index: bool = True) -> torch.Tensor:
    """
    Convert an integer segmentation mask to a binary segmentation mask.

    Parameters:
    mask (torch.Tensor): The integer segmentation mask, shape (B, H, W).
    num_classes (int): The number of classes, including ignore index if any!
    ignore_index (int|None):
    keep_ignor_index (bool): If True, adds additional channel for the ignore index

    Returns:
    binary_mask (torch.Tensor): The binary segmentation mask, shape (B, K, H, W).
    """
    B, H, W = mask.shape
    device = mask.device
    dtype = torch.float32  # Using float32 for compatibility with scatter_

    # Adjust num_classes if keeping ignore index as a separate channel
    # if keep_ignore_index and ignore_index is not None:
    #    num_classes += 1

    if keep_ignore_index or ignore_index is None or ignore_index < num_classes:
        binary_mask = torch.zeros((B, num_classes, H, W), dtype=dtype, device=device)
    else:
        binary_mask = torch.zeros((B, num_classes - 1, H, W), dtype=dtype, device=device)

    try:

        if ignore_index is not None and ignore_index >= num_classes:

            # Create a mask for valid (non-ignore) locations
            valid_mask = mask != ignore_index

            # Temporarily set ignore_index to 0 to avoid scatter_ errors
            adjusted_mask = torch.where(valid_mask, mask, torch.tensor(0, device=device))

            # Use scatter_ to populate the binary mask for valid indices
            binary_mask.scatter_(1, adjusted_mask.unsqueeze(1), valid_mask.unsqueeze(1).type(dtype))

            if keep_ignore_index:
                # Directly set the last channel for ignore_index locations
                binary_mask[:, -1] = (~valid_mask).type(dtype)
        elif ignore_index is not None and ignore_index < num_classes:
            binary_mask.scatter_(1, mask.unsqueeze(1), 1)
            if not keep_ignore_index:
                binary_mask = torch.cat([binary_mask[:, :ignore_index], binary_mask[:, ignore_index + 1:]], dim=1)
        else:
            binary_mask.scatter_(1, mask.unsqueeze(1), 1)

    except RuntimeError as e:
        print(e)
        print(f"{binary_mask.shape=}")
        print(f"{mask.shape=}")
        print(f"{torch.unique(mask)=}")
        exit()

    return binary_mask


def _convert_to_binary_mask(mask: torch.Tensor,
                            num_classes: int,
                            ignore_index: Union[list, None],
                            keep_ignore_index: bool = True) -> torch.Tensor:
    """
    Convert an integer segmentation mask to a binary segmentation mask, excluding certain indices.

    Parameters:
    mask (torch.Tensor): The integer segmentation mask, shape (B, H, W).
    num_classes (int): The number of classes, including ignore indices if any!
    ignore_indices (list[int]|None): List of indices to ignore.
    keep_ignore_index (bool): If True, adds additional channel for the ignore indices.

    Returns:
    binary_mask (torch.Tensor): The binary segmentation mask, shape (B, K, H, W).
    """
    B, H, W = mask.shape
    device = mask.device
    dtype = torch.float32  # Using float32 for compatibility with scatter_

    # Adjust num_classes if keeping ignore index as a separate channel
    if keep_ignore_index or ignore_index is None:
        binary_mask = torch.zeros((B, num_classes, H, W), dtype=dtype, device=device)
    else:
        binary_mask = torch.zeros((B, num_classes - len(ignore_index), H, W), dtype=dtype, device=device)

    if ignore_index is not None:
        # Create a mask for valid (non-ignore) locations using multiple ignore indices
        ignore_mask = torch.zeros_like(mask, dtype=torch.bool, device=device)
        for idx in ignore_index:
            ignore_mask |= (mask == idx)

        valid_mask = ~ignore_mask

        # Mask out any values in `mask` that exceed `num_classes - 1`
        mask_out_of_bounds = mask >= num_classes
        valid_mask &= ~mask_out_of_bounds  # Exclude out-of-bounds values from valid mask

        # Temporarily set ignore indices and out-of-bound values to 0 to avoid scatter_ errors
        adjusted_mask = torch.where(valid_mask, mask, torch.tensor(0, device=device))

        # Use scatter_ to populate the binary mask for valid indices
        binary_mask.scatter_(1, adjusted_mask.unsqueeze(1), valid_mask.unsqueeze(1).type(dtype))

        if keep_ignore_index:
            # Directly set the last channel for ignore index locations
            binary_mask[:, -1] = (~valid_mask).type(dtype)
    else:
        # Handle cases where no ignore indices are provided
        mask_out_of_bounds = mask >= num_classes
        valid_mask = ~mask_out_of_bounds

        # Ensure mask is within bounds of `num_classes`
        adjusted_mask = torch.clamp(mask, min=0, max=num_classes - 1)
        binary_mask.scatter_(1, adjusted_mask.unsqueeze(1), valid_mask.unsqueeze(1).type(dtype))

    return binary_mask


def convert_to_integer_mask(binary_mask: torch.Tensor,
                            num_classes: int,
                            ignore_index: Union[int, None]) -> torch.Tensor:
    """
    Convert a binary segmentation mask back to an integer segmentation mask using PyTorch.
    If an ignore index is provided, the last channel is converted to that index.

    Parameters:
    binary_mask (torch.Tensor): The binary segmentation mask, shape (B, K, H, W).

    Returns:
    mask (torch.Tensor): The integer segmentation mask, shape (B, H, W).
    """
    mask = torch.argmax(binary_mask, dim=1)
    if ignore_index is not None:
        if ignore_index >= num_classes:
            mask[mask == binary_mask.shape[1] - 1] = ignore_index
        else:
            mask[mask == ignore_index] = ignore_index
    return mask


def convert_mask_to_RGB(mask: torch.Tensor,
                        palette: torch.Tensor,
                        ignore_index: Union[int, None]) -> torch.Tensor:
    """
    Convert a segmentation mask into an RGB image.

    Parameters:
    mask (torch.Tensor): The segmentation mask, shape (B, H, W).
    palette (torch.Tensor): The color palette for segmentation map, shape (num_classes, 3).

    Returns:
    rgb_images (torch.Tensor): The RGB images, shape (B, 3, H, W).
    """

    # Check if ignore index exists in the mask
    if ignore_index is not None and (mask == ignore_index).any():
        # Extend the palette to have an entry for label 255
        palette = torch.cat([palette, torch.tensor([[0, 0, 0]], device=palette.device)], dim=0)
        mask = mask.clone()  # Clone to ensure we don't modify the original mask in place
        mask[mask == ignore_index] = palette.size(0) - 1

    # Convert the mask to one-hot encoded tensor
    mask_onehot = F.one_hot(mask, num_classes=palette.shape[0]).permute(0, 3, 1,
                                                                        2).float()  # shape: (B, num_classes, H, W)
    # mask_onehot.to(palette.device)

    # Expand palette dimensions to match mask_onehot
    palette = palette[None, :, :, None, None]  # shape: (1, num_classes, 3, 1, 1)

    # Convert one-hot to rgb by multiplying with palette and summing over the classes dimension
    rgb_images = (palette * mask_onehot[:, :, None, :, :]).sum(dim=1)  # shape: (B, 3, H, W)

    return rgb_images
