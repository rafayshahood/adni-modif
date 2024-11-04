import random
from typing import Tuple

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

import nibabel as nib  # Ensure nibabel is imported
from data_processing.utils import Mode


def get_transform_functions():
    """
    Creates random transform functions that will be applied to input data.
    :return: transform functions
    """
    rnd_resizedcrop = transforms.RandomResizedCrop(size=(179, 169),
                                                   scale=(0.08, 1.0),
                                                   ratio=(0.75, 1.3333333333333333),
                                                   interpolation=transforms.InterpolationMode.BILINEAR)
    rnd_erase = transforms.RandomErasing()
    rnd_vflip = transforms.RandomHorizontalFlip()
    transform = transforms.Compose([rnd_vflip, rnd_resizedcrop, rnd_erase])

    return transform


class DataProvider(Dataset):
    """
    Provides the access to data.
    """

    def __init__(self, files: list, targets: list, diagnoses: list, slices_range: int, mode: Mode,
                 middle_slice: bool = False) -> None:
        """
        Initialize with all required attributes.
        :param files: paths to data
        :param targets: diagnoses as int values
        :param diagnoses: diagnoses as str values
        :param slices_range: the range from which slices will be sampled
        :param mode: Mode object
        :param middle_slice: If True then always a middle coronal slice will be selected, otherwise a random slice
        """
        self.transform_func = get_transform_functions()
        self.files = files
        self.targets = targets
        self.diagnoses = diagnoses
        self.slices_range = slices_range
        self.nr_samples = len(self.targets)
        self.mode = mode
        self.middle_slice = middle_slice

    def __len__(self) -> int:
        """
        Returns the number of samples.
        :return: the number of samples.
        """
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Returns two different views of one slice and the corresponding target/label.
        :param idx: the ID of a sample.
        :return: two different views of one slice and the corresponding target/label.
        """
        label = self.targets[idx]
        filename = self.files[idx]

        try:
            # Load the 3D MRI data
            nifti_data = nib.load(filename)
            volume_data = nifti_data.get_fdata()

            # Select a 2D slice, e.g., the middle slice along one axis
            middle_slice_idx = volume_data.shape[2] // 2  # Selecting the middle slice in the depth dimension
            slice_data = volume_data[:, :, middle_slice_idx]  # Shape: [height, width]

            # Convert to a tensor with an added channel dimension for grayscale
            slice_data = torch.tensor(slice_data, dtype=torch.float32).unsqueeze(0)  # Shape: [1, height, width]

        except Exception as e:
            # print(f"Error loading file {filename}: {e}")
            raise

        # Apply transformations for data augmentation during training
        if self.mode == Mode.training:
            view_one = self.transform_func(slice_data)
            view_two = self.transform_func(slice_data)
        else:
            # During evaluation, no augmentation is applied; same slice returned for both views
            view_one = slice_data
            view_two = slice_data

        return view_one, view_two, label
