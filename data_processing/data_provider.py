import random
from typing import Tuple

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms


def get_transform_functions():
    """
    Create random transform functions that will be applied to input data
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


class DataProviderSSL(Dataset):
    """
    Provides the access to data.
    """

    def __init__(self, files: list, targets: list, diagnoses: list, slices_range: int, mode: str) -> None:
        """
        Initialize with all required attributes.
        :param files: paths to data
        :param targets: diagnoses as int values
        :param diagnoses: diagnoses as str values
        :param slices_range: the range from which slices will be sampled
        :param mode: Mode
        """
        self.transform_func = get_transform_functions()
        self.files = files
        self.targets = targets
        self.diagnoses = diagnoses
        self.slices_range = slices_range
        self.nr_samples = len(self.targets)
        self.mode = mode

    def __len__(self) -> int:
        """
        Return the number of samples.
        :return: the number of samples.
        """
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Return two different views of one slice and the corresponding target/label.
        :param idx: the ID of a sample.
        :return: two different views of one slice and the corresponding target/label.
        """
        label = self.targets[idx]
        filename = self.files[idx]

        slice_data = torch.load(filename)
        slice_data = slice_data.squeeze(dim=0)

        middle_point = int(slice_data.shape[1] / 2)  # (m) idx of the middle slice across one plane

        # a random value within the range [m-n, m+n]
        reference_point = random.randrange(middle_point - int(self.slices_range / 2),
                                           middle_point + int(self.slices_range / 2))

        # view: select coronal slice, correct view by rotating, put channels first
        coronal_view = torch.rot90(slice_data[:, reference_point, :].unsqueeze(dim=0), k=1, dims=(1, 2))

        # apply transformations
        if self.mode == "training":
            view_one = self.transform_func(coronal_view)
            view_two = self.transform_func(coronal_view)
        else:
            view_one = coronal_view
            view_two = coronal_view

        return view_one, view_two, label
