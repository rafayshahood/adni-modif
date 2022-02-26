import random
from typing import Tuple

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class DataProviderSSL(Dataset):
    """
    Provides the access to data.
    """

    def __init__(self, files: list, targets: list, diagnoses: list, slices_range: int, slices_per_view: int,
                 mode: str) -> None:
        """
        Initialize with all required attributes.
        :param files: paths to data
        :param targets: diagnoses as int values
        :param slices_range: the range from which slices will be sampled
        :param slices_per_view: number of slices used as channels in input data
        :param mode: Mode
        """
        self.files = files
        self.targets = targets
        self.diagnoses = diagnoses
        self.slices_range = slices_range
        self.slices_per_view = slices_per_view
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

        # slices within the range [r - s, r)
        view_one = slice_data[:, reference_point - self.slices_per_view:reference_point, :]
        view_one = torch.swapaxes(view_one, 0, 1)

        # slices within the range (r, r + s]
        view_two = slice_data[:, reference_point + 1:reference_point + self.slices_per_view + 1, :]
        view_two = torch.swapaxes(view_two, 0, 1)

        # apply transformations
        if self.mode == "training":
            rnd_erase = transforms.RandomErasing()
            transform = transforms.Compose([rnd_erase])
            view_one = transform(view_one)
            view_two = transform(view_two)

        return view_one, view_two, label
