import torch
import torchvision
from torch import nn

from models.nnclr.nnclr import NNCLR
from visual.Visual import FeatureMap

FILE = "/data_dzne_archiv2/Studien/ClinicNET/data/adni2/CAPS/subjects/sub-ADNI002S0729/ses-M60/deeplearning_prepare_data/image_based/t1_linear/sub-ADNI002S0729_ses-M60_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.pt"
LABEL = "AD"


def get_item(file_path: str):
    slice_data = torch.load(file_path)
    slice_data = slice_data.squeeze(dim=0)

    middle_point = int(slice_data.shape[1] / 2)  # (m) idx of the middle slice across one plane

    coronal_view = torch.rot90(slice_data[:, middle_point, :].squeeze(), k=1, dims=(0, 1))
    view = torch.stack([coronal_view, coronal_view, coronal_view], dim=1)
    view = torch.swapaxes(view, 0, 1)

    return view


sample = get_item(FILE)
backbone = torchvision.models.convnext_tiny()
backbone = nn.Sequential(*list(backbone.children())[:-1])
backbone = NNCLR.load_state_dict_(backbone,
                                  '/mnt/ssd2/ClinicNET/checkpoints/convnext_tiny_fine_tuned/nnclr_epoch_1000.ckpt')

visual_filter = FeatureMap(backbone)
visual_filter.extract_conv_weights()
visual_filter.visualize_feature_maps(sample)
