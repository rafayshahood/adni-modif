import numpy as np
import torch
import torchvision
from torch import nn

from models.nnclr.linear_eval import ClassificationModel
from models.nnclr.nnclr import NNCLR, get_convnext
from visual.Visual import plot_2d_tsne, plot_3d_tsne, plot_loss_figure, plot_cm_mlxtend, plot_attributions

FILE = "/data_dzne_archiv2/Studien/ClinicNET/data/adni2/CAPS/subjects/sub-ADNI002S0729/ses-M60/deeplearning_prepare_data/image_based/t1_linear/sub-ADNI002S0729_ses-M60_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.pt"
LABEL = "AD"


def get_item(file_path: str, shift: int):
    slice_data = torch.load(file_path, map_location="cuda")
    slice_data.requires_grad = True
    slice_data = slice_data.squeeze(dim=0)

    middle_point = int(slice_data.shape[1] / 2)  # (m) idx of the middle slice across one plane

    # view: select coronal slice, correct view by rotating, put channels first
    coronal_view = torch.rot90(slice_data[:, middle_point+shift, :].unsqueeze(dim=0), k=1, dims=(1, 2))

    return coronal_view


# sample = get_item(FILE)
# backbone = torchvision.models.convnext_tiny()
# backbone = nn.Sequential(*list(backbone.children())[:-1])
# backbone = NNCLR.load_state_dict_(backbone,
#                                   '/mnt/ssd2/ClinicNET/checkpoints/convnext_tiny_fine_tuned/seed32/one_slice_nnclr_epoch_2000.ckpt')
#
# visual_filter = FeatureMap(backbone)
# visual_filter.extract_conv_weights()
# visual_filter.visualize_feature_maps(sample)

# Plot attributions
backbone = get_convnext()
linear_eval = ClassificationModel(backbone, 4, freeze_backbone=False)
linear_eval.load("/mnt/ssd2/ClinicNET/checkpoints/convnext_tiny_fine_tuned/seed32/one_slice_le.ckpt", "cuda")
linear_eval.to("cuda")

for i, k in enumerate(range(-10, 10)):
    sample = get_item(FILE, shift=k)
    plot_attributions(sample, linear_eval, target=0, name="attributions_deep_lift_{}".format(i))

# plot the loss of NNCLR:
plot_loss_figure(["/mnt/ssd2/ClinicNET/log/version3/debug_nnclr_32.log"],
                 suffix="loss_nnclr")

# plot the loss of the classification model:
files = ["/mnt/ssd2/ClinicNET/log/version1/debug_le_22.log",
         "/mnt/ssd2/ClinicNET/log/version1/debug_le_32.log",
         "/mnt/ssd2/ClinicNET/log/version1/debug_le_42.log"]
plot_loss_figure(files, suffix="loss_classification_model")

plot_cm_mlxtend(np.array([[368, 30, 20, 79],
                          [19, 130, 7, 27],
                          [3, 0, 29, 0],
                          [105, 60, 6, 100]]
                         ),
                "Confusion matrix: \n0 - CN, 1 - AD, 2 - BV, 3 - MCI",
                suffix="cm")

# visualise features using t-SNE:
data = np.load("/mnt/ssd2/ClinicNET/features/nifd_adni_aibl_train.npy")
plot_2d_tsne(data, labels_filter=["AD", "BV"], class_mapping={0: 'CN', 1: 'AD', 2: 'BV', 3: 'MCI'},
             perplexities=[5, 15, 30, 50])
plot_3d_tsne(data, labels_filter=["AD", "BV"], class_mapping={0: 'CN', 1: 'AD', 2: 'BV', 3: 'MCI'}, perplexity=30)
