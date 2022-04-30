import torch
import yaml

from data_processing.utils import create_folder, get_item
from models.classifier import ClassificationModel, LOG_IDENTIFIER_CLASSIFIER
from models.nnclr import NNCLR, get_convnext, LOG_IDENTIFIER as NNCLR_LOG_IDENTIFIER
from visual.Visual import plot_loss_figure, plot_attributions, FeatureMap, \
    search_for_files

FILE = "/data_dzne_archiv2/Studien/ClinicNET/data/adni2/CAPS/subjects/sub-ADNI002S0729/ses-M60/deeplearning_prepare_data/image_based/t1_linear/sub-ADNI002S0729_ses-M60_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.pt"
LABEL = "AD"

# Setup:
with open('./configuration/configuration.yaml', 'r') as stream:
    settings = yaml.load(stream, yaml.Loader)
figures_folder = create_folder(settings['working_dir'], "figures/")
visual_backbone_ckpt = settings['visualisation']['backbone_checkpoint']
visual_classifier_ckpt = settings['visualisation']['classifier_checkpoint']
logs_dir = settings['visualisation']['log_dir']
sample = get_item(file_path=FILE, shift=0)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Plot feature maps:
backbone = get_convnext()
backbone = NNCLR.load_state_dict_(backbone, visual_backbone_ckpt)
backbone.to(device)
visual_filter = FeatureMap(backbone)
visual_filter.extract_conv_weights()
visual_filter.visualize_feature_maps(sample, figures_folder)

# Plot attributions:
backbone = get_convnext()
linear_eval = ClassificationModel(backbone, 4, freeze_backbone=False)
linear_eval.load(visual_classifier_ckpt, device=device)
linear_eval.to(device)
for i, k in enumerate(range(-10, 10)):
    sample = get_item(FILE, shift=k)
    plot_attributions(sample, linear_eval, target=0, name="attributions_IG_{}".format(i))

# Plot the loss of NNCLR:
plot_loss_figure(search_for_files(logs_dir, NNCLR_LOG_IDENTIFIER), suffix="loss_nnclr", out_dir=figures_folder)

# Plot the loss of the classification model:
plot_loss_figure(search_for_files(logs_dir, LOG_IDENTIFIER_CLASSIFIER), suffix="loss_classifier",
                 out_dir=figures_folder)

# mcc = [0.4294, 0.4318, 0.4167]
# precision = [0.5729, 0.5766, 0.5583]
# recall = [0.6978, 0.6970, 0.6861]
# mean_confidence_interval(mcc)

# plot_cm_mlxtend(np.array([[368, 30, 20, 79],
#                           [19, 130, 7, 27],
#                           [3, 0, 29, 0],
#                           [105, 60, 6, 100]]
#                          ),
#                 "Confusion matrix: \n0 - CN, 1 - AD, 2 - BV, 3 - MCI",
#                 suffix="cm")
#
# # visualise features using t-SNE:
# data = np.load("/mnt/ssd2/ClinicNET/features/nifd_adni_aibl_train.npy")
# plot_2d_tsne(data, labels_filter=["AD", "BV"], class_mapping={0: 'CN', 1: 'AD', 2: 'BV', 3: 'MCI'},
#              perplexities=[5, 15, 30, 50])
# plot_3d_tsne(data, labels_filter=["AD", "BV"], class_mapping={0: 'CN', 1: 'AD', 2: 'BV', 3: 'MCI'}, perplexity=30)
