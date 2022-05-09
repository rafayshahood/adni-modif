import numpy as np
import torch
import yaml

from data_processing.utils import create_folder, get_item
from models.classifier import ClassificationModel, LOG_IDENTIFIER_CLASSIFIER
from models.nnclr import NNCLR, get_convnext, LOG_IDENTIFIER as NNCLR_LOG_IDENTIFIER
from visual.Visual import plot_loss_figure, plot_attributions, FeatureMap, \
    search_for_files, mean_confidence_interval, plot_cm_mlxtend

FILE_CN = "/mnt/ssd2/ClinicNET/data/aibl/CAPS/subjects/sub-AIBL98/ses-M00/deeplearning_prepare_data/image_based/t1_linear/sub-AIBL98_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.pt/sub-AIBL98_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.pt"
LABEL_CN = "CN"
FILE_AD = "/mnt/ssd2/ClinicNET/data/aibl/CAPS/subjects/sub-AIBL851/ses-M18/deeplearning_prepare_data/image_based/t1_linear/sub-AIBL851_ses-M18_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.pt/sub-AIBL851_ses-M18_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.pt"
LABEL_AD = "AD"

# Setup:
with open('./configuration/configuration.yaml', 'r') as stream:
    settings = yaml.load(stream, yaml.Loader)
figures_folder = create_folder(settings['working_dir'], "figures/")
visual_backbone_ckpt = settings['visualisation']['backbone_checkpoint']
visual_classifier_ckpt = settings['visualisation']['classifier_checkpoint']
logs_dir = settings['visualisation']['log_dir']
device = "cuda" if torch.cuda.is_available() else "cpu"
sample_cn = get_item(FILE_CN, shift=0)
sample_ad = get_item(FILE_AD, shift=0)

# Plot attributions:
backbone = get_convnext()
linear_eval = ClassificationModel(backbone, 2, freeze_backbone=False)
linear_eval.load(visual_classifier_ckpt, device=device)
linear_eval.to(device)
# linear_eval.eval()
# output = linear_eval(sample_cn.unsqueeze(0))
# _, predicted = torch.max(output, 1)
for i in range(-30, 31, 30):
    plot_attributions(get_item(file_path=FILE_CN, shift=i), linear_eval, target=0,
                      name="attributions_IG_{}_{}".format(LABEL_CN, i),
                      output_folder=figures_folder)
    plot_attributions(get_item(file_path=FILE_AD, shift=i), linear_eval, target=1,
                      name="attributions_IG_{}_{}".format(LABEL_AD, i),
                      output_folder=figures_folder)
for i in range(0, 2):
    plot_attributions(get_item(file_path=FILE_AD, shift=0), linear_eval, target=i,
                      name="attributions_IG_comparison_{}_output-{}".format(LABEL_AD, i),
                      output_folder=figures_folder)

# Independent evaluation (2 classes, see logs):
b_acc = [0.8081052702885787, 0.7867271824052748, 0.797108567041503]
mcc = [0.6039586111050201, 0.5836928183136699, 0.5786677215853893]
mean_confidence_interval(b_acc)
mean_confidence_interval(mcc)

# Independent evaluation (3 classes, see logs):
b_acc = [0.5478616594245441, 0.5242475293988286, 0.5193006001176804]
mcc = [0.33579998531822985, 0.2853459390148554, 0.27214416616012815]
mean_confidence_interval(b_acc)
mean_confidence_interval(mcc)

# Test (4 classes, see logs):
b_acc = [0.5633593707708178, 0.609588929813742, 0.6215943889383793]
mcc = [0.3104248637119342, 0.30445353383093143, 0.3490155319878724]
mean_confidence_interval(b_acc)
mean_confidence_interval(mcc)

# Test (2 classes, see logs):
b_acc = [0.8379952768302283, 0.7608447488584476, 0.8473687227700069]
mcc = [0.656438709378569, 0.4983431929660993, 0.668786948826537]
mean_confidence_interval(b_acc)
mean_confidence_interval(mcc)

# Plot the loss of NNCLR:
plot_loss_figure(search_for_files(logs_dir, NNCLR_LOG_IDENTIFIER), suffix="loss_nnclr", out_dir=figures_folder,
                 title="")

# Plot the loss of the classification model:
plot_loss_figure(search_for_files(logs_dir, "{}_4_".format(LOG_IDENTIFIER_CLASSIFIER)), suffix="loss_classifier",
                 out_dir=figures_folder, is_classifier_mode=True, title="")

# Plot feature maps:
backbone = get_convnext()
backbone = NNCLR.load_state_dict_(backbone, visual_backbone_ckpt)
backbone.to(device)
visual_filter = FeatureMap(backbone)
visual_filter.extract_conv_weights()
visual_filter.visualize_feature_maps(sample_ad, figures_folder)

plot_cm_mlxtend(np.array([[116, 18, 36, 26],
                          [8, 62, 11, 8],
                          [43, 53, 33, 13],
                          [1, 0, 0, 28]]
                         ),
                class_names=['CN', 'AD', 'MCI', 'BV'],
                output_folder=figures_folder,
                suffix="cm")
#
# # visualise features using t-SNE:
# data = np.load("/mnt/ssd2/ClinicNET/features/nifd_adni_aibl_train.npy")
# plot_2d_tsne(data, labels_filter=["AD", "BV"], class_mapping={0: 'CN', 1: 'AD', 2: 'BV', 3: 'MCI'},
#              perplexities=[5, 15, 30, 50])
# plot_3d_tsne(data, labels_filter=["AD", "BV"], class_mapping={0: 'CN', 1: 'AD', 2: 'BV', 3: 'MCI'}, perplexity=30)
