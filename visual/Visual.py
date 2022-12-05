import logging
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import scipy
import scipy.stats
import seaborn as sns
import torch
from captum.attr import IntegratedGradients, NoiseTunnel
from captum.attr._utils.visualization import visualize_image_attr
from matplotlib import pyplot
from mlxtend.plotting import plot_confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE as TSNE_sklearn
from sklearn.metrics import roc_curve, roc_auc_score
from torch import nn
from tsnecuda import TSNE

from data_processing.utils import FREEZE_BACKBONE, SEED, string_to_bool

sns.set(font_scale=1.6)


def plot_cm_mlxtend(conf_matrix, class_names, output_folder, title="", suffix=""):
    fig, ax = plot_confusion_matrix(conf_mat=conf_matrix, figsize=(6, 6), cmap=plt.cm.Greens, class_names=class_names)
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title(title, fontsize=18)
    plt.tight_layout()
    plt.savefig("{}{}.pdf".format(output_folder, suffix), format="pdf")
    plt.savefig("{}{}.png".format(output_folder, suffix), format="png")
    plt.close()


def plot_loss_figure(log_files, suffix, out_dir, title: str, is_classifier_mode: bool = False):
    epochs = []
    losses = []
    backbone_identifiers = []
    for log_file in log_files:
        with open(log_file) as file:
            search_backbone = True
            backbone_identifier = None
            for line in file:
                if search_backbone and is_classifier_mode and FREEZE_BACKBONE in line:
                    backbone_identifier = int(string_to_bool(line.split(FREEZE_BACKBONE)[-1][:-1]))
                    search_backbone = False
                if "epoch:" in line:
                    epochs.append(re.search('\d+', line.split("epoch")[-1]).group())
                    losses.append(re.search('\d+\.\d+', line.split("loss")[-1]).group())
                    backbone_identifiers.append("ConvNeXt is not trained" if backbone_identifier else "ConvNeXt is trained")
                if SEED in line:
                    search_backbone = True

    epochs = [float(x) for x in epochs]
    losses = [float(x) for x in losses]
    if not is_classifier_mode:
        ax = sns.lineplot(x=epochs, y=losses, ci="sd")
    else:
        ax = sns.lineplot(x=epochs, y=losses, ci="sd", hue=backbone_identifiers, style=backbone_identifiers,
                          markers=False, dashes=True)
    ax.set(xlabel="Epoch", ylabel="Training loss", title=title)
    plt.tight_layout()
    plt.savefig("{}{}.pdf".format(out_dir, suffix), format="pdf")
    plt.savefig("{}{}.png".format(out_dir, suffix), format="png")
    plt.close()


def plot_2d_tsne(data: np.array,
                 perplexities: list,
                 class_mapping: dict,
                 labels_filter: list = None,
                 n: int = None,
                 with_pca: bool = False):
    """
    Creates a t-SNE visualisation for 4 perplexity values.
    :param with_pca:
    :param labels_filter:
    :param data: Array containing data and class
    :param perplexities: related to the number of nearest neighbors
    :param class_mapping: a mapping from int ot str
    :param n: samples per each class that are used for visualisation
    """

    df = pd.DataFrame(data)
    df.replace({df.columns[data.shape[1] - 1]: class_mapping}, inplace=True)  # map int values to str values
    df = df[df[df.columns[data.shape[1] - 1]].isin(labels_filter)]
    if n is not None:
        df = df.groupby(df.columns[data.shape[1] - 1]).apply(lambda x: x.sample(n))
        df.reset_index(drop=True, inplace=True)  # sample n samples for visualisation

    X = df.iloc[:, :-1]  #

    if df.shape[1] > 50 and with_pca:
        pca = PCA(n_components=50)
        pca.fit(X)
        X = pca.transform(X)

    Y = df.iloc[:, -1]  # class labels
    C = len(np.unique(Y))

    x_embeded_list = []
    for i, perplexity in enumerate(perplexities):
        x_embeded = TSNE(perplexity=perplexity, n_iter=10000, random_seed=1111).fit_transform(X)
        df = pd.DataFrame(x_embeded, columns=["X1", "X2"])
        df["Perplexity"] = perplexity
        df["Y"] = Y
        x_embeded_list.append(df)

    df = pd.concat(x_embeded_list)
    g = sns.FacetGrid(df, col="Perplexity", col_wrap=2, height=2)
    g.map_dataframe(sns.scatterplot, x="X1", y="X2", hue="Y", alpha=0.4, palette=sns.color_palette("hls", C), s=20,
                    style="Y")
    g.add_legend()
    g.set(xticks=[])
    g.set(yticks=[])
    g.set(xlabel=None)
    g.set(ylabel=None)
    plt.savefig("./images/tsne_2D.pdf", format="pdf")
    plt.savefig("./images/tsne_2D.png", format="png")
    plt.close()


def plot_3d_tsne(data: np.array,
                 perplexity: int, class_mapping: dict,
                 n_vis: int = None, labels_filter: list = None,
                 with_pca: bool = False):
    """
    Creates a t-SNE visualisation for 4 perplexity values.
    :param data: Array containing data and class
    :param perplexity: related to the number of nearest neighbors
    :param class_mapping: a mapping from int ot str
    :param n_vis: samples per each class that are used for visualisation
    """

    df = pd.DataFrame(data)
    df.replace({df.columns[data.shape[1] - 1]: class_mapping}, inplace=True)  # map int values to str values
    df = df[df[df.columns[data.shape[1] - 1]].isin(labels_filter)]
    if n_vis is not None:
        df = df.groupby(df.columns[data.shape[1] - 1]).apply(lambda x: x.sample(n_vis))
        df.reset_index(drop=True, inplace=True)  # sample n samples for visualisation

    X = df.iloc[:, :-1]  # features

    if data.shape[1] > 50 and with_pca:
        pca = PCA(n_components=50)
        pca.fit(X)
        X = pca.transform(X)

    Y = df.iloc[:, -1]  # class labels
    projections = TSNE_sklearn(n_components=3, perplexity=perplexity, n_iter=10000).fit_transform(X)
    df = pd.DataFrame(projections, columns=["X", "W", "Z"])
    df["Y"] = list(Y)

    fig = px.scatter_3d(
        projections, x=0, y=1, z=2,
        color=df.Y, labels={'color': 'Y'}
    )
    fig.update_traces(marker_size=4)
    fig.show()


def plot_density(data, filename):
    data.loc[data['Group'] == 0, 'Group'] = "Control"
    data.loc[data['Group'] == 1, 'Group'] = "Patient"
    sns.set(rc={'figure.figsize': (10, 10)})
    ax = sns.displot(data, x="Preds", hue="Group", kind="kde", fill=True, height=8, rug=True)
    plt.xlabel('Predictions', fontsize=16)
    plt.ylabel('Density', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.axvline(0, color='red', linestyle='--')
    plt.gcf().subplots_adjust(bottom=0.1, left=0.1)
    plt.savefig(filename)
    plt.close()


def plot_cm(conf_matrix):
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()


def plot_roc(y, probs):
    fpr, tpr, thresholds = roc_curve(y, probs)
    auc = roc_auc_score(y, probs)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    logging.info('NOptimal threshold=%.3f' % (optimal_threshold))

    # generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(y))]

    ns_auc = roc_auc_score(y, ns_probs)
    lr_auc = roc_auc_score(y, probs)
    # summarize scores
    logging.info('Random Classifier: ROC AUC=%.3f' % (ns_auc))
    logging.info('Disease Classifier: ROC AUC=%.3f' % (lr_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y, probs)
    # plot the roc curve for the model
    pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='Random Classifier (area = 0.5)')
    pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Disease Classifier (area = %.3f)' % (lr_auc))
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()


def mean_confidence_interval(data, confidence=0.95):
    # taken from https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    print("Mean: {}; SE: {}".format(m, se))
    return m, m - h, m + h, se


class FeatureMap:

    def __init__(self, model: nn.Sequential) -> None:
        self.model = model
        self.conv_layers = []

    def extract_conv_weights(self):

        print(self.model)

        # get a model with its all sub-components
        model_children = list(list(self.model.children())[0])

        counter = 0
        for i in range(len(model_children)):
            for m in model_children[i].modules():
                if type(m) == nn.Conv2d:
                    counter += 1
                    self.conv_layers.append(m)
            # elif type(model_children[i]) == nn.Sequential:
            #     for j in range(len(model_children[i])):
            #         for child in model_children[i][j].children():
            #             if type(child) == nn.Conv2d:
            #                 counter += 1
            #                 self.conv_layers.append(child)

        print(f"Total convolutional layers: {counter}")

    def visualize_feature_maps(self, sample, output_folder):
        # pass the image through all the layers
        results = [self.conv_layers[0](sample)]
        for i in range(1, len(self.conv_layers)):
            # pass the result from the last layer to the next layer
            results.append(self.conv_layers[i](results[-1]))
        # make a copy of the `results`
        outputs = results.copy()

        for num_layer in range(len(outputs)):
            plt.figure(figsize=(30, 30))
            layer_viz = outputs[num_layer]
            layer_viz = layer_viz.data.detach().cpu()
            print(layer_viz.size())
            c = 1
            for i, filter in enumerate(layer_viz):
                if i % (layer_viz.shape[0] / 16) == 0:
                    plt.subplot(4, 4, c)
                    plt.imshow(filter, cmap='gray')
                    plt.axis("off")
                    c += 1
                if c == 17:
                    break
            plt.savefig("{}feature_maps_layer-{}.png".format(output_folder, num_layer))
            plt.close()


from matplotlib import cm
def plot_attributions(sample, model, target, name, output_folder, device="cuda", sigma=1.0, alpha=0.5):
    ig = IntegratedGradients(model)
    nt = NoiseTunnel(ig)
    sample = torch.unsqueeze(sample, dim=0)
    baseline = torch.zeros((1, 1, 179, 169), device=device)
    #attributions, delta = nt.attribute(sample, nt_type='smoothgrad', stdevs=0.02, nt_samples=5,
    attributions, delta = nt.attribute(sample, nt_type='smoothgrad', stdevs=0.02, nt_samples=20,
    #attributions, delta = nt.attribute(sample, nt_type='smoothgrad_sq', stdevs=0.02, nt_samples=20,
                                       baselines=baseline, target=target, return_convergence_delta=True)

    np_attribution = torch.unsqueeze(torch.squeeze(attributions), dim=2).detach().cpu().numpy()
    np_attribution = scipy.ndimage.filters.gaussian_filter(np_attribution, sigma=sigma)  # smooth activity image
    np_sample = torch.unsqueeze(torch.squeeze(sample), dim=2).detach().cpu().numpy()

    fig, ax = plot_overlay_image(np_attribution, np_sample, alpha_overlay=alpha, alpha_threshold=0.2, cmap= None) #cm.get_cmap('RdYlGn_r'))
    #fig, ax = visualize_image_attr(np_attribution, np_sample, "blended_heat_map", alpha_overlay=alpha, show_colorbar=True,
    #                               sign="all", use_pyplot=False)
    ax.plot()
    plt.show()
    fig.savefig("{}{}.png".format(output_folder, name), format="png")



from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.pyplot import axis, figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
def plot_overlay_image(attr, original_image, alpha_overlay=0.5, alpha_threshold=0.2, cmap = None, sign = "absolute_value", outlier_perc = 2, show_colorbar = False, use_pyplot = True):
    plt_fig = Figure(figsize=(6,6))
    plt_axis = plt_fig.subplots()
    default_cmap = LinearSegmentedColormap.from_list(
                "RdWhGn", ["red", "white", "green"]
            )
    vmin, vmax = -1, 1
    if cmap is None:
        cmap = default_cmap

    # Remove ticks and tick labels from plot.
    plt_axis.xaxis.set_ticks_position("none")
    plt_axis.yaxis.set_ticks_position("none")
    plt_axis.set_yticklabels([])
    plt_axis.set_xticklabels([])
    plt_axis.grid(b=False)
    
    # Choose appropriate signed attributions and normalize.
    norm_attr = _normalize_image_attr(attr, sign, outlier_perc)
    plt_axis.imshow(np.mean(original_image, axis=2), cmap="gray")

    # TODO: add heatmap with proper thresholding
    plt_axis.imshow(overlay2rgba(norm_attr, alpha=alpha_overlay, alpha_threshold=alpha_threshold, overlay_colormap=cmap))
    heat_map = True
    #heat_map = plt_axis.imshow(
    #    norm_attr, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha_overlay
    #)

    # Add colorbar. If given method is not a heatmap and no colormap is relevant,
    # then a colormap axis is created and hidden. This is necessary for appropriate
    # alignment when visualizing multiple plots, some with heatmaps and some
    # without.
    if show_colorbar:
        axis_separator = make_axes_locatable(plt_axis)
        colorbar_axis = axis_separator.append_axes("bottom", size="5%", pad=0.1)
        if heat_map:
            plt_fig.colorbar(heat_map, orientation="horizontal", cax=colorbar_axis)
        else:
            colorbar_axis.axis("off")

    if use_pyplot:
        plt.show()

    return plt_fig, plt_axis


from PIL import Image
def overlay2rgba(relevance_map, alpha=0.5, alpha_threshold=0.2, overlay_colormap=None):
    """
    Converts the 3D relevance map to RGBA.

    :param numpy.ndarray relevance_map: The 3D relevance map.
    :param float alpha: the transparency/the value for the alpha channel.
    :return: the voxel values converted to RGBA data.
    :rtype: numpy.ndarray
    """
    # assume map to be in range of -1..1 with 0 for hidden content
    alpha_mask = np.zeros_like(relevance_map)
    alpha_mask[np.abs(relevance_map) > alpha_threshold] = alpha  # final transparency of visible content
    relevance_map = relevance_map / 2 + 0.5  # range 0-1 float
    ovl = np.uint8(overlay_colormap(relevance_map) * 255)  # cm translates range 0 - 255 uint to rgba array
    ovl[:, :, 3] = np.uint8(alpha_mask * 255)          #ovl[:, :, 3]              # replace alpha channel (fourth dim) with calculated values
    ret = Image.fromarray(ovl)
   
    return ret


class VisualizeSign:
    positive = 1
    absolute_value = 2
    negative = 3
    all = 4


def _prepare_image(attr_visual):
    return np.clip(attr_visual.astype(int), 0, 255)


def _normalize_scale(attr, scale_factor):
    assert scale_factor != 0, "Cannot normalize by scale factor = 0"
    if abs(scale_factor) < 1e-5:
        warnings.warn(
            "Attempting to normalize by value approximately 0, visualized results"
            "may be misleading. This likely means that attribution values are all"
            "close to 0."
        )
    attr_norm = attr / scale_factor
    return np.clip(attr_norm, -1, 1)


def _cumulative_sum_threshold(values, percentile):
    # given values should be non-negative
    assert percentile >= 0 and percentile <= 100, (
        "Percentile for thresholding must be " "between 0 and 100 inclusive."
    )
    sorted_vals = np.sort(values.flatten())
    cum_sums = np.cumsum(sorted_vals)
    threshold_id = np.where(cum_sums >= cum_sums[-1] * 0.01 * percentile)[0][0]
    return sorted_vals[threshold_id]


def _normalize_image_attr(attr, sign, outlier_perc = 2):
    attr_combined = np.sum(attr, axis=2)
    # Choose appropriate signed values and rescale, removing given outlier percentage.
    #if VisualizeSign[sign] == VisualizeSign.all:
    threshold = _cumulative_sum_threshold(np.abs(attr_combined), 100 - outlier_perc)
    #elif VisualizeSign[sign] == VisualizeSign.positive:
    #    attr_combined = (attr_combined > 0) * attr_combined
    #    threshold = _cumulative_sum_threshold(attr_combined, 100 - outlier_perc)
    #elif VisualizeSign[sign] == VisualizeSign.negative:
    #    attr_combined = (attr_combined < 0) * attr_combined
    #    threshold = -1 * _cumulative_sum_threshold(
    #        np.abs(attr_combined), 100 - outlier_perc
    #    )
    #elif VisualizeSign[sign] == VisualizeSign.absolute_value:
    #    attr_combined = np.abs(attr_combined)
    #    threshold = _cumulative_sum_threshold(attr_combined, 100 - outlier_perc)
    #else:
    #    raise AssertionError("Visualize Sign type is not valid.")
    return _normalize_scale(attr_combined, threshold)




def search_for_files(search_dir: str, file_identifier: str):
    """
    Searches for files within a directory.
    :param search_dir:  a search directory
    :param file_identifier: an identifier for relevant files
    :return: a list of found files
    """
    paths = []
    for root, dirs, files in os.walk(search_dir):
        for name in files:
            if file_identifier in name:
                paths.append(os.path.join(os.path.abspath(root), name))
    return paths

# input_dir = "/mnt/ssd2/ClinicNET/data/adni3/CAPS/subjects/"
# output_dir = "/mnt/ssd2/ClinicNET/data/temp/ADNI3/"
# for root, dirs, files in os.walk(input_dir):
#     for name in files:
#         if "ses-M00" in name and name.endswith(".pt"):
#             file_path = os.path.join(os.path.abspath(root), name)
#             if not os.path.exists(output_dir):
#                 os.makedirs(output_dir)
#             copy2(file_path, output_dir)
