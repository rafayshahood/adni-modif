import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import torch
from captum.attr import DeepLift
from captum.attr._utils.visualization import visualize_image_attr
from mlxtend.plotting import plot_confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE as TSNE_sklearn
from torch import nn
from tsnecuda import TSNE


def plot_cm_mlxtend(conf_matrix, title, suffix):
    fig, ax = plot_confusion_matrix(conf_mat=conf_matrix, figsize=(6, 6), cmap=plt.cm.Greens)
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title(title, fontsize=18)
    plt.savefig("./images/{}.pdf".format(suffix), format="pdf")
    plt.savefig("./images/{}.png".format(suffix), format="png")
    plt.close()


def plot_loss_figure(log_files, suffix):
    epochs = []
    losses = []
    for log_file in log_files:
        with open(log_file) as file:
            for line in file:
                if "epoch:" in line:
                    epochs.append(re.search('\d+', line.split("epoch")[-1]).group())
                    losses.append(re.search('\d+\.\d+', line.split("loss")[-1]).group())
    epochs = [float(x) for x in epochs]
    losses = [float(x) for x in losses]
    sns.lineplot(x=epochs, y=losses, ci="sd")
    plt.savefig("./images/{}.pdf".format(suffix), format="pdf")
    plt.savefig("./images/{}.png".format(suffix), format="png")
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

    def visualize_feature_maps(self, sample):
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
            plt.savefig(f"./output/feature_maps/layer_{num_layer}.png")
            plt.close()


def plot_attributions(sample, model, target, name):
    dl = DeepLift(model)
    sample = torch.unsqueeze(sample, dim=0)
    attribution = dl.attribute(sample, target=target)

    np_attribution = torch.unsqueeze(torch.squeeze(attribution), dim=2).detach().cpu().numpy()
    np_sample = torch.unsqueeze(torch.squeeze(sample), dim=2).detach().cpu().numpy()
    fig, ax = visualize_image_attr(np_attribution, np_sample, "blended_heat_map", alpha_overlay=0.5, show_colorbar=True,
                                  sign="negative", use_pyplot=False)
    ax.plot()
    fig.savefig("/mnt/ssd2/ClinicNET/output/{}.png".format(name), format="png")
