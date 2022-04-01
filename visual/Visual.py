from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import utils


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



