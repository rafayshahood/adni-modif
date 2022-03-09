from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import utils


class Filter:

    def __init__(self, model: nn.Sequential, architecture: str = "efficientnet_b0") -> None:
        self.model = model
        self.model_weights = []
        self.conv_layers = []

    def set_conv_weights(self, mbconv: int = 3, block: int = 1):

        print(self.model)

        # get a model with its all sub-components
        model_children = list(self.model.children())

        for i in range(3, 7):
            # get a conv layer and its weights
            conv_layer = model_children[0][i][block].block[1][0]
            self.model_weights.append(conv_layer.weight)
            self.conv_layers.append(conv_layer)

    def visualize_filters(self, ch=0, allkernels=False, nrow=32, padding=1):

        for i, filters_weigth in enumerate(self.model_weights):
            tensor = filters_weigth.data.clone()
            n, c, w, h = tensor.shape

            if allkernels:
                tensor = tensor.view(n * c, -1, w, h)
            tensor = tensor[:, ch, :, :].unsqueeze(dim=1)

            rows = np.min((tensor.shape[0] // nrow + 1, 32))
            grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
            plt.figure(figsize=(nrow, rows))
            plt.imshow(grid.numpy().transpose((1, 2, 0)))

            plt.axis('off')
            plt.ioff()
            #plt.show()
            plt.savefig('filter_weights_{}.png'.format(i))



