'''A quick example to generate heatmaps for vgg16.'''
import random

import click
import nibabel
import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.backends import cudnn
from torchvision.models import vgg16, vgg16_bn, resnet50, resnet18
from zennit.attribution import Gradient, SmoothGrad, IntegratedGradients, Occlusion
from zennit.composites import COMPOSITES
from zennit.image import imsave, CMAPS
from zennit.torchvision import VGGCanonizer, ResNetCanonizer

from configuration.configuration import Configuration
from data_processing.data_loader import DataLoaderSSL
from models.nnclr.linear_eval import LinearEval

MODELS = {
    'vgg16': (vgg16, VGGCanonizer),
    'vgg16_bn': (vgg16_bn, VGGCanonizer),
    'resnet50': (resnet50, ResNetCanonizer),
    'resnet18': (resnet18, ResNetCanonizer),
}

ATTRIBUTORS = {
    'gradient': Gradient,
    'smoothgrad': SmoothGrad,
    'integrads': IntegratedGradients,
    'occlusion': Occlusion,
}


@click.command()
# @click.argument('dataset-root', type=click.Path(file_okay=False))
@click.argument('relevance_format', type=click.Path(dir_okay=False, writable=True))
@click.option('--attributor', 'attributor_name', type=click.Choice(list(ATTRIBUTORS)), default='gradient')
@click.option('--composite', 'composite_name', type=click.Choice(list(COMPOSITES)))
@click.option('--model', 'model_name', type=click.Choice(list(MODELS)), default='resnet18')
@click.option('--parameters', type=click.Path(dir_okay=False))
@click.option(
    '--inputs',
    'input_format',
    type=click.Path(dir_okay=False, writable=True),
    help='Input image format string.  {sample} is replaced with the sample index.'
)
@click.option('--max-samples', type=int)
@click.option('--n-outputs', type=int, default=1)
@click.option('--cpu/--gpu', default=True)
@click.option('--relevance-norm', type=click.Choice(['symmetric', 'absolute', 'unaligned']), default='symmetric')
@click.option('--cmap', type=click.Choice(list(CMAPS)), default='coldnhot')
@click.option('--level', type=float, default=1.0)
def main(
        relevance_format,
        attributor_name,
        composite_name,
        model_name,
        parameters,
        input_format,
        max_samples,
        n_outputs,
        cpu,
        cmap,
        level,
        relevance_norm,
):
    '''Generate heatmaps of an image folder at DATASET_ROOT to files RELEVANCE_FORMAT.
    RELEVANCE_FORMAT is a format string, for which {sample} is replaced with the sample index.
    '''
    # Load a configuration file
    configuration = Configuration()

    random.seed(configuration.seed)
    np.random.seed(configuration.seed)
    torch.manual_seed(configuration.seed)
    cudnn.deterministic = True

    # use the gpu if requested and available, else use the cpu
    device = torch.device('cuda:0' if torch.cuda.is_available() and not cpu else 'cpu')

    # use a resnet backbone
    resnet = torchvision.models.efficientnet_b4()
    resnet = nn.Sequential(*list(resnet.children())[:-1])

    model = LinearEval(resnet, n_outputs, configuration.nnclr_conf.is_binary_label)
    model.load(configuration.nnclr_conf.checkpoint_le, configuration.device)
    model.to(configuration.device)
    model.eval()

    # create a data loader
    data_loader = DataLoaderSSL(configuration).eval_loader
    data_loader.mode = "test"

    # disable requires_grad for all parameters, we do not need their modified gradients
    for param in model.parameters():
        param.requires_grad = False

    # convenience identity matrix to produce one-hot encodings
    eye = torch.eye(n_outputs, device=device)

    # create a composite if composite_name was set, otherwise we do not use a composite
    composite = None
    if composite_name is not None:
        composite_kwargs = {}
        if composite_name == 'epsilon_gamma_box':
            # the maximal input shape, needed for the ZBox rule
            shape = (configuration.nnclr_conf.batch_size, 3, 169, 179)

            # the highest and lowest pixel values for the ZBox rule
            composite_kwargs['low'] = torch.zeros(*shape, device=device)
            composite_kwargs['high'] = torch.ones(*shape, device=device)

        # use torchvision specific canonizers, as supplied in the MODELS dict
        composite_kwargs['canonizers'] = [MODELS[model_name][1]()]

        # create a composite specified by a name; the COMPOSITES dict includes all preset composites provided by zennit.
        composite = COMPOSITES[composite_name](**composite_kwargs)

    # specify some attributor-specific arguments
    attributor_kwargs = {
        'smoothgrad': {'noise_level': 0.1, 'n_iter': 20},
        'integrads': {'n_iter': 20},
        'occlusion': {'window': (56, 56), 'stride': (28, 28)},
    }.get(attributor_name, {})

    # create an attributor, given the ATTRIBUTORS dict given above. If composite is None, the gradient will not be
    # modified for the attribution
    attributor = ATTRIBUTORS[attributor_name](model, composite, **attributor_kwargs)

    # the current sample index for creating file names
    sample_index = 0

    # the accuracy
    accuracy = 0.

    # enter the attributor context outside the data loader loop, such that its canonizers and hooks do not need to be
    # registered and removed for each step. This registers the composite (and applies the canonizer) to the model
    # within the with-statement
    with attributor:
        for idx, (view_one, view_two, target) in enumerate(data_loader):
            # we use data without the normalization applied for visualization, and with the normalization applied as
            # the model input
            view_one = view_one.to(configuration.device)

            for idx in range(int(view_one.shape[0])):
                d = view_one[idx, 0, :, :].cpu().numpy()
                img_data = (((d - d.min()) / (d.max() - d.min())) * 255.9).astype(np.uint8)
                img = Image.fromarray(img_data)
                img.save(
                    "/home/gryshchukv/Projects/clinic-net/zennit-example/results/sample_{}_target{}.png".format(idx,
                                                                                                                target[
                                                                                                                    idx].item()))

                file = nibabel.Nifti1Image(view_one[idx, :, :, :].cpu().numpy(), None)
                nibabel.save(file,
                             "/home/gryshchukv/Projects/clinic-net/zennit-example/results/sample_{}_target_{}".format(
                                 idx, target[idx].item()))

            # one-hot encoding of the target labels of size (len(target), 1000)
            output_relevance = torch.ones(len(target)).unsqueeze(1)

            # this will compute the modified gradient of model, with the on
            output, relevance = attributor(view_one, output_relevance)

            # sum over the color channel for visualization
            relevance = np.array(relevance.sum(1).detach().cpu())

            # normalize between 0. and 1. given the specified strategy
            if relevance_norm == 'symmetric':
                # 0-aligned symmetric relevance, negative and positive can be compared, the original 0. becomes 0.5
                amax = np.abs(relevance).max((1, 2), keepdims=True)
                relevance = (relevance + amax) / 2 / amax
            elif relevance_norm == 'absolute':
                # 0-aligned absolute relevance, only the amplitude of relevance matters, the original 0. becomes 0.
                relevance = np.abs(relevance)
                relevance /= relevance.max((1, 2), keepdims=True)
            elif relevance_norm == 'unaligned':
                # do not align, the original minimum value becomes 0., the original maximum becomes 1.
                rmin = relevance.min((1, 2), keepdims=True)
                rmax = relevance.max((1, 2), keepdims=True)
                relevance = (relevance - rmin) / (rmax - rmin)

            for n in range(len(view_one)):
                fname = relevance_format.format(sample=sample_index + n)
                # zennit.image.imsave will create an appropriate heatmap given a cmap specification
                imsave(fname, relevance[n], vmin=0., vmax=1., level=level, cmap=cmap)
                if input_format is not None:
                    fname = input_format.format(sample=sample_index + n)
                    # if there are 3 color channels, imsave will not create a heatmap, but instead save the image with
                    # its appropriate colors
                    imsave(fname, view_one[n])
            sample_index += len(view_one)


if __name__ == '__main__':
    main()
