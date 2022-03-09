import torchvision
from torch import nn

from models.nnclr.nnclr import NNCLR
from visual.filter import Filter

backbone = torchvision.models.efficientnet_b0()
backbone = nn.Sequential(*list(backbone.children())[:-1])
backbone = NNCLR.load_state_dict_(backbone,
                                  '/data_dzne_archiv/Studien/ClinicNET/clinic-net-git/output/checkpoints/nnclr_300.ckpt')

visual_filter = Filter(backbone)
visual_filter.set_conv_weights()
visual_filter.visualize_filters(ch=0, allkernels=False)
