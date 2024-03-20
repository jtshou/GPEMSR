"""
PatchGAN Discriminator (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py#L538)
"""

import torch.nn as nn
import torch


class Discriminator(nn.Module):
    def __init__(self,args):
        super(Discriminator, self).__init__()
        im_channel = args['im_channel']
        num_filters_last = args['num_filters_last']
        n_layers = args['n_layers']
        layers = [nn.Conv2d(im_channel, num_filters_last, 4, 2, 0), nn.LeakyReLU(0.2)]
        num_filters_mult = 1

        for i in range(1, n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2 ** i, 8)
            layers += [
                nn.Conv2d(num_filters_last * num_filters_mult_last, num_filters_last * num_filters_mult, 4,
                          2 if i < n_layers else 1, 0, bias=False),
                nn.InstanceNorm2d(num_filters_last * num_filters_mult),
                nn.LeakyReLU(0.2, True)
            ]

        layers.append(nn.Conv2d(num_filters_last * num_filters_mult, 1, 4, 1, 0))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
