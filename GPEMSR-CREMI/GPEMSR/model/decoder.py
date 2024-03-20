import torch
import torch.nn as nn
from model.blocks import ResidualBlock,UpBlock,NonLocalBlock


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.args = args
        self.channel_list = args['channel_list']
        self.num_res_blocks = args['num_resblock_per_scale']
        self.num_input_resblck = args['num_input_resblck']
        self.latent_dim = args['latent_dim']
        self.use_non_local = args['use_non_local']

        layers = []
        layers.append(nn.Conv2d(self.latent_dim,self.channel_list[0],1))
        for i in range(self.num_input_resblck):
            layers.append(ResidualBlock(self.channel_list[0], self.channel_list[0]))
        self.input_layer = nn.Sequential(*layers)

        layers = []
        if self.use_non_local:
            layers.append(NonLocalBlock(self.channel_list[0]))
        for i in range(len(self.channel_list)-1):
            in_channels = self.channel_list[i]
            out_channels = self.channel_list[i + 1]
            for j in range(self.num_res_blocks):
                layers.append(ResidualBlock(in_channels, in_channels))
            layers.append(UpBlock(in_channels,out_channels))
        self.feat_extract = nn.Sequential(*layers)

        self.output_layer = nn.Conv2d(self.channel_list[-1],args['im_channel'], 3, 1, 1)



    def forward(self, x):
        return self.output_layer(self.feat_extract(self.input_layer(x)))

    def multi_scale_feat_calculate(self,x):
        feat_l = []
        x = self.input_layer(x)
        if self.use_non_local:
            x = self.feat_extract[0](x)
            for i in range(len(self.feat_extract)-1):
                x = self.feat_extract[i+1](x)
                if (i - self.num_res_blocks + 1) % (self.num_res_blocks + 1) == 0:
                    feat_l.append(x)
        else:
            for i in range(len(self.feat_extract)):
                x = self.feat_extract[i](x)
                if (i - self.num_res_blocks + 1) % (self.num_res_blocks + 1) == 0:
                    feat_l.append(x)
        feat_l.append(self.output_layer((x)))


        return feat_l

