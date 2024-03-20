import torch
import torch.nn as nn
from model.blocks import ResidualBlock,DownBlock,NonLocalBlock,UpBlock


class Indexer16(nn.Module):
    def __init__(self, args):
        super(Indexer16, self).__init__()
        self.args = args
        self.channel_list = args['channel_list']
        self.input_layer = nn.Sequential(nn.Conv2d(args['im_channel'],self.channel_list[0],3,1,1),
                                         nn.ReLU(inplace=True))


        self.num_res_blocks = args['num_resblock_per_scale']
        self.num_output_resblck = args['num_output_resblck']
        self.latent_dim = args['latent_dim']
        self.use_non_local = args['use_non_local']


        layers = []
        for i in range(len(self.channel_list)-1):
            in_channels = self.channel_list[i]
            out_channels = self.channel_list[i + 1]
            for j in range(self.num_res_blocks-1):
                layers.append(ResidualBlock(in_channels, in_channels))
            if i==4:
                layers.append(DownBlock(in_channels, out_channels))
            else:
                layers.append(ResidualBlock(in_channels, out_channels))
        if len(self.channel_list) == 4:
            for j in range(self.num_res_blocks-1):
                layers.append(ResidualBlock(self.channel_list[-1], self.channel_list[-1]))
            layers.append(UpBlock(self.channel_list[-1], self.channel_list[-1]))
        if self.use_non_local:
            layers.append(NonLocalBlock(self.channel_list[-1]))
        self.feat_extract = nn.Sequential(*layers)


        layers = []
        for i in range(self.num_output_resblck):
            layers.append(ResidualBlock(self.channel_list[-1],self.channel_list[-1]))
        layers.append(nn.Conv2d(self.channel_list[-1],self.latent_dim,1))
        self.output_layer = nn.Sequential(*layers)


        self.embedding = nn.Linear(self.latent_dim,1024)



    def forward(self, x):
        feat = self.output_layer(self.feat_extract(self.input_layer(x)))
        feat_p = self.embedding(feat.permute(0,2,3,1))

        return feat_p


class Indexer8(nn.Module):
    def __init__(self, args):
        super(Indexer8, self).__init__()
        self.args = args
        self.channel_list = args['channel_list']
        self.input_layer = nn.Sequential(nn.Conv2d(args['im_channel'],self.channel_list[0],3,1,1),
                                         nn.ReLU(inplace=True))


        self.num_res_blocks = args['num_resblock_per_scale']
        self.num_output_resblck = args['num_output_resblck']
        self.latent_dim = args['latent_dim']
        self.use_non_local = args['use_non_local']

        layers = []
        for i in range(len(self.channel_list) - 1):
            in_channels = self.channel_list[i]
            out_channels = self.channel_list[i + 1]
            for j in range(self.num_res_blocks - 1):
                layers.append(ResidualBlock(in_channels, in_channels))
            if i == 3:
                layers.append(DownBlock(in_channels, out_channels))
            else:
                layers.append(ResidualBlock(in_channels, out_channels))


        if self.use_non_local:
            layers.append(NonLocalBlock(self.channel_list[-1]))
        self.feat_extract = nn.Sequential(*layers)


        layers = []
        for i in range(self.num_output_resblck):
            layers.append(ResidualBlock(self.channel_list[-1],self.channel_list[-1]))
        layers.append(nn.Conv2d(self.channel_list[-1],self.latent_dim,1))
        self.output_layer = nn.Sequential(*layers)

        # classification
        self.embedding = nn.Linear(self.latent_dim, 1024)

    def forward(self, x):
        feat = self.output_layer(self.feat_extract(self.input_layer(x)))
        feat_p = self.embedding(feat.permute(0, 2, 3, 1))

        return feat_p


