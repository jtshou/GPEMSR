import torch
import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder
from model.codebook import Codebook
from model.discriminator import Discriminator


class VQGAN(nn.Module):
    def __init__(self, args):
        super(VQGAN, self).__init__()
        self.generator = Generator(args['Generator'])
        self.discriminator = Discriminator(args['Discriminator'])


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.encoder = Encoder(args['Encoder'])
        self.decoder = Decoder(args['Decoder'])
        self.codebook = Codebook(args['Codebook'])

    def forward(self, imgs):
        encoded_images = self.encoder(imgs)
        codebook_mapping, codebook_indices, q_loss = self.codebook(encoded_images)
        decoded_images = self.decoder(codebook_mapping)
        return decoded_images, codebook_indices, q_loss
