import torch
import torch.nn as nn
from model.indexer import Indexer16, Indexer8
from model.encoder import Encoder
from model.decoder import Decoder
from model.codebook import Codebook
from model.discriminator import Discriminator



class VQGAN_Indexer16(nn.Module):
    def __init__(self, args):
        super(VQGAN_Indexer16, self).__init__()
        self.lrgenerator16 = lrGenerator16(args['lrGenerator16'])
        self.discriminator = Discriminator(args['Discriminator'])



class lrGenerator16(nn.Module):
    def __init__(self, args):
        super(lrGenerator16, self).__init__()
        self.indexer = Indexer16(args['Indexer16'])
        self.decoder = Decoder(args['Decoder'])
        self.codebook = Codebook(args['Codebook'])
        self.encoder = Encoder(args['Encoder'])

    def output_ref(self, imgs):
        # input: lr_img
        logits = self.indexer(imgs)
        codebook_mapping = self.codebook.inference_lr(logits)
        decoded_images = self.decoder(codebook_mapping)
        return decoded_images


    def forward(self, lr, gt):
        logits = self.indexer(lr)
        gtencoded_feat = self.encoder(gt)
        gtcodebook_mapping, gtcodebook_indices, gtq_loss = self.codebook(gtencoded_feat)
        B, H, W, C = logits.shape
        logits = logits.view(B * H * W, -1).contiguous()
        return logits, gtcodebook_indices  # [B*H*W,1024];[B*H*W]


    def ref_extract(self,imgs):
        logits = self.indexer(imgs)
        codebook_mapping = self.codebook.inference_lr(logits)
        feat_l = self.decoder.multi_scale_feat_calculate(codebook_mapping)
        return feat_l



class VQGAN_Indexer8(nn.Module):
    def __init__(self, args):
        super(VQGAN_Indexer8, self).__init__()
        self.lrgenerator8 = lrGenerator8(args['lrGenerator8'])
        self.discriminator = Discriminator(args['Discriminator'])



class lrGenerator8(nn.Module):
    def __init__(self, args):
        super(lrGenerator8, self).__init__()
        self.indexer = Indexer8(args['Indexer8'])
        self.decoder = Decoder(args['Decoder'])
        self.codebook = Codebook(args['Codebook'])
        self.encoder = Encoder(args['Encoder'])


    def output_ref(self, imgs):
        # input: lr_img
        logits = self.indexer(imgs)
        codebook_mapping = self.codebook.inference_lr(logits)
        decoded_images = self.decoder(codebook_mapping)
        return decoded_images


    def forward(self, lr, gt):
        # input: lr_img,gt_img
        logits = self.indexer(lr)
        gtencoded_feat = self.encoder(gt)
        gtcodebook_mapping, gtcodebook_indices, gtq_loss = self.codebook(gtencoded_feat)
        B,H,W,C = logits.shape
        logits = logits.view(B*H*W,-1).contiguous()
        return logits, gtcodebook_indices#[B*H*W,1024];[B*H*W]


    def ref_extract(self,imgs):
        logits = self.indexer(imgs)
        codebook_mapping = self.codebook.inference_lr(logits)
        feat_l = self.decoder.multi_scale_feat_calculate(codebook_mapping)
        return feat_l

