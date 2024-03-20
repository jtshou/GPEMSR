import torch
import torch.nn as nn
import math
import basicsr.archs.arch_util as arch_util
import functools
import torch.nn.functional as F
from basicsr.archs.arch_util import DCNv2Pack,ResidualBlockNoBN
from basicsr.archs.spynet_arch import SpyNet
from model.VGG import VGG19
from model.vqgan_indexer import lrGenerator16,lrGenerator8
from torchvision import models


def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows-1)*strides[0]+effective_k_row-rows)
    padding_cols = max(0, (out_cols-1)*strides[1]+effective_k_col-cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ReflectionPad2d(paddings)(images)
    return images

def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()

    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks



class POD(nn.Module):
    def __init__(self,spynet_path = '/GPEMSR-CREMI/GPEMSR/pre-train_model/spynet_sintel_final-3d2a1287.pth',nf=64, groups=8):
        super(POD, self).__init__()
        self.spynet = SpyNet(spynet_path)
        for k,v in self.spynet.named_parameters():
            v.requires_grad = False
        self.flowdsconv0_1 = nn.Conv2d(2, 16, 3, 4, 1, bias=True)
        self.flowdsconv0_2 = nn.Conv2d(2, 16, 3, 4, 1, bias=True)
        self.flowdsconv1_1 = nn.Conv2d(16,16, 3, 2, 1, bias=True)
        self.flowdsconv1_2 = nn.Conv2d(16, 16, 3, 2, 1, bias=True)
        self.flowdsconv2_1 = nn.Conv2d(16, 16, 3, 2, 1, bias=True)
        self.flowdsconv2_2 = nn.Conv2d(16, 16, 3, 2, 1, bias=True)

        self.L3_offset_conv1 = nn.Conv2d(nf * 2+34, nf, 3, 1, 1, bias=True)
        self.L3_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack = DCNv2Pack(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        self.L2_offset_conv1 = nn.Conv2d(nf * 2+34, nf, 3, 1, 1, bias=True)
        self.L2_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.L2_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack = DCNv2Pack(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        self.L2_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.L1_offset_conv1 = nn.Conv2d(nf * 2+34, nf, 3, 1, 1, bias=True)
        self.L1_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.L1_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack = DCNv2Pack(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        self.L1_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        # Cascading DCN
        self.cas_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.cas_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.cas_dcnpack = DCNv2Pack(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_fea_l, ref_fea_l,neighbor_frame,ref_frame):
        L1_flow1 = self.spynet(F.interpolate(neighbor_frame, scale_factor=4, mode='bilinear', align_corners=False),F.interpolate(ref_frame, scale_factor=4, mode='bilinear', align_corners=False))#B,2,2*H,2*W
        L1_flow2 = self.spynet(F.interpolate(neighbor_frame, scale_factor=4, mode='bilinear', align_corners=False),F.interpolate(ref_frame, scale_factor=4, mode='bilinear', align_corners=False))#B,2,2*H,2*W
        L1_flow1 = self.flowdsconv0_1(L1_flow1)#B,16,H,W
        L1_flow2 = self.flowdsconv0_2(L1_flow2)#B,16,H,W
        L2_flow1 = self.flowdsconv1_1(L1_flow1)#B,16,H/2,W/2
        L2_flow2 = self.flowdsconv1_2(L1_flow2)#B,16,H/2,W/2
        L3_flow1 = self.flowdsconv2_1(L2_flow1)#B,16,H/4,W/4
        L3_flow2 = self.flowdsconv2_2(L2_flow2)#B,16,H/4,W/4
        neighbor_frame_L2 = F.interpolate(neighbor_frame, scale_factor=1/2, mode='bilinear', align_corners=False)
        ref_frame_L2 = F.interpolate(ref_frame, scale_factor=1 / 2, mode='bilinear', align_corners=False)
        neighbor_frame_L3 = F.interpolate(neighbor_frame_L2, scale_factor=1 / 2, mode='bilinear', align_corners=False)
        ref_frame_L3 = F.interpolate(ref_frame_L2, scale_factor=1 / 2, mode='bilinear', align_corners=False)

        L3_offset = torch.cat([nbr_fea_l[2], ref_fea_l[2],L3_flow1,L3_flow2,neighbor_frame_L3,ref_frame_L3], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2(L3_offset))
        L3_fea = self.lrelu(self.L3_dcnpack(nbr_fea_l[2], L3_offset))

        L2_offset = torch.cat([nbr_fea_l[1], ref_fea_l[1],L2_flow1,L2_flow2,neighbor_frame_L2,ref_frame_L2], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1(L2_offset))
        L3_offset = F.interpolate(L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L2_offset = self.lrelu(self.L2_offset_conv2(torch.cat([L2_offset, L3_offset * 2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3(L2_offset))
        L2_fea = self.L2_dcnpack(nbr_fea_l[1], L2_offset)
        L3_fea = F.interpolate(L3_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L2_fea = self.lrelu(self.L2_fea_conv(torch.cat([L2_fea, L3_fea], dim=1)))

        L1_offset = torch.cat([nbr_fea_l[0], ref_fea_l[0],L1_flow1,L1_flow2,neighbor_frame,ref_frame], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1(L1_offset))
        L2_offset = F.interpolate(L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L1_offset = self.lrelu(self.L1_offset_conv2(torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3(L1_offset))
        L1_fea = self.L1_dcnpack(nbr_fea_l[0], L1_offset)
        L2_fea = F.interpolate(L2_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L1_fea = self.L1_fea_conv(torch.cat([L1_fea, L2_fea], dim=1))

        offset = torch.cat([L1_fea, ref_fea_l[0]], dim=1)
        offset = self.lrelu(self.cas_offset_conv1(offset))
        offset = self.lrelu(self.cas_offset_conv2(offset))
        L1_fea = self.lrelu(self.cas_dcnpack(L1_fea, offset))

        return L1_fea


class ThreeDA(nn.Module):
    def __init__(self, num_feat=64, num_frame=5, center_frame_idx=2):
        super(ThreeDA, self).__init__()
        self.center_frame_idx = center_frame_idx
        self.temporal_attn1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.temporal_attn2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.feat_fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)
        self.conv3D_1 = nn.Conv3d(num_frame, num_frame, kernel_size=1, bias=True)
        self.conv3D_2 = nn.Conv3d(num_frame, num_frame, kernel_size=1, bias=True)
        self.conv3D_fusion_1 = nn.Conv2d(num_frame * num_feat, num_feat, kernel_size=1, bias=True)
        self.conv3D_fusion_2 = nn.Conv2d(num_frame * num_feat, num_feat, kernel_size=1, bias=True)
        self.conv2D_fusion_3 = nn.Conv2d(num_feat, num_feat, kernel_size=1, bias=True)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.spatial_attn1 = nn.Conv2d(num_frame * num_feat, num_feat, 1)
        self.spatial_attn2 = nn.Conv2d(num_feat * 2, num_feat, 1)
        self.spatial_attn3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn4 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn5 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_l1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_l2 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.spatial_attn_l3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_add1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_add2 = nn.Conv2d(num_feat, num_feat, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, aligned_feat):
        b, t, c, h, w = aligned_feat.size()
        embedding_ref = self.temporal_attn1(
            aligned_feat[:, self.center_frame_idx, :, :, :].clone())
        embedding = self.temporal_attn2(aligned_feat.view(-1, c, h, w))
        embedding = embedding.view(b, t, -1, h, w)  # (b, t, c, h, w)

        corr_l = []  # correlation list
        for i in range(t):
            emb_neighbor = embedding[:, i, :, :, :]
            corr = torch.sum(emb_neighbor * embedding_ref, 1)  # (b, h, w)
            corr_l.append(corr.unsqueeze(1))  # (b, 1, h, w)
        corr_prob = torch.sigmoid(torch.cat(corr_l, dim=1))  # (b, t, h, w)
        corr_prob = corr_prob.unsqueeze(2).expand(b, t, c, h, w)
        corr_prob = corr_prob.contiguous().view(b, -1, h, w)  # (b, t*c, h, w)
        aligned_feat = aligned_feat.view(b, -1, h, w) * corr_prob

        # fusion
        feat = self.lrelu(self.feat_fusion(aligned_feat))
        fea_3d1 = self.lrelu(self.conv3D_1(aligned_feat.view(b, t, -1, h, w)))
        fea_3d1 = self.lrelu(self.conv3D_fusion_1(fea_3d1.view(b, -1, h, w)))
        fea_3d2 = self.lrelu(self.conv3D_2(aligned_feat.view(b, t, -1, h, w)))
        fea_3d2 = self.lrelu(self.conv3D_fusion_2(fea_3d2.view(b, -1, h, w)))

        feat = feat + fea_3d1
        fea_3d3 = self.conv2D_fusion_3(feat)

        # spatial attention
        attn = self.lrelu(self.spatial_attn1(aligned_feat))
        attn_max = self.max_pool(attn)
        attn_avg = self.avg_pool(attn)
        attn = self.lrelu(
            self.spatial_attn2(torch.cat([attn_max, attn_avg], dim=1)))
        attn_level = self.lrelu(self.spatial_attn_l1(attn))
        attn_max = self.max_pool(attn_level)
        attn_avg = self.avg_pool(attn_level)
        attn_level = self.lrelu(
            self.spatial_attn_l2(torch.cat([attn_max, attn_avg], dim=1)))
        attn_level = self.lrelu(self.spatial_attn_l3(attn_level))
        attn_level = self.upsample(attn_level)

        attn = self.lrelu(self.spatial_attn3(attn)) + attn_level
        attn = self.lrelu(self.spatial_attn4(attn))
        attn = self.upsample(attn)
        attn = self.spatial_attn5(attn)
        attn_add = self.spatial_attn_add2(
            self.lrelu(self.spatial_attn_add1(attn)))
        attn = torch.sigmoid(attn)

        feat = feat * attn * 2 + attn_add + fea_3d2 + fea_3d3
        return feat


class GPEMSR(nn.Module):
    def __init__(self,ref_path_G,ref_path_Indexer,argref,nf=64, nframes=5, groups=8, front_RBs=5, back_RBs=10,w_ref=True,
                 ref_fusion_feat_RBs=3,align_mode='POD',fusion_mode='ThreeDA',
                 mode='16to1',scale=16):

        super(GPEMSR, self).__init__()
        self.nf = nf
        self.center = nframes // 2
        self.scale = scale
        self.w_ref = w_ref
        self.align_mode = align_mode
        self.fusion_mode = fusion_mode
        self.mode = mode

        ResidualBlock_noBN_f = functools.partial(ResidualBlockNoBN, num_feat=nf)

        #### extract features (for each frame)
        self.conv_first = nn.Conv2d(1, nf, 3, 1, 1, bias=True)
        self.feature_extraction = arch_util.make_layer(ResidualBlock_noBN_f, front_RBs)

        # ref
        if self.w_ref:
            self.vgg = VGG19()
            self.refmaskconv1 = nn.Conv2d(1,nf,3,1,1)
            self.refmaskconv2 = nn.Conv2d(nf,nf,3,1,1)
            self.refmaskconv3 = nn.Conv2d(nf,1,3,1,1)

            self.reffea_L2_conv1 = nn.ConvTranspose2d(nf, nf, 3, 2, 1, 1, bias=True)
            self.reffea_L3_conv1 = nn.ConvTranspose2d(nf, nf, 3, 2, 1, 1, bias=True)
            self.reffea_L4_conv1 = nn.ConvTranspose2d(nf, nf, 3, 2, 1, 1, bias=True)
            self.reffusionconv1 = nn.Conv2d(nf + 64, nf, 3, 1, 1)
            self.fusion_fea_block1 = arch_util.make_layer(ResidualBlock_noBN_f, ref_fusion_feat_RBs)
            self.down_fea_conv1 = nn.Conv2d(nf, nf, 3, 2, 1)
            self.reffusionconv2 = nn.Conv2d(2 * nf + 128, nf, 3, 1, 1)
            self.fusion_fea_block2 = arch_util.make_layer(ResidualBlock_noBN_f, ref_fusion_feat_RBs)
            self.down_fea_conv2 = nn.Conv2d(nf * 2, nf * 2, 3, 2, 1)
            self.reffusionconv3 = nn.Conv2d(3 * nf + 256, nf, 3, 1, 1)
            self.fusion_fea_block3 = arch_util.make_layer(ResidualBlock_noBN_f, ref_fusion_feat_RBs)
            self.down_fea_conv3 = nn.Conv2d(nf * 3, nf * 3, 3, 2, 1)
            self.reffusionconv4 = nn.Conv2d(4 * nf + 512, nf, 3, 1, 1)
            self.fusion_fea_block4 = arch_util.make_layer(ResidualBlock_noBN_f, ref_fusion_feat_RBs)
            if self.scale == 16:
                self.reduce_dim_conv = nn.Conv2d(5 * nf, nf, 1, 1, 0)
            elif self.scale == 8:
                self.reduce_dim_conv = nn.Conv2d(4 * nf, nf, 1, 1, 0)

            if self.scale == 16:
                self.refmodel = lrGenerator16(argref)
                for k,v in self.refmodel.named_parameters():
                    v.requires_grad = False
                self.refmodel.load_state_dict(torch.load(ref_path_G), strict=False)
                self.refmodel.indexer.load_state_dict(torch.load(ref_path_Indexer),strict=True)


            elif self.scale == 8:
                self.refmodel = lrGenerator8(argref)
                for k, v in self.refmodel.named_parameters():
                    v.requires_grad = False
                self.refmodel.load_state_dict(torch.load(ref_path_G), strict=False)
                self.refmodel.indexer.load_state_dict(torch.load(ref_path_Indexer),strict=True)

            else:
                raise ValueError('scale is wrong!')


        #align
        if self.align_mode== 'POD':
            self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
            self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
            self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.align_module = POD(nf=nf, groups=groups)

        if self.fusion_mode == 'ThreeDA':
            self.ThreeDA = ThreeDA(num_feat=nf, num_frame=nframes, center_frame_idx=self.center)

        #### reconstruction
        self.recon_trunk = arch_util.make_layer(ResidualBlock_noBN_f, back_RBs)
        #### upsampling
        if self.mode == '16to1':
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(nf, 64 * 4, 3, 1, 1, bias=True)
            self.upconv3 = nn.Conv2d(64, 64*4, 3, 1, 1, bias=True)
            self.upconv4 = nn.Conv2d(64, 64*4, 3, 1, 1, bias=True)

        if self.mode == '8to1':
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(nf, 64 * 4, 3, 1, 1, bias=True)
            self.upconv3 = nn.Conv2d(64, 64 * 4, 3, 1, 1, bias=True)


        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 1, 3, 1, 1, bias=True)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        B, N, C, H, W = x.size()
        x_center = x[:, self.center, :, :, :].contiguous()

        #### extract LR features
        # L1
        L1_fea = self.lrelu(self.conv_first(x.view(-1, C, H, W)))
        L1_fea = self.feature_extraction(L1_fea)#[B*N,C,H,W]

        #### calculate ref_feat
        if self.w_ref:
            if self.mode == '16to1':
                # L2
                L2_fea = self.lrelu(self.reffea_L2_conv1(L1_fea))
                # L3
                L3_fea = self.lrelu(self.reffea_L3_conv1(L2_fea))
                # L4
                L4_fea = self.lrelu(self.reffea_L4_conv1(L3_fea))

                ref_x16, ref_x8, ref_x4, ref_x2, ref_img = self.refmodel.ref_extract(
                    x.view(-1, C, H, W))
                up_lr = F.interpolate(x.view(-1, C, H, W), scale_factor=16, mode='bilinear', align_corners=False)
                ds_ref_feat = getattr(self.vgg(ref_img.expand(-1, 3, -1, -1)), 'relu1_2')
                ds_ref_feat = extract_image_patches(ds_ref_feat, ksizes=[16, 16], strides=[16, 16],
                                                    rates=[1, 1], padding='same')
                ds_ref_feat = F.normalize(ds_ref_feat, dim=1)
                lr_feat = getattr(self.vgg(up_lr.expand(-1, 3, -1, -1)), 'relu1_2')
                lr_feat = extract_image_patches(lr_feat, ksizes=[16, 16], strides=[16, 16],
                                                rates=[1, 1], padding='same')
                lr_feat = F.normalize(lr_feat, dim=1)
                mask = torch.sum(ds_ref_feat.contiguous() * lr_feat.contiguous(), dim=1, keepdim=True)  # B,1,L
                mask = mask.view(B * N, 1, H, W)
                mask = self.lrelu(self.refmaskconv1(mask))
                mask = self.lrelu(self.refmaskconv2(mask))
                mask = self.lrelu(self.refmaskconv3(mask))
                mask = torch.sigmoid(mask)  # [B*N,1,H,W]

                # fusion
                ref_x2 = self.reffusionconv1(torch.cat((L4_fea, ref_x2), dim=1))
                ref_x2 = self.fusion_fea_block1(ref_x2) * F.interpolate(mask, scale_factor=8, mode='bilinear',
                                                                        align_corners=False)
                ref_x2 = self.down_fea_conv1(ref_x2)
                ref_x4 = self.reffusionconv2(torch.cat((L3_fea, ref_x4, ref_x2), dim=1))
                ref_x4 = self.fusion_fea_block2(ref_x4) * F.interpolate(mask, scale_factor=4, mode='bilinear',
                                                                        align_corners=False)
                ref_x4 = self.down_fea_conv2(torch.cat((ref_x4, ref_x2), dim=1))
                ref_x8 = self.reffusionconv3(torch.cat((L2_fea, ref_x8, ref_x4), dim=1))
                ref_x8 = self.fusion_fea_block3(ref_x8) * F.interpolate(mask, scale_factor=2, mode='bilinear',
                                                                        align_corners=False)
                ref_x8 = self.down_fea_conv3(torch.cat((ref_x8, ref_x4), dim=1))
                ref_x16 = self.reffusionconv4(torch.cat((L1_fea, ref_x16, ref_x8), dim=1))
                ref_x16 = self.fusion_fea_block4(ref_x16) * mask
                L1_fea = torch.cat((ref_x16, ref_x8, L1_fea), dim=1)
                L1_fea = self.reduce_dim_conv(L1_fea)

            elif self.mode == '8to1':
                # L2
                L2_fea = self.lrelu(self.reffea_L2_conv1(L1_fea))
                # L3
                L3_fea = self.lrelu(self.reffea_L3_conv1(L2_fea))
                ref_x16, ref_x8, ref_x4, ref_x2, ref_img = self.refmodel.ref_extract(
                    x.view(-1, C, H, W))
                up_lr = F.interpolate(x.view(-1, C, H, W), scale_factor=8, mode='bilinear', align_corners=False)
                ds_ref_feat = getattr(self.vgg(ref_img.expand(-1, 3, -1, -1)), 'relu1_2')
                ds_ref_feat = extract_image_patches(ds_ref_feat, ksizes=[16, 16], strides=[16, 16],
                                                    rates=[1, 1], padding='same')
                ds_ref_feat = F.normalize(ds_ref_feat, dim=1)
                lr_feat = getattr(self.vgg(up_lr.expand(-1, 3, -1, -1)), 'relu1_2')
                lr_feat = extract_image_patches(lr_feat, ksizes=[16, 16], strides=[16, 16],
                                                rates=[1, 1], padding='same')
                lr_feat = F.normalize(lr_feat, dim=1)
                mask = torch.sum(ds_ref_feat.contiguous() * lr_feat.contiguous(), dim=1, keepdim=True)
                mask = mask.view(B * N, 1, H//2, W//2)
                mask = self.lrelu(self.refmaskconv1(mask))
                mask = self.lrelu(self.refmaskconv2(mask))
                mask = self.lrelu(self.refmaskconv3(mask))
                mask = torch.sigmoid(mask)

                # fusion
                ref_x2 = self.reffusionconv1(torch.cat((L3_fea, ref_x2), dim=1))
                ref_x2 = self.fusion_fea_block1(ref_x2) * F.interpolate(mask, scale_factor=8, mode='bilinear',
                                                                        align_corners=False)
                ref_x2 = self.down_fea_conv1(ref_x2)
                ref_x4 = self.reffusionconv2(torch.cat((L2_fea, ref_x4, ref_x2), dim=1))
                ref_x4 = self.fusion_fea_block2(ref_x4) * F.interpolate(mask, scale_factor=4, mode='bilinear',
                                                                        align_corners=False)
                ref_x4 = self.down_fea_conv2(torch.cat((ref_x4, ref_x2), dim=1))
                ref_x8 = self.reffusionconv3(torch.cat((L1_fea, ref_x8, ref_x4), dim=1))
                ref_x8 = self.fusion_fea_block3(ref_x8) * F.interpolate(mask, scale_factor=2, mode='bilinear',
                                                                        align_corners=False)

                L1_fea = torch.cat((ref_x8, ref_x4, L1_fea), dim=1)
                L1_fea = self.reduce_dim_conv(L1_fea)
        # align
        if self.align_mode == 'POD':
            # L2
            L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))
            L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))
            # L3
            L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea))
            L3_fea = self.lrelu(self.fea_L3_conv2(L3_fea))
            L1_fea = L1_fea.view(B, N, -1, H, W)
            L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2)
            L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4)
            ref_fea_l = [
                L1_fea[:, self.center, :, :, :].clone(), L2_fea[:, self.center, :, :, :].clone(),
                L3_fea[:, self.center, :, :, :].clone()
            ]
            aligned_fea = []
            for i in range(N):
                nbr_fea_l = [
                    L1_fea[:, i, :, :, :].clone(), L2_fea[:, i, :, :, :].clone(),
                    L3_fea[:, i, :, :, :].clone()
                ]
                aligned_fea.append(self.align_module(nbr_fea_l, ref_fea_l,x[:,i, :, :, :],x_center))
            aligned_fea = torch.stack(aligned_fea, dim=1)  # [B, N, C, H, W]
        if self.fusion_mode == 'ThreeDA':
            fea = self.ThreeDA(aligned_fea)
        out = self.recon_trunk(fea)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        if self.mode == '16to1':
            out = self.lrelu(self.pixel_shuffle(self.upconv3(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv4(out)))
        if self.mode == '8to1':
            out = self.lrelu(self.pixel_shuffle(self.upconv3(out)))
        out = self.lrelu(self.HRconv(out))
        out = self.conv_last(out)
        if self.mode == '16to1':
            base = F.interpolate(x_center, scale_factor=16, mode='bilinear', align_corners=False)
        elif self.mode == '8to1':
            base = F.interpolate(x_center, scale_factor=8, mode='bilinear', align_corners=False)
        out += base
        return out,ref_img.view(B,N,C,H*self.scale,W*self.scale)



