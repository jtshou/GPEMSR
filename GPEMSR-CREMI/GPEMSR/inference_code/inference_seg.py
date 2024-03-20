import os
import cv2
import h5py
import yaml
import torch
import argparse
import numpy as np
from skimage import morphology
from attrdict import AttrDict
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm import tqdm
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
from model.unetr import UNETR
from data.provider_valid import Provider_valid
from model.unet3d_mala import UNet3D_MALA
from model.model_superhuman import UNet_PNI
from utils.show import draw_fragments_3d, draw_raw_image
from utils.shift_channels import shift_func
import waterz
from utils.fragment import watershed, randomlabel, relabel
from skimage.metrics import adapted_rand_error as adapted_rand_ref
from skimage.metrics import variation_of_information as voi_ref
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='seg_3d')
    parser.add_argument('-mn', '--model_name', type=str, default=None)
    parser.add_argument('-id', '--model_id', type=str, default='GT')
    parser.add_argument('-m', '--mode', type=str, default='cremiC')
    parser.add_argument('-ts', '--test_split', type=int, default=50)
    parser.add_argument('-pm', '--pixel_metric', action='store_true', default=False)
    parser.add_argument('-s', '--save', action='store_true', default=True)
    parser.add_argument('-sw', '--show', action='store_true', default=True)
    parser.add_argument('-malis', '--malis_loss', action='store_true', default=False)
    parser.add_argument('-waterz', '--waterz', action='store_false', default=True)
    args = parser.parse_args()

    cfg_file = args.cfg
    print('cfg_file: ' + cfg_file)

    with open('/GPEMSR-CREMI/GPEMSR/inference_code/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.safe_load(f))

    if cfg.DATA.shift_channels is None:
        assert cfg.MODEL.output_nc == 3, "output_nc must be 3"
        cfg.shift = None
    else:
        assert cfg.MODEL.output_nc == cfg.DATA.shift_channels, "output_nc must be equal to shift_channels"
        cfg.shift = shift_func(cfg.DATA.shift_channels)

    if args.model_name is not None:
        trained_model = args.model_name
    else:
        trained_model = cfg.TEST.model_name

    #path
    out_path = os.path.join('/GPEMSR-CREMI/GPEMSR/inference_code/Result', trained_model)#output path
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    img_folder = 'affs_'+args.model_id
    out_affs = os.path.join(out_path, img_folder)
    if not os.path.exists(out_affs):
        os.makedirs(out_affs)
    print('out_path: ' + out_affs)
    affs_img_path = os.path.join(out_affs, 'affs_img')
    seg_img_path = os.path.join(out_affs, 'seg_img')
    if not os.path.exists(affs_img_path):
        os.makedirs(affs_img_path)
    if not os.path.exists(seg_img_path):
        os.makedirs(seg_img_path)

    device = torch.device('cuda:0')
    if cfg.MODEL.model_type == 'mala':
        print('load mala model!')
        ckpt_path = os.path.join('/GPEMSR-CREMI/GPEMSR/pre-train_model/MALA.pt')
        model = UNet3D_MALA(output_nc=cfg.MODEL.output_nc, if_sigmoid=cfg.MODEL.if_sigmoid, init_mode=cfg.MODEL.init_mode_mala).to(device)
    elif cfg.MODEL.model_type == 'superhuman':
        print('load superhuman model!')
        ckpt_path = os.path.join('/GPEMSR-CREMI/GPEMSR/pre-train_model/superhuman.pt')
        model = UNet_PNI(in_planes=cfg.MODEL.input_nc,
                        out_planes=cfg.MODEL.output_nc,
                        filters=cfg.MODEL.filters,
                        upsample_mode=cfg.MODEL.upsample_mode,
                        decode_ratio=cfg.MODEL.decode_ratio,
                        merge_mode=cfg.MODEL.merge_mode,
                        pad_mode=cfg.MODEL.pad_mode,
                        bn_mode=cfg.MODEL.bn_mode,
                        relu_mode=cfg.MODEL.relu_mode,
                        init_mode=cfg.MODEL.init_mode).to(device)

    checkpoint = torch.load(ckpt_path)

    new_state_dict = OrderedDict()
    state_dict = checkpoint['model_weights']
    for k, v in state_dict.items():
        name = k[7:] # remove module.
        # name = k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    valid_provider = Provider_valid(cfg, valid_data=None, test_split=args.test_split)
    val_loader = torch.utils.data.DataLoader(valid_provider, batch_size=1)
    model.eval()
    loss_all = []
    f_txt = open(os.path.join(out_affs, 'scores.txt'), 'w')
    print('the number of sub-volume:', len(valid_provider))
    t1 = time.time()
    pbar = tqdm(total=len(valid_provider))
    for k, data in enumerate(val_loader, 0):
        inputs, target, weightmap = data
        inputs = inputs.cuda()
        target = target.cuda()
        weightmap = weightmap.cuda()
        with torch.no_grad():
            pred = model(inputs)
        valid_provider.add_vol(np.squeeze(pred.data.cpu().numpy()))
        pbar.update(1)
    pbar.close()
    cost_time = time.time() - t1
    print('Inference time=%.6f' % cost_time)
    f_txt.write('Inference time=%.6f' % cost_time)
    f_txt.write('\n')
    output_affs = valid_provider.get_results()
    gt_affs = valid_provider.get_gt_affs()
    gt_seg = valid_provider.get_gt_lb()
    gt_raw = valid_provider.get_raw_data()
    valid_provider.reset_output()
    gt_seg = gt_seg.astype(np.uint32)
    print(f'gt_affs shape is {gt_affs.shape},gt raw shape is {gt_raw.shape}')
    print('*****'*3)

    # save
    if args.save:
        print('save affs...')
        f = h5py.File(os.path.join(out_affs, 'affs.hdf'), 'w')
        f.create_dataset('main', data=output_affs, dtype=np.float32, compression='gzip')
        f.close()

    # segmentation
    if args.waterz:
        print('Waterz segmentation...')
        fragments = watershed(output_affs, 'maxima_distance')
        sf = 'OneMinus<EdgeStatisticValue<RegionGraphType, MeanAffinityProvider<RegionGraphType, ScoreValue>>>'
        segmentation = list(waterz.agglomerate(output_affs, [0.50],
                                            fragments=fragments,
                                            scoring_function=sf,
                                            discretize_queue=256))[0]
        segmentation = relabel(segmentation).astype(np.uint64)
        print('the max id = %d' % np.max(segmentation))
        f = h5py.File(os.path.join(out_affs, 'seg_waterz.hdf'), 'w')
        f.create_dataset('main', data=segmentation, dtype=segmentation.dtype, compression='gzip')
        f.close()

        arand = adapted_rand_ref(gt_seg, segmentation, ignore_labels=(0))[0]
        voi_split, voi_merge = voi_ref(gt_seg, segmentation, ignore_labels=(0))
        voi_sum = voi_split + voi_merge
        print('waterz: voi_split=%.6f, voi_merge=%.6f, voi_sum=%.6f, arand=%.6f' % \
            (voi_split, voi_merge, voi_sum, arand))
        f_txt.write('waterz: voi_split=%.6f, voi_merge=%.6f, voi_sum=%.6f, arand=%.6f' % \
            (voi_split, voi_merge, voi_sum, arand))
        f_txt.write('\n')


    output_affs_prop = output_affs
    f_txt.close()

    # show
    if args.show:
        print('show affs...')
        output_affs_prop = (output_affs_prop * 255).astype(np.uint8)
        gt_affs = (gt_affs * 255).astype(np.uint8)
        for i in range(output_affs_prop.shape[1]):
            cat1 = np.concatenate([output_affs_prop[0,i], output_affs_prop[1,i], output_affs_prop[2,i]], axis=1)
            cat2 = np.concatenate([gt_affs[0,i], gt_affs[1,i], gt_affs[2,i]], axis=1)
            im_cat = np.concatenate([cat1, cat2], axis=0)
            cv2.imwrite(os.path.join(affs_img_path, str(i).zfill(4)+'.png'), im_cat)
        
        print('show seg...')
        segmentation[gt_seg==0] = 0
        color_raw = draw_raw_image(gt_raw)
        color_seg = draw_fragments_3d(segmentation)
        color_gt = draw_fragments_3d(gt_seg)
        for i in range(color_seg.shape[0]):
            im_cat = np.concatenate([color_raw[i],color_seg[i], color_gt[i]], axis=1)
            cv2.imwrite(os.path.join(seg_img_path, str(i).zfill(4)+'.png'), im_cat)
    print('Done')

