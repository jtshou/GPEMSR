import os.path as osp
from util.util import OrderedYaml
import util.util as util
from data import create_dataloader
from model.GPEMSR import GPEMSR
import argparse
import yaml
import random
import logging
import numpy as np
import os
import torch
import torch.utils.data as data
import data.util as utils

Loader, Dumper = OrderedYaml()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YAML file.')
    parser.add_argument('--local_rank', type=int, default= os.getenv('LOCAL_RANK', -1))
    args = parser.parse_args()
    opt_path = args.opt
    with open(opt_path, mode='r',encoding='utf-8') as f:
        opt = yaml.load(f, Loader=Loader)
    im_path_SR = opt['save_path']
    scale = opt['scale']
    util.mkdirs([im_path_SR])

    #### Create test dataset and dataloader
    dataset_dict = opt['dataset']
    dataset_dict['scale'] = scale
    dataset_dict = util.dict_to_nonedict(dataset_dict)
    test_set = CREMIDataset(dataset_dict)
    test_loader = create_dataloader(test_set, dataset_dict)
    model = GPEMSR(ref_path_G=opt['network']['ref_path_G'], ref_path_Indexer=opt['network']['ref_path_Indexer'],
                 argref=opt['network']['argref'], nf=opt['network']['nf'],
                 nframes=opt['network']['nframes'], groups=opt['network']['groups'],
                 front_RBs=opt['network']['front_RBs'], back_RBs=opt['network']['back_RBs'],
                 w_ref=opt['network']['w_ref'],
                 ref_fusion_feat_RBs=opt['network']['ref_fusion_feat_RBs'], align_mode=opt['network']['align_mode'],
                 fusion_mode=opt['network']['fusion_mode'],
                 mode=opt['network']['mode'], scale=scale)
    device = torch.device("cuda")
    model.eval()
    model = model.to(device)
    im_num = (dataset_dict['N_frames']-1)/2
    center = int(im_num)
    with torch.no_grad():
        pretrain_path = opt['pretrain_path']
        
        model.load_state_dict(torch.load(pretrain_path), strict=True)
        # pad
        LQ1, GT = test_set[0]['LQ'], test_set[0]['GT']
        LQ = torch.zeros_like(LQ1)
        LQ[0] = LQ1[0]
        LQ[1] = LQ1[0]
        LQ[2] = LQ1[0]
        LQ[3] = LQ1[1]
        LQ[4] = LQ1[2]
        LQ = LQ.unsqueeze(0)
        LQ = LQ.to(device)
        SR, _ = model(LQ)
        SR = SR.cpu()
        SR = util.tensor2img(SR)  # uint8
        save_img_path_SR = osp.join(im_path_SR, '{}.png'.format(int(im_num) - center))
        util.save_img(SR, save_img_path_SR)
        im_num = im_num + 1

        LQ1, GT = test_set[0]['LQ'], test_set[0]['GT']
        LQ = torch.zeros_like(LQ1)
        LQ[0] = LQ1[0]
        LQ[1] = LQ1[0]
        LQ[2] = LQ1[1]
        LQ[3] = LQ1[2]
        LQ[4] = LQ1[3]
        LQ = LQ.unsqueeze(0)
        LQ = LQ.to(device)
        SR, ref = model(LQ)
        SR = SR.cpu()
        SR = util.tensor2img(SR)  # uint8
        save_img_path_SR = osp.join(im_path_SR, '{}.png'.format(int(im_num) - center))
        util.save_img(SR, save_img_path_SR)
        im_num = im_num + 1

        for data in test_loader:
            LQ, GT = data['LQ'], data['GT']
            LQ = LQ.to(device)
            SR, _ = model(LQ)
            SR = SR.cpu()
            SR = util.tensor2img(SR)
            save_img_path_SR = osp.join(im_path_SR, '{}.png'.format(int(im_num) - center))
            if not os.path.exists(im_path_SR):
                os.makedirs(im_path_SR)
            util.save_img(SR, save_img_path_SR)
            im_num = im_num + 1

        # pad
        LQ1, GT = test_set[-1]['LQ'], test_set[-1]['GT']
        LQ = torch.zeros_like(LQ1)
        LQ[0] = LQ1[-4]
        LQ[1] = LQ1[-3]
        LQ[2] = LQ1[-2]
        LQ[3] = LQ1[-1]
        LQ[4] = LQ1[-1]
        LQ = LQ.unsqueeze(0)
        LQ = LQ.to(device)
        SR, _ = model(LQ)
        SR = SR.cpu()
        SR = util.tensor2img(SR)  # uint8
        save_img_path_SR = osp.join(im_path_SR, '{}.png'.format(int(im_num) - center))
        util.save_img(SR, save_img_path_SR)
        im_num = im_num + 1

        LQ1, GT = test_set[-1]['LQ'], test_set[-1]['GT']
        LQ = torch.zeros_like(LQ1)
        LQ[0] = LQ1[-3]
        LQ[1] = LQ1[-2]
        LQ[2] = LQ1[-1]
        LQ[3] = LQ1[-1]
        LQ[4] = LQ1[-1]
        LQ = LQ.unsqueeze(0)
        LQ = LQ.to(device)
        SR, _ = model(LQ)
        SR = SR.cpu()
        SR = util.tensor2img(SR)  # uint8
        save_img_path_SR = osp.join(im_path_SR, '{}.png'.format(int(im_num) - center))
        util.save_img(SR, save_img_path_SR)



class CREMIDataset(data.Dataset):
    '''
    This dataset is slightly different from data.CREMI_dataset.CREMIDataset.
    '''

    def __init__(self, opt):
        super(CREMIDataset, self).__init__()
        self.opt = opt
        self.N_frames = self.opt['N_frames']
        # temporal augmentation
        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']

        self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
        self.data_type = self.opt['data_type']
        self.LR_input = False if opt['GT_size'] == opt['LQ_size'] else True  # low resolution inputs

        self.GT_list = []
        ls_cur = []
        for i in os.listdir(self.GT_root):
            ls_cur.append(int(i[:-4]))
        ls_cur.sort()
        for GT_path in ls_cur[int((self.N_frames - 1) / 2):int(-(self.N_frames - 1) / 2)]:
            self.GT_list.append(os.path.join(self.GT_root, str(GT_path) + '.png'))

        self.LQ_l = list(range(int(-(self.N_frames - 1) / 2), int((self.N_frames - 1) / 2 + 1)))

    def __getitem__(self, index):
        scale = self.opt['scale']
        GT_size = self.opt['GT_size']

        img_GT = utils.read_img(None, self.GT_list[index])
        org_size = img_GT.shape[0:-1]

        #### get LQ images
        LQ_size_tuple = (1, org_size[0] // scale, org_size[1] // scale) if self.LR_input else (
        1, org_size[0], org_size[1])

        GT_split_l = self.GT_list[index].split('/')

        img_LQ_center = int(GT_split_l[-1][0:-4])
        img_LQ_dir_path = self.LQ_root
        img_LQ_l = []

        for i in self.LQ_l:
            LQ_path = seek_path(i, img_LQ_dir_path, img_LQ_center)

            img_LQ = utils.read_img(None, LQ_path)
            img_LQ_l.append(img_LQ)

        if self.opt['phase'] == 'train':
            C, H, W = LQ_size_tuple  # LQ size
            # randomly crop
            if self.LR_input:
                LQ_size = GT_size // scale
                rnd_h = random.randint(0, max(0, H - LQ_size))
                rnd_w = random.randint(0, max(0, W - LQ_size))
                img_LQ_l = [v[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :] for v in img_LQ_l]
                rnd_h_HR, rnd_w_HR = int(rnd_h * scale), int(rnd_w * scale)
                img_GT = img_GT[rnd_h_HR:rnd_h_HR + GT_size, rnd_w_HR:rnd_w_HR + GT_size, :]
            else:
                rnd_h = random.randint(0, max(0, H - GT_size))
                rnd_w = random.randint(0, max(0, W - GT_size))
                img_LQ_l = [v[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :] for v in img_LQ_l]
                img_GT = img_GT[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :]

            # augmentation - flip, rotate
            img_LQ_l.append(img_GT)
            rlt = utils.augment(img_LQ_l, self.opt['use_flip'], self.opt['use_rot'])
            img_LQ_l = rlt[0:-1]
            img_GT = rlt[-1]

        # stack LQ images to NHWC, N is the frame number
        img_LQs = np.stack(img_LQ_l, axis=0)
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1))).copy()).float()
        img_LQs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs,
                                                                     (0, 3, 1, 2))).copy()).float()

        return {'LQ': img_LQs, 'GT': img_GT}

    def __len__(self):
        count = len(self.GT_list)
        return count

def seek_path(idx,dir_path,center):
    cur = center + idx
    if os.path.exists(os.path.join(dir_path,str(cur)+'.png')):
        return os.path.join(dir_path,str(cur)+'.png')
    else:
        idx -= 1
        return seek_path(idx,dir_path,center)

if '__name__'==main():
    main()
