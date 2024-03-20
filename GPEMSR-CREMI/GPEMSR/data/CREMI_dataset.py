import os.path as osp
import random
import pickle
import logging
import numpy as np
import cv2
import os
from PIL import Image
import torch
import torch.utils.data as data
import data.util as util

try:
    import mc  # import memcached
except ImportError:
    pass
logger = logging.getLogger('base')


'''
    Training dataset and validation dataset for Stage 3.
'''



class CREMIDataset(data.Dataset):
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
        for i in os.listdir(self.GT_root):
            dir_path = os.path.join(self.GT_root,i)
            ls_cur = []
            for j in os.listdir(dir_path):
                ls_cur.append(int(j[:-4]))
            ls_cur.sort()
            for GT_path in ls_cur[2*int((self.N_frames-1)/2):2*int(-(self.N_frames-1)/2)]:
                self.GT_list.append(os.path.join(dir_path,str(GT_path)+'.png'))

        self.LQ_l = list(range(int(-(self.N_frames-1)/2),int((self.N_frames-1)/2+1)))



    def __getitem__(self, index):
        scale = self.opt['scale']
        GT_size = self.opt['GT_size']
        img_GT = util.read_img(None, self.GT_list[index])
        org_size = img_GT.shape[0:-1]
        #### get LQ images
        LQ_size_tuple = (1, org_size[0]//scale, org_size[1]//scale) if self.LR_input else (1, org_size[0], org_size[1])
        GT_split_l = self.GT_list[index].split('/')
        img_LQ_center = int(GT_split_l[-1][0:-4])
        img_LQ_dir_path = os.path.join(self.LQ_root, GT_split_l[-2])
        img_LQ_l = []
        for i in self.LQ_l:
            LQ_path = seek_path(i,img_LQ_dir_path,img_LQ_center)
            img_LQ = util.read_img(None,LQ_path)
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
            rlt = util.augment(img_LQ_l, self.opt['use_flip'], self.opt['use_rot'])
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
    '''
    In CREMI dataset, some EM images are damaged, we manually delete these images.
    We replace the damaged images with the undamaged ones before them.
    This function is used to find the undamaged images before the damaged images.
    '''
    cur = center + idx
    if os.path.exists(os.path.join(dir_path,str(cur)+'.png')):
        return os.path.join(dir_path,str(cur)+'.png')
    else:
        idx -= 1
        return seek_path(idx,dir_path,center)
    
