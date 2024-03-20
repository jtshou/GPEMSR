import random
import logging
import numpy as np
import cv2
import os
from PIL import Image
import torch
import torch.utils.data as data
import data.util as util

logger = logging.getLogger('base')

'''
    Training dataset and validation dataset for Stage 1.
'''


class VQGANTrainDataset(data.Dataset):


    def __init__(self, opt):
        super(VQGANTrainDataset, self).__init__()
        self.opt = opt
        self.chooseGTtxt = opt['chooseGTtxt']
        self.GT_size = opt['GT_size']
        # augmentation
        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        self.GT_root = opt['dataroot_GT']
        self.GT_list = []
        self.GT_dir = []
        with open(self.chooseGTtxt, 'r') as f:
            for line in f:
                if line.endswith('\n'):
                    self.GT_dir.append(line[:-1])
                else:
                    self.GT_dir.append(line)               
        for i in os.listdir(self.GT_root):
            if i in self.GT_dir:
                dir1_path = os.path.join(self.GT_root, i)
                for j in os.listdir(dir1_path):
                    ls_cur = []
                    dir2_path = os.path.join(dir1_path, j)
                    for k in os.listdir(dir2_path):
                        ls_cur.append(int(k[:-4]))
                    ls_cur.sort()
                    for GT_path in ls_cur:
                        if GT_path<10:
                            self.GT_list.append(os.path.join(dir2_path, '000' + str(GT_path) + '.png'))
                        else:
                            self.GT_list.append(os.path.join(dir2_path, '00' + str(GT_path) + '.png'))
        

    def __getitem__(self, index):
        GT_size = self.opt['GT_size']
        img_GT = util.read_img(None, self.GT_list[index])
        H, W, C = img_GT.shape  # LQ size
        # randomly crop
        rnd_h = random.randint(0, max(0, H - GT_size))
        rnd_w = random.randint(0, max(0, W - GT_size))
        img_GT = img_GT[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :]
        # augmentation - flip, rotate
        rlt = []
        rlt.append(img_GT)
        rlt = util.augment(rlt, self.opt['use_flip'], self.opt['use_rot'])
        img_GT = rlt[-1]
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1))).copy()).float()
        return img_GT

    def __len__(self):
        count = len(self.GT_list)
        return count

class VQGANValDataset(data.Dataset):

    def __init__(self, opt):
        super(VQGANValDataset, self).__init__()
        self.opt = opt
        self.chooseGTtxt = opt['chooseGTtxt']
        self.GT_root = opt['dataroot_GT']
        self.GT_list = []
        self.GT_dir = []
        with open(self.chooseGTtxt, 'r') as f:
            for line in f:
                if line.endswith('\n'):
                    self.GT_dir.append(line[:-1])
                else:
                    self.GT_dir.append(line)
        for i in os.listdir(self.GT_root):
            if i in self.GT_dir:
                dir1_path = os.path.join(self.GT_root, i)
                ls_cur = []
                for j in os.listdir(dir1_path):
                    ls_cur.append(int(j[:-4]))
                ls_cur.sort()
                for GT_path in ls_cur:
                    self.GT_list.append(os.path.join(dir1_path, str(GT_path) + '.png'))

    def __getitem__(self, index):
        img_GT = util.read_img(None, self.GT_list[index])
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1))).copy()).float()
        return img_GT

    def __len__(self):
        count = len(self.GT_list)
        return count
