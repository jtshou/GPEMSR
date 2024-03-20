import random
import logging
import numpy as np
import os
import torch
import torch.utils.data as data
import data.util as util

logger = logging.getLogger('base')

'''
    Training dataset and validation dataset for Stage 2.
'''


class IndexerTrainDataset(data.Dataset):

    def __init__(self, opt):
        super(IndexerTrainDataset, self).__init__()
        self.opt = opt
        self.chooseGTtxt = opt['chooseGTtxt']
        self.GT_size = opt['GT_size']
        # augmentation
        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        self.GT_root = opt['dataroot_GT']
        self.LR_root = opt['dataroot_LR']
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
        scale = self.opt['scale']
        GT_size = self.opt['GT_size']
        LR_size = GT_size // scale
        img_GT = util.read_img(None, self.GT_list[index])
        #### get LQ images
        LR_size_tuple = (1, LR_size, LR_size)
        C, H, W = LR_size_tuple
        img_LR_dir_path = os.path.join(self.LR_root, os.path.relpath(self.GT_list[index],self.GT_root))
        img_LR = util.read_img(None, img_LR_dir_path)
        # randomly crop
        rnd_h = random.randint(0, max(0, H - LR_size))
        rnd_w = random.randint(0, max(0, W - LR_size))
        rnd_h_HR, rnd_w_HR = int(rnd_h * scale), int(rnd_w * scale)
        img_GT = img_GT[rnd_h_HR:rnd_h_HR + GT_size, rnd_w_HR:rnd_w_HR + GT_size, :]
        img_LR = img_LR[rnd_h:rnd_h + LR_size, rnd_w:rnd_w + LR_size, :]

        # augmentation - flip, rotate
        rlt = []
        rlt.append(img_GT)
        rlt.append(img_LR)
        rlt = util.augment(rlt, self.opt['use_flip'], self.opt['use_rot'])
        img_GT,img_LR = rlt[0],rlt[1]
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1))).copy()).float()
        img_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1))).copy()).float()
        return img_GT,img_LR

    def __len__(self):
        count = len(self.GT_list)
        return count


class IndexerValDataset(data.Dataset):

    def __init__(self, opt):
        super(IndexerValDataset, self).__init__()
        self.opt = opt
        self.chooseGTtxt = opt['chooseGTtxt']
        self.GT_root = opt['dataroot_GT']
        self.LR_root = opt['dataroot_LR']
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
        img_LR_dir_path = os.path.join(self.LR_root, os.path.relpath(self.GT_list[index], self.GT_root))
        img_LR = util.read_img(None, img_LR_dir_path)
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1))).copy()).float()
        img_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1))).copy()).float()
        return img_GT,img_LR

    def __len__(self):
        count = len(self.GT_list)
        return count
