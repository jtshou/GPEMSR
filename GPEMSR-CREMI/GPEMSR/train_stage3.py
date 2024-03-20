import os
import math
import argparse
import random
import logging
from torch.nn.parallel import DistributedDataParallel,DataParallel
import torch.optim as optim
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from util.parse import parse
import util.util as util
from model.GPEMSR import GPEMSR
from data import create_dataloader, create_dataset
from data.data_sampler import DistIterSampler
import model.lr_scheduler as lr_scheduler
from model.contextual import ContextualLoss
import numpy as np

def init_dist(backend='nccl', **kwargs):
    """initialization for distributed training"""
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)

def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YAML file.')
    parser.add_argument('--local_rank', type=int, default=os.getenv('LOCAL_RANK', -1))
    args = parser.parse_args()
    opt = parse(args.opt)

    #### distributed training settings
    if opt['dist']:
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    else:
        rank = 0

    #### loading resume state if exists
    if opt['pretrain'].get('lrEncoder', None):
        resume_state = True
        logger = logging.getLogger('base')
        logger.info('Resume training starts!!!!!!!!!!!!!!!!')
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            util.mkdir_and_rename(
                opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items()))
        if not os.path.exists(opt['val']['val_path']):
            os.makedirs(opt['val']['val_path'])
        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(util.dict2str(opt))
        # tensorboard logger
        if opt['use_tb_logger'] and 'debug' not in opt['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir='../tb_logger/' + opt['name'])
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')
        tb_logger = None

    # convert to NoneDict, which returns None for missing keys
    opt = util.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if opt['use_gpu'] else 'cpu')


    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch ,support only distributed training
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt['dist']:
                train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
                total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    #### create model
    model = GPEMSR(ref_path_G=opt['network']['ref_path_G'],ref_path_Indexer=opt['network']['ref_path_Indexer'],
                 argref=opt['network']['argref'],nf=opt['network']['nf'],
                 nframes=opt['network']['nframes'], groups=opt['network']['groups'],
                 front_RBs=opt['network']['front_RBs'], back_RBs=opt['network']['back_RBs'],
                 w_ref=opt['network']['w_ref'],
                 ref_fusion_feat_RBs=opt['network']['ref_fusion_feat_RBs'],align_mode=opt['network']['align_mode'],
                 fusion_mode=opt['network']['fusion_mode'],
                 mode=opt['network']['mode'],scale=opt['network']['scale'])


    if resume_state:
        model.load_state_dict(torch.load(opt['pretrain']['EMSR']),
                              strict=opt['pretrain']['strict_load'])
        training_state = torch.load(opt['pretrain']['training_state'])


    if opt['dist']:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model.to(device), device_ids=[torch.cuda.current_device()],find_unused_parameters=True)

    else:
        model = model.to(device)
    logger.info('Model [{:s}] is created.'.format(model.__class__.__name__))
    current_step = opt['train']['current_step']
    start_epoch = opt['train']['start_epoch']
    if resume_state and current_step != training_state['iter']:
        raise ValueError()

    #### training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    #optimizer
    optim_params_G = []

    for k, v in model.named_parameters():
        if v.requires_grad:
            optim_params_G.append(v)
        else:
            if rank <= 0:
                logger.warning('Params [{:s}] will not optimize.'.format(k))
    wd_G = opt['train']['weight_decay_G'] if opt['train']['weight_decay_G'] else 0
    optimizer_G = optim.Adam(optim_params_G, lr=opt['train']['lr_G'], betas=(opt['train']['beta1'], opt['train']['beta2']), weight_decay=wd_G)

    #### schedulers
    if opt['train']['lr_scheme'] == 'MultiStepLR':
        scheduler_G = lr_scheduler.MultiStepLR_Restart(optimizer_G, opt['train']['lr_steps'],
                                                 restarts=opt['train']['restarts'],
                                                 weights=opt['train']['restart_weights'],
                                                 gamma=opt['train']['lr_gamma'],
                                                 clear_state=opt['train']['clear_state'])

    elif opt['train']['lr_scheme'] == 'CosineAnnealingLR_Restart':
        scheduler_G = lr_scheduler.CosineAnnealingLR_Restart(optimizer_G, opt['train']['T_period'],
                                                             eta_min=opt['train']['eta_min'],
                                                             restarts=opt['train']['restarts'],
                                                             weights=opt['train']['restart_weights'])

    else:
        raise NotImplementedError()

    if resume_state:
        optimizer_G.load_state_dict(training_state['optimizers']['0'])
        scheduler_G.load_state_dict(training_state['schedulers']['0'])


    for epoch in range(start_epoch, total_epochs + 1):
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            GT, LR = train_data['GT'], train_data['LQ']
            current_step += 1
            if current_step > total_iters:
                break
            #### update learning rate
            train_EMSR_onestep(model,opt['train'],current_step,GT, LR,device,optimizer_G,scheduler_G,tb_logger,rank)
            model.eval()
            #### validation
            if current_step % opt['val']['val_freq'] == 0:
                with torch.no_grad():
                    if opt['dist']:
                        # multi-GPU testing
                        val_len = len(val_set)
                        psnr_rlt = torch.zeros(val_len,dtype=torch.float32,device='cuda') # with border
                        save_img_path_gt_dir = os.path.join(opt['val']['val_path'],str(current_step),'gt')
                        save_img_path_fake_dir = os.path.join(opt['val']['val_path'],str(current_step),'fake_reference')
                        if not os.path.exists(save_img_path_fake_dir) and rank==0:
                            os.makedirs(save_img_path_fake_dir)
                        if not os.path.exists(save_img_path_gt_dir) and rank==0:
                            os.makedirs(save_img_path_gt_dir)
                        dist.barrier()
                        for idx in range(rank, val_len, world_size):
                            val_data = val_set[idx]
                            gt,lr = val_data['GT'], val_data['LQ']
                            gt = gt.unsqueeze_(0)
                            LQ = lr.unsqueeze_(0)
                            B, N, C, H, W = LQ.size()
                            '''
                            We crop patches during inference time to prevent insufficient memory.
                            '''
                            LQ1 = LQ[:, :, :, 0:H // 2, 0:W // 2]
                            LQ2 = LQ[:, :, :, 0:H // 2, W // 2:]
                            LQ3 = LQ[:, :, :, H // 2:, 0:W // 2]
                            LQ4 = LQ[:, :, :, H // 2:, W // 2:]
                            LQ1 = LQ1.to(device)
                            SR1, _ = model(LQ1)
                            SR1 = SR1.cpu()
                            SR1 = util.tensor2img(SR1)  # uint8
                            LQ2 = LQ2.to(device)
                            SR2, _ = model(LQ2)
                            SR2 = SR2.cpu()
                            SR2 = util.tensor2img(SR2)  # uint8
                            LQ3 = LQ3.to(device)
                            SR3, _ = model(LQ3)
                            SR3 = SR3.cpu()
                            SR3 = util.tensor2img(SR3)  # uint8
                            LQ4 = LQ4.to(device)
                            SR4, _ = model(LQ4)
                            SR4 = SR4.cpu()
                            SR4 = util.tensor2img(SR4)  # uint8
                            GT = util.tensor2img(gt)
                            SR = np.zeros_like(GT)
                            SR[0:H // 2 * opt['scale'], 0:W // 2 *opt['scale']] = SR1
                            SR[0:H // 2 * opt['scale'], W // 2 * opt['scale']:] = SR2
                            SR[H // 2 * opt['scale']:, 0:W // 2 * opt['scale']] = SR3
                            SR[H // 2 * opt['scale']:, W // 2 * opt['scale']:] = SR4
                            psnr_rlt[idx] = util.calculate_psnr(GT, SR)
                            if idx<20:
                                save_img_path_gt = os.path.join(save_img_path_gt_dir,str(idx)+'.png')
                                save_img_path_fake = os.path.join(save_img_path_fake_dir, str(idx) + '.png')
                                util.save_img(GT, save_img_path_gt)
                                util.save_img(SR, save_img_path_fake)
                        dist.barrier()
                        dist.reduce(psnr_rlt,0)
                        if rank == 0:
                            psnr_total_avg = torch.mean(psnr_rlt).item()
                            log_s = '# Validation # PSNR: {:.4e},current_step:{}'.format(psnr_total_avg,current_step)
                            logger.info(log_s)
                            if opt['use_tb_logger']:
                                tb_logger.add_scalar('psnr_avg', psnr_total_avg, current_step)

                    else:
                        psnr_rlt = torch.zeros(len(val_set), dtype=torch.float32,
                                           device='cuda')  # with border
                        save_img_path_gt_dir = os.path.join(opt['val']['val_path'], str(current_step), 'gt')
                        save_img_path_fake_dir = os.path.join(opt['val']['val_path'], str(current_step),
                                                              'fake_reference')
                        if not os.path.exists(save_img_path_fake_dir) and rank == 0:
                            os.makedirs(save_img_path_fake_dir)
                        if not os.path.exists(save_img_path_gt_dir) and rank == 0:
                            os.makedirs(save_img_path_gt_dir)
                        for idx in range(len(val_set)):
                            val_data = val_set[idx]
                            gt, lr = val_data['GT'], val_data['LQ']
                            gt = gt.unsqueeze_(0)
                            LQ = lr.unsqueeze_(0)
                            B, N, C, H, W = LQ.size()
                            '''
                            We crop patches during inference time to prevent insufficient memory.
                            '''
                            LQ1 = LQ[:, :, :, 0:H // 2, 0:W // 2]
                            LQ2 = LQ[:, :, :, 0:H // 2, W // 2:]
                            LQ3 = LQ[:, :, :, H // 2:, 0:W // 2]
                            LQ4 = LQ[:, :, :, H // 2:, W // 2:]
                            LQ1 = LQ1.to(device)
                            SR1, _ = model(LQ1)
                            SR1 = SR1.cpu()
                            SR1 = util.tensor2img(SR1)  # uint8
                            LQ2 = LQ2.to(device)
                            SR2, _ = model(LQ2)
                            SR2 = SR2.cpu()
                            SR2 = util.tensor2img(SR2)  # uint8
                            LQ3 = LQ3.to(device)
                            SR3, _ = model(LQ3)
                            SR3 = SR3.cpu()
                            SR3 = util.tensor2img(SR3)  # uint8
                            LQ4 = LQ4.to(device)
                            SR4, _ = model(LQ4)
                            SR4 = SR4.cpu()
                            SR4 = util.tensor2img(SR4)  # uint8
                            GT = util.tensor2img(gt)
                            SR = np.zeros_like(GT)
                            SR[0:H // 2 * opt['scale'], 0:W // 2 * opt['scale']] = SR1
                            SR[0:H // 2 * opt['scale'], W // 2 * opt['scale']:] = SR2
                            SR[H // 2 * opt['scale']:, 0:W // 2 * opt['scale']] = SR3
                            SR[H // 2 * opt['scale']:, W // 2 * opt['scale']:] = SR4
                            psnr_rlt[idx] = util.calculate_psnr(GT, SR)
                            if idx < 20:
                                save_img_path_gt = os.path.join(save_img_path_gt_dir, str(idx) + '.png')
                                save_img_path_fake = os.path.join(save_img_path_fake_dir, str(idx) + '.png')
                                util.save_img(GT, save_img_path_gt)
                                util.save_img(SR, save_img_path_fake)
                        psnr_total_avg = torch.mean(psnr_rlt).item()
                        log_s = '# Validation # PSNR: {:.4e},current_step:{}'.format(psnr_total_avg,current_step)
                        logger.info(log_s)
                        if opt['use_tb_logger']:
                            tb_logger.add_scalar('psnr_avg', psnr_total_avg, current_step)


            #### save models and training states
            if current_step % opt['save_checkpoint_freq'] == 0:
                if rank <= 0:
                    logger.info('Saving models.')
                    save_filename_G = '{}_{}.pth'.format(current_step, 'G')
                    save_path_G = os.path.join(opt['path']['model'], save_filename_G)
                    if isinstance(model, DataParallel) or isinstance(model, DistributedDataParallel):
                        model_m = model.module
                        state_dict_G = model_m.state_dict()
                    else:
                        state_dict_G = model.state_dict()

                    for key, param in state_dict_G.items():
                        state_dict_G[key] = param.cpu()
                    torch.save(state_dict_G, save_path_G)



    if rank <= 0:
        logger.info('End of training.')
        tb_logger.close()


def train_EMSR_onestep(model,opt_train,current_step,GT, LR,device,optimizer_G,scheduler_G,tb_logger,rank):
    model.train()
    GT = GT.to(device)
    LR = LR.to(device)
    logger = logging.getLogger('base')
    optimizer_G.zero_grad()
    SR,ref_img = model(LR)
    L1_loss = torch.nn.L1Loss().to(device)  # mean
    rec_loss = L1_loss(GT, SR)  # loss1
    if isinstance(model, DataParallel) or isinstance(model, DistributedDataParallel):
        CLoss = ContextualLoss(model.module.vgg).to(device)
    else:
        CLoss = ContextualLoss(model.vgg).to(device)
    b, c, h, w = SR.size()
    b_ref, t, _, _, _ = ref_img.size()
    sr_frame_batch = SR[:, None].expand(-1, -1, 3, -1, -1).expand(-1, t, -1, -1, -1).reshape(b * t, 3, h, w)
    ref_frame_batch = ref_img.expand(-1, -1, 3, -1, -1).reshape(b_ref * t, 3, h, w)
    ref_loss, u = CLoss(sr_frame_batch, ref_frame_batch)  # loss2

    loss_total = rec_loss * opt_train['rec_loss_factor'] + opt_train[
        'ref_loss_factor'] * ref_loss
    loss_total.backward()
    optimizer_G.step()
    scheduler_G.step()
    if current_step % opt_train['logger_freq'] == 0:
        lr_g = optimizer_G.param_groups[0]['lr']
        logger.info('learning_rate:{},current_step:{},rec_loss:{},reference_loss:{}'.format(
            lr_g, current_step, rec_loss, ref_loss))
        if tb_logger != None:
            if rank <= 0:
                tb_logger.add_scalar('rec_loss', rec_loss, current_step)
                tb_logger.add_scalar('ref_loss', ref_loss, current_step)


if __name__ == '__main__':
    main()
