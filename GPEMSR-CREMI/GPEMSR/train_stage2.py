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
from model.vqgan_indexer import VQGAN_Indexer16,VQGAN_Indexer8
from data import create_dataloader, create_dataset
from data.data_sampler import DistIterSampler
import model.lr_scheduler as lr_scheduler
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
    if opt['pretrain'].get('Indexer', None):
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
    if opt['scale'] == 16:
        model = VQGAN_Indexer16(opt['network'])
        lrgenerator = model.lrgenerator16
        lrgenerator.load_state_dict(torch.load(opt['pretrain']['VQGAN_G']),
                              strict=False)

    elif opt['scale'] == 8:
        model = VQGAN_Indexer8(opt['network'])
        lrgenerator = model.lrgenerator8
        lrgenerator.load_state_dict(torch.load(opt['pretrain']['VQGAN_G']),
                                    strict=False)

    if resume_state:
        lrgenerator.lrencoder.load_state_dict(torch.load(opt['pretrain']['lrEncoder']),
                              strict=opt['pretrain']['strict_load'])

    if opt['dist']:
        lrgenerator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(lrgenerator)
        lrgenerator = DistributedDataParallel(lrgenerator.to(device), device_ids=[torch.cuda.current_device()],find_unused_parameters=True)

    else:
        lrgenerator = lrgenerator.to(device)
    logger.info('Model [{:s}] is created.'.format(model.__class__.__name__))
    current_step = opt['train']['current_step']
    start_epoch = opt['train']['start_epoch']

    #### training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    #optimizer
    optim_params_G = []

    if opt['dist']:
        for k, v in lrgenerator.module.encoder.named_parameters():
            v.requires_grad = False

        for k, v in lrgenerator.module.codebook.named_parameters():
            v.requires_grad = False

        for k, v in lrgenerator.module.decoder.named_parameters():
            v.requires_grad = False
    else:
        for k, v in lrgenerator.encoder.named_parameters():
            v.requires_grad = False

        for k, v in lrgenerator.codebook.named_parameters():
            v.requires_grad = False

        for k, v in lrgenerator.decoder.named_parameters():
            v.requires_grad = False


    for k, v in lrgenerator.named_parameters():
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



    for epoch in range(start_epoch, total_epochs + 1):
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            img_GT, img_LR = train_data
            current_step += 1
            if current_step > total_iters:
                break
            #### update learning rate
            train_vqgan_onestep(lrgenerator,opt['train'],current_step,img_GT, img_LR,device,optimizer_G,scheduler_G)
            lrgenerator.eval()
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
                            GT,val_data = val_set[idx]
                            LQ = val_data.unsqueeze_(0)
                            B, C, H, W = LQ.size()
                            '''
                            We crop patches during inference time to prevent insufficient memory.
                            '''
                            LQ1 = LQ[:, :, 0:H // 2, 0:W // 2]
                            LQ2 = LQ[:, :, 0:H // 2, W // 2:]
                            LQ3 = LQ[:, :, H // 2:, 0:W // 2]
                            LQ4 = LQ[:, :, H // 2:, W // 2:]
                            LQ1 = LQ1.to(device)
                            SR1 = lrgenerator.module.output_ref(LQ1)
                            SR1 = SR1.cpu()
                            SR1 = util.tensor2img(SR1)  # uint8
                            LQ2 = LQ2.to(device)
                            SR2 = lrgenerator.module.output_ref(LQ2)
                            SR2 = SR2.cpu()
                            SR2 = util.tensor2img(SR2)  # uint8
                            LQ3 = LQ3.to(device)
                            SR3 = lrgenerator.module.output_ref(LQ3)
                            SR3 = SR3.cpu()
                            SR3 = util.tensor2img(SR3)  # uint8
                            LQ4 = LQ4.to(device)
                            SR4 = lrgenerator.module.output_ref(LQ4)
                            SR4 = SR4.cpu()
                            SR4 = util.tensor2img(SR4)  # uint8
                            GT = util.tensor2img(GT)
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
                        dist.barrier()
                        dist.reduce(psnr_rlt, 0)
                        if rank == 0:
                            psnr_total_avg = torch.mean(psnr_rlt).item()
                            log_s = '# Validation # PSNR: {:.4e},current_step:{}'.format(psnr_total_avg, current_step)
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
                            GT, val_data = val_set[idx]
                            LQ = val_data.unsqueeze_(0)
                            B, C, H, W = LQ.size()
                            '''
                            We crop patches during inference time to prevent insufficient memory.
                            '''
                            LQ1 = LQ[:, :, 0:H // 2, 0:W // 2]
                            LQ2 = LQ[:, :, 0:H // 2, W // 2:]
                            LQ3 = LQ[:, :, H // 2:, 0:W // 2]
                            LQ4 = LQ[:, :, H // 2:, W // 2:]
                            LQ1 = LQ1.to(device)
                            SR1 = lrgenerator.output_ref(LQ1)
                            SR1 = SR1.cpu()
                            SR1 = util.tensor2img(SR1)  # uint8
                            LQ2 = LQ2.to(device)
                            SR2 = lrgenerator.output_ref(LQ2)
                            SR2 = SR2.cpu()
                            SR2 = util.tensor2img(SR2)  # uint8
                            LQ3 = LQ3.to(device)
                            SR3 = lrgenerator.output_ref(LQ3)
                            SR3 = SR3.cpu()
                            SR3 = util.tensor2img(SR3)  # uint8
                            LQ4 = LQ4.to(device)
                            SR4 = lrgenerator.output_ref(LQ4)
                            SR4 = SR4.cpu()
                            SR4 = util.tensor2img(SR4)  # uint8
                            GT = util.tensor2img(GT)
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
                    save_path_G = os.path.join(opt['path']['lrindexer'], save_filename_G)
                    if isinstance(lrgenerator, DataParallel) or isinstance(lrgenerator, DistributedDataParallel):
                        lrencoder_m = lrgenerator.module.indexer
                        state_dict_G = lrencoder_m.state_dict()
                    else:
                        state_dict_G = lrgenerator.indexer.state_dict()

                    for key, param in state_dict_G.items():
                        state_dict_G[key] = param.cpu()
                    torch.save(state_dict_G, save_path_G)



    if rank <= 0:
        logger.info('End of training.')
        tb_logger.close()


def train_vqgan_onestep(lrgenerator,opt_train,current_step,img_GT, img_LR,device,optimizer_G,scheduler_G):
    lrgenerator.train()
    img_GT = img_GT.to(device)
    img_LR = img_LR.to(device)
    logger = logging.getLogger('base')
    optimizer_G.zero_grad()
    logits, gtcodebook_indices = lrgenerator(img_LR,img_GT)
    CELoss = torch.nn.CrossEntropyLoss().to(device)
    loss_total = CELoss(logits, gtcodebook_indices)
    loss_total.backward()
    optimizer_G.step()
    scheduler_G.step()
    if current_step % opt_train['logger_freq'] == 0:
        lr_g = optimizer_G.param_groups[0]['lr']
        logger.info('learning_rate:{},current_step:{},loss:{}'.format(
                    lr_g,current_step,loss_total))

if __name__ == '__main__':
    main()
