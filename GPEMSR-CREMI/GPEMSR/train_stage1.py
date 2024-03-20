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
from model.vqgan import VQGAN
from data import create_dataloader, create_dataset
from data.data_sampler import DistIterSampler
import model.lr_scheduler as lr_scheduler

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
    if opt['pretrain'].get('pretrain_model_G', None):
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
    model = VQGAN(opt['network'])
    generator = model.generator
    discriminator = model.discriminator
    if resume_state:
        generator.load_state_dict(torch.load(opt['pretrain']['pretrain_model_G']),
                              strict=opt['pretrain']['strict_load'])
        discriminator.load_state_dict(torch.load(opt['pretrain']['pretrain_model_D']),
                                  strict=opt['pretrain']['strict_load'])

    if opt['dist']:
        generator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(generator)
        generator = DistributedDataParallel(generator.to(device), device_ids=[torch.cuda.current_device()])
        discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)
        discriminator = DistributedDataParallel(discriminator.to(device), device_ids=[torch.cuda.current_device()])
    else:
        discriminator = discriminator.to(device)
        generator = generator.to(device)
    logger.info('Model [{:s}] is created.'.format(model.__class__.__name__))
    current_step = opt['train']['current_step']
    start_epoch = opt['train']['start_epoch']

    #### training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    #optimizer
    optim_params_G = []
    optim_params_D = []
    for k, v in model.named_parameters():
        if v.requires_grad:
            if 'discriminator' in k:
                optim_params_D.append(v)
            else:
                optim_params_G.append(v)
        else:
            if rank <= 0:
                logger.warning('Params [{:s}] will not optimize.'.format(k))
    wd_G = opt['train']['weight_decay_G'] if opt['train']['weight_decay_G'] else 0
    wd_D = opt['train']['weight_decay_D'] if opt['train']['weight_decay_D'] else 0
    optimizer_G = optim.Adam(optim_params_G, lr=opt['train']['lr_G'], betas=(opt['train']['beta1'], opt['train']['beta2']), weight_decay=wd_G)
    optimizer_D = optim.Adam(optim_params_D, lr=opt['train']['lr_D'],
                             betas=(opt['train']['beta1'], opt['train']['beta2']), weight_decay=wd_D)

    #### schedulers
    if opt['train']['lr_scheme'] == 'MultiStepLR':
        scheduler_G = lr_scheduler.MultiStepLR_Restart(optimizer_G, opt['train']['lr_steps'],
                                                 restarts=opt['train']['restarts'],
                                                 weights=opt['train']['restart_weights'],
                                                 gamma=opt['train']['lr_gamma'],
                                                 clear_state=opt['train']['clear_state'])
        scheduler_D = lr_scheduler.MultiStepLR_Restart(optimizer_D, opt['train']['lr_steps'],
                                                       restarts=opt['train']['restarts'],
                                                       weights=opt['train']['restart_weights'],
                                                       gamma=opt['train']['lr_gamma'],
                                                       clear_state=opt['train']['clear_state'])
    elif opt['train']['lr_scheme'] == 'CosineAnnealingLR_Restart':
        scheduler_G = lr_scheduler.CosineAnnealingLR_Restart(optimizer_G, opt['train']['T_period'],
                                                             eta_min=opt['train']['eta_min'],
                                                             restarts=opt['train']['restarts'],
                                                             weights=opt['train']['restart_weights'])
        scheduler_D = lr_scheduler.CosineAnnealingLR_Restart(optimizer_D, opt['train']['T_period'],
                                                             eta_min=opt['train']['eta_min'],
                                                             restarts=opt['train']['restarts'],
                                                             weights=opt['train']['restart_weights'])
    else:
        raise NotImplementedError()


    for epoch in range(start_epoch, total_epochs + 1):
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break
            #### update learning rate
            train_vqgan_onestep(generator,discriminator, opt['train'], current_step, train_data, device,optimizer_G,optimizer_D,scheduler_G,scheduler_D)

            generator.eval()
            #### validation
            if current_step % opt['val']['val_freq'] == 0:
                with torch.no_grad():
                    if opt['dist']:
                        # multi-GPU testing
                        val_len = len(val_set)
                        psnr_rlt = torch.zeros(val_len,dtype=torch.float32,device='cuda') # with border
                        save_img_path_gt_dir = os.path.join(opt['val']['val_path'],str(current_step),'original')
                        save_img_path_fake_dir = os.path.join(opt['val']['val_path'],str(current_step),'fake')
                        if not os.path.exists(save_img_path_fake_dir) and rank==0:
                            os.makedirs(save_img_path_fake_dir)
                        if not os.path.exists(save_img_path_gt_dir) and rank==0:
                            os.makedirs(save_img_path_gt_dir)
                        dist.barrier()
                        for idx in range(rank, val_len, world_size):
                            val_data = val_set[idx].unsqueeze_(0).to(device)
                            #val_data = val_data[:,:,0:512,0:512]
                            fake_data,indice,los = generator(val_data)
                            fake_img = util.tensor2img(fake_data)
                            gt_img = util.tensor2img(val_data)
                            psnr_rlt[idx] = util.calculate_psnr(fake_img, gt_img)
                            #We save 20 images for visualization.
                            if idx<20:
                                save_img_path_gt = os.path.join(save_img_path_gt_dir,str(idx)+'.png')
                                save_img_path_fake = os.path.join(save_img_path_fake_dir, str(idx) + '.png')
                                util.save_img(gt_img, save_img_path_gt)
                                util.save_img(fake_img, save_img_path_fake)
                        dist.barrier()
                        dist.reduce(psnr_rlt,0)
                        if rank == 0:
                            psnr_total_avg = torch.mean(psnr_rlt).item()
                            log_s = '# Validation # PSNR: {:.4e}:'.format(psnr_total_avg)
                            logger.info(log_s)
                            logger.info('current_step:{}'.format(current_step))
                            logger.info('----------------------------')
                            if opt['use_tb_logger'] and 'debug' not in opt['name']:
                                tb_logger.add_scalar('psnr_avg', psnr_total_avg, current_step)
                
                    else:
                        psnr_rlt = torch.zeros(len(val_set), dtype=torch.float32,
                                           device='cuda')  # with border
                        for idx in range(len(val_set)):
                            val_data = val_set[idx].unsqueeze_(0).to(device)
                            fake_data,indice,los = generator(val_data)
                            fake_img = util.tensor2img(fake_data)  # uint8
                            gt_img = util.tensor2img(val_data)  # uint8
                            psnr_rlt[idx] = util.calculate_psnr(fake_img, gt_img)
                            save_img_path_gt_dir = os.path.join(opt['val']['val_path'], str(current_step), 'original')
                            save_img_path_fake_dir = os.path.join(opt['val']['val_path'], str(current_step), 'fake')
                            if not os.path.exists(save_img_path_fake_dir):
                                os.makedirs(save_img_path_fake_dir)
                            if not os.path.exists(save_img_path_gt_dir):
                                os.makedirs(save_img_path_gt_dir)
                            # We save 20 images for visualization.
                            if idx < 20:
                                save_img_path_gt = os.path.join(save_img_path_gt_dir, str(idx) + '.png')
                                save_img_path_fake = os.path.join(save_img_path_fake_dir, str(idx) + '.png')
                                util.save_img(gt_img, save_img_path_gt)
                                util.save_img(fake_img, save_img_path_fake)
                        psnr_total_avg = torch.mean(psnr_rlt).item()
                        log_s = '# Validation # PSNR: {:.4e},current_step:{}'.format(psnr_total_avg,current_step)
                        logger.info(log_s)
                        if opt['use_tb_logger'] and 'debug' not in opt['name']:
                            tb_logger.add_scalar('psnr_avg', psnr_total_avg, current_step)

            #### save models and training states
            if current_step % opt['save_checkpoint_freq'] == 0:
                if rank <= 0:
                    logger.info('Saving models.')
                    save_filename_G = '{}_{}.pth'.format(current_step, 'G')
                    save_filename_D = '{}_{}.pth'.format(current_step, 'D')
                    save_path_G = os.path.join(opt['path']['generator'], save_filename_G)
                    save_path_D = os.path.join(opt['path']['discriminator'], save_filename_D)
                    if isinstance(generator, DataParallel) or isinstance(generator, DistributedDataParallel):
                        generator_m = generator.module
                    state_dict_G = generator_m.state_dict()
                    for key, param in state_dict_G.items():
                        state_dict_G[key] = param.cpu()
                    torch.save(state_dict_G, save_path_G)
                    if current_step > opt['train']['gan_start']:
                        if isinstance(discriminator, DataParallel) or isinstance(discriminator,DistributedDataParallel):
                            discriminator_m = discriminator.module
                        state_dict_D = discriminator_m.state_dict()
                        for key, param in state_dict_D.items():
                            state_dict_D[key] = param.cpu()
                        torch.save(state_dict_D, save_path_D)


    if rank <= 0:
        logger.info('End of training.')
        tb_logger.close()


def train_vqgan_onestep(generator,discriminator,opt_train,current_step,imgs,device,optimizer_G,optimizer_D,scheduler_G,scheduler_D):
    generator.train()
    discriminator.train()
    imgs = imgs.to(device)
    logger = logging.getLogger('base')
    for p in discriminator.parameters():
        p.requires_grad = False
    optimizer_G.zero_grad()
    optimizer_D.zero_grad()
    if current_step % opt_train['generator_update_rate'] == 0 and current_step > opt_train['gan_start']:
        decoded_images, _, q_loss = generator(imgs)
        L1_loss = torch.nn.L1Loss().to(device)
        rec_loss = L1_loss(imgs, decoded_images)
        perceptual_rec_loss = opt_train['rec_loss_factor'] * rec_loss
        disc_fake = discriminator(decoded_images)
        g_loss = torch.mean(- disc_fake)
        vq_loss = perceptual_rec_loss + opt_train['codebook_loss_factor'] * q_loss + opt_train[
                'gan_loss_factor'] * g_loss
        if current_step % opt_train['logger_freq'] == 0:
            logger.info(
                    'Current_step:{},Generator_loss:reconstruction_loss:{},\ngan_loss:{},codebook_feat_loss:{}'.format(
                        current_step,rec_loss.item(), g_loss.item(), q_loss.item()))
        vq_loss.backward()
        optimizer_G.step()
        scheduler_G.step()
    elif current_step <= opt_train['gan_start']:
        decoded_images, _, q_loss = generator(imgs)
        L1_loss = torch.nn.L1Loss().to(device)
        rec_loss = L1_loss(imgs, decoded_images)
        perceptual_rec_loss = opt_train['rec_loss_factor'] * rec_loss
        vq_loss = perceptual_rec_loss + opt_train['codebook_loss_factor'] * q_loss
        if current_step % opt_train['logger_freq'] == 0:
            logger.info('Current_step:{},Generator_loss:reconstruction_loss:{},codebook_feat_loss:{}'.format(current_step,rec_loss.item(),
                                                                                                 q_loss.item()))

        vq_loss.backward()
        optimizer_G.step()
        scheduler_G.step()
    else:
        decoded_images, _, q_loss = generator(imgs)

    if current_step > opt_train['gan_start']:
        for p in discriminator.parameters():
            p.requires_grad = True
        disc_real = discriminator(imgs)
        disc_fake = discriminator(decoded_images.detach())
        d_loss_real = torch.mean(-disc_real)
        d_loss_fake = torch.mean(disc_fake)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)
        d_loss.backward()
        if current_step % opt_train['net_d_reg_every'] == 0:
            imgs.requires_grad = True
            disc_real = discriminator(imgs)
            l_d_r1 = r1_penalty(disc_real, imgs)
            l_d_r1 = opt_train['r1_reg_weight'] / 2 * l_d_r1 * opt_train['net_d_reg_every']+ 0 * torch.mean(disc_real)
            l_d_r1.backward()
            if current_step % opt_train['logger_freq'] == 0:
                logger.info('R1_regularization:{}'.format(l_d_r1.item()))
        optimizer_D.step()
        scheduler_D.step()
        if current_step % opt_train['logger_freq'] == 0:
            logger.info(
                'Current_step:{},Discriminator_loss:d_loss_real:{},d_loss_fake:{}'.format(current_step,d_loss_real.item(), d_loss_fake.item()))
            # learning rate
            lr_g = optimizer_G.param_groups[0]['lr']
            lr_d = optimizer_D.param_groups[0]['lr']
            logger.info('learning_rate_G:{},learning_rate_D:{}'.format(lr_g, lr_d))


def r1_penalty(real_pred, real_img):
    """R1 regularization for discriminator. The core idea is to
        penalize the gradient on real data alone: when the
        generator distribution produces the true data distribution
        and the discriminator is equal to 0 on the data manifold, the
        gradient penalty ensures that the discriminator cannot create
        a non-zero gradient orthogonal to the data manifold without
        suffering a loss in the GAN game.
        """
    grad_real = torch.autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)[0]
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty

if __name__ == '__main__':
    main()
