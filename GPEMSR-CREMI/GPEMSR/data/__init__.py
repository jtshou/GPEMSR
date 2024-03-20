"""create dataset and dataloader"""
import logging
import torch
import torch.utils.data


def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    phase = dataset_opt['phase']
    if phase == 'train':
        if opt['dist']:
            world_size = torch.distributed.get_world_size()
            num_workers = dataset_opt['n_workers']
            assert dataset_opt['batch_size'] % world_size == 0
            batch_size = dataset_opt['batch_size'] // world_size
            shuffle = False
        else:
            num_workers = dataset_opt['n_workers']
            batch_size = dataset_opt['batch_size']
            shuffle = True
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler, drop_last=True,
                                           pin_memory=False)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                                           pin_memory=False)


def create_dataset(dataset_opt):
    if dataset_opt['mode']=='train' and dataset_opt['name']=='VQGAN_train':
        from data.VQGAN_dataset import VQGANTrainDataset as D
    elif dataset_opt['mode']=='val' and dataset_opt['name']=='VQGAN_val':
        from data.VQGAN_dataset import VQGANValDataset as D
    elif dataset_opt['mode']=='train' and dataset_opt['name']=='Indexer_train':
        from data.Indexer_dataset import IndexerTrainDataset as D
    elif dataset_opt['mode']=='val' and dataset_opt['name']=='Indexer_val':
        from data.Indexer_dataset import IndexerValDataset as D
    elif dataset_opt['name']=='CREMIDataset':
        from data.CREMI_dataset import CREMIDataset as D

    dataset = D(dataset_opt)
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
