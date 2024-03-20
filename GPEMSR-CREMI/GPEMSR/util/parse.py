import os
import logging
import yaml
from util.util import OrderedYaml
Loader, Dumper = OrderedYaml()


def parse(opt_path):
    with open(opt_path, mode='r',encoding='utf-8') as f:
        opt = yaml.load(f, Loader=Loader)
    # datasets
    for phase, dataset in opt['datasets'].items():
        dataset['phase'] = phase
        if dataset.get('dataroot_GT', None) is not None:
            dataset['dataroot_GT'] = os.path.expanduser(dataset['dataroot_GT'])
        if dataset.get('chooseGTtxt', None) is not None:
            dataset['chooseGTtxt'] = os.path.expanduser(dataset['chooseGTtxt'])
        if dataset.get('dataroot_LR', None) is not None:
            dataset['dataroot_LR'] = os.path.expanduser(dataset['dataroot_LR'])

    if opt.get('scale',None) is not None:
        scale = opt['scale']
        opt['datasets']['train']['scale'] = scale
        opt['datasets']['val']['scale'] = scale
        opt['network']['scale'] = scale
        if opt['stage']==3:
            opt['network']['patch_size'] = opt['datasets']['train']['LQ_size']


    # path
    for key, path in opt['pretrain'].items():
        if path and key in opt['pretrain'] and key != 'strict_load':
            opt['path'][key] = os.path.expanduser(path)
    opt['path']['root'] = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir))

    experiments_root = os.path.join(opt['path']['root'], 'experiments', opt['name'])
    val_path = os.path.join(opt['path']['root'], 'val', opt['val']['val_path_version'])
    opt['path']['experiments_root'] = experiments_root
    opt['path']['log'] = experiments_root
    opt['path']['state'] = os.path.join(experiments_root, 'state')
    if opt['stage'] == 1:
        opt['path']['generator'] = os.path.join(experiments_root, 'models', 'generator')
        opt['path']['discriminator'] = os.path.join(experiments_root, 'models', 'discriminator')
    elif opt['stage'] == 2:
        opt['path']['lrindexer'] = os.path.join(experiments_root, 'models', 'lrindexer{}'.format(scale))
    elif opt['stage'] == 3:
        opt['path']['model'] = os.path.join(experiments_root, 'model')

    #val
    opt['val']['val_path'] = val_path

    return opt



