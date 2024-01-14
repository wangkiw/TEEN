import json
import os
import pprint as pprint
import random
import shutil
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import logging
from logging.config import dictConfig
from dataloader.data_utils import *

_utils_pp = pprint.PrettyPrinter()

def set_logging(level, work_dir):
    LOGGING = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {
                "format": f"%(message)s"
            },
        },
        "handlers": {
            "console": {
                "level": f"{level}",
                "class": "logging.StreamHandler",
                'formatter': 'simple',
            },
            'file': {
                'level': f"{level}",
                'formatter': 'simple',
                'class': 'logging.FileHandler',
                'filename': f'{work_dir if work_dir is not None else "."}/train.log',
                'mode': 'a',
            },
        },
        "loggers": {
            "": {
                "level": f"{level}",
                "handlers": ["console", "file"] if work_dir is not None else ["console"],
            },
        },
    }
    dictConfig(LOGGING)
    logging.info(f"Log level set to: {level}")

def pprint(x):
    _utils_pp.pprint(x)
class ConfigEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        elif isinstance(o, Enum):
            return {
                '$enum': o.__module__ + "." + o.__class__.__name__ + '.' + o.name
            }
        elif callable(o):
            return {
                '$function': o.__module__ + "." + o.__name__
            }
        return json.JSONEncoder.default(self, o)
class Logger(object):
    def __init__(self, args, log_dir, **kwargs):
        self.logger_path = os.path.join(log_dir, 'scalars.json')
        # self.tb_logger = SummaryWriter(
        #                     logdir=osp.join(log_dir, 'tflogger'),
        #                     **kwargs,
        #                     )
        self.log_config(vars(args))

        self.scalars = defaultdict(OrderedDict) 

    # def add_scalar(self, key, value, counter):
    def add_scalar(self, key, value, counter):
        assert self.scalars[key].get(counter, None) is None, 'counter should be distinct'
        self.scalars[key][counter] = value
        # self.tb_logger.add_scalar(key, value, counter)

    def log_config(self, variant_data):
        config_filepath = os.path.join(os.path.dirname(self.logger_path), 'configs.json')
        with open(config_filepath, "w") as fd:
            json.dump(variant_data, fd, indent=2, sort_keys=True, cls=ConfigEncoder)

    def dump(self):
        with open(self.logger_path, 'w') as fd:
            json.dump(self.scalars, fd, indent=2)

def set_seed(seed):
    if seed == 0:
        logging.info(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        logging.info('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
def set_gpu(args):
    gpu_list = [int(x) for x in args.gpu.split(',')]
    logging.info('use gpu:', gpu_list)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()

def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        logging.info('create folder:', path)
        os.makedirs(path)

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()

def count_acc_topk(x,y,k=5):
    _,maxk = torch.topk(x,k,dim=-1)
    total = y.size(0)
    test_labels = y.view(-1,1) 
    #top1=(test_labels == maxk[:,0:1]).sum().item()
    topk=(test_labels == maxk).sum().item()
    return float(topk/total)

def count_acc_taskIL(logits, label,args):
    basenum=args.base_class
    incrementnum=(args.num_classes-args.base_class)/args.way
    for i in range(len(label)):
        currentlabel=label[i]
        if currentlabel<basenum:
            logits[i,basenum:]=-1e9
        else:
            space=int((currentlabel-basenum)/args.way)
            low=basenum+space*args.way
            high=low+args.way
            logits[i,:low]=-1e9
            logits[i,high:]=-1e9

    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()

def confmatrix(logits,label):
    
    font={'family':'FreeSerif','size':18}
    matplotlib.rc('font',**font)
    matplotlib.rcParams.update({'font.family':'FreeSerif','font.size':18})
    plt.rcParams["font.family"]="FreeSerif"

    pred = torch.argmax(logits, dim=1)
    cm=confusion_matrix(label, pred, normalize='true')

    return cm

def save_list_to_txt(name, input_list):
    f = open(name, mode='a+')
    for item in input_list:
        f.write(str(item) + '\n')
    f.close()
    
def postprocess_results(result_list, trlog):
        result_list.append('Base Session Best Epoch {}\n'.format(trlog['max_acc_epoch']))
        result_list.append(trlog['max_acc'])
        result_list.append("Seen acc:")
        result_list.append(trlog['seen_acc'])
        result_list.append('Unseen acc:')
        result_list.append(trlog['unseen_acc'])
        hmeans = harm_mean(trlog['seen_acc'], trlog['unseen_acc'])
        result_list.append('Harmonic mean:')
        result_list.append(hmeans)

        logging.info(f"max_acc: {trlog['max_acc']}")        
        logging.info(f"Unseen acc: {trlog['unseen_acc']}")
        logging.info(f"Seen acc: {trlog['seen_acc']}")
        logging.info(f"Harmonic mean: {hmeans}")
        return result_list, hmeans

def save_result(args, trlog, hmeans, **kwargs):
    params_info = args.save_path.split('/')[-1]
    main_path = f"results/main/{args.project}"
    os.makedirs(main_path, exist_ok=True)
    details_path = f"results/details/{args.project}"
    os.makedirs(details_path, exist_ok=True)
    with open(os.path.join(main_path, f"{args.dataset}_results.csv"), "a+") as f:
        f.write(f"{params_info}-{trlog['max_acc'][0]},{trlog['max_acc'][-1]},{trlog['unseen_acc'][0]},{trlog['unseen_acc'][-1]},{hmeans[0]},{hmeans[-1]},{args.time_str} \n")
    with open(os.path.join(details_path, f"{args.dataset}_results.csv"), "a+") as f:
        f.write(f">>> {params_info}-Avg_acc:{trlog['max_acc']} \n Seen_acc:{trlog['seen_acc']} \n Unseen_acc:{trlog['unseen_acc']} \n HMean_acc:{hmeans} \n")

def harm_mean(seen, unseen):
    # compute from session1
    assert len(seen) == len(unseen)
    harm_means = []
    for _seen, _unseen in zip(seen, unseen):
        _hmean = (2 * _seen * _unseen) / (_seen + _unseen + 1e-12)
        _hmean = float('%.3f' % (_hmean))
        harm_means.append(_hmean)
    return harm_means

def get_optimizer(args, model, **kwargs):
        # prepare optimizer
        if args.project in ['teen']:
            if args.optim == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), args.lr_base, 
                                            momentum=args.momentum, nesterov=True,
                                            weight_decay=args.decay)
            elif args.optim == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), 
                                             lr=args.lr_base, weight_decay=args.decay)
        
        
        # prepare scheduler
        if args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, 
                                                        gamma=args.gamma)
        elif args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones,
                                                             gamma=args.gamma)
        elif args.schedule == 'Cosine':
            assert args.tmax >= 0 , "args.tmax should be greater than 0"
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.tmax)
        return optimizer, scheduler

