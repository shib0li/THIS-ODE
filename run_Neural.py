import numpy as np
import torch
import torch.distributions as distributions
from torch.optim import Adam
from torch.optim import LBFGS
from torchdiffeq import odeint
import torch.autograd.functional as F 
import random
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
import os
import pickle
import time
import fire

from infrastructure.misc import *
from tqdm.auto import tqdm, trange
from torch.utils.data import Dataset, DataLoader

from data.real_events import EventData, EventDataBT
from baselines.Neural_linear import *
from baselines.Neural_rnn import *
from baselines.Neural_time import *

from infrastructure.configs import *

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

    

def evaluation(**kwargs):

    config = NeuralExpConfig()
    config.parse(kwargs)

    device = torch.device(config.device)
    
    domain = config.domain
    fold = config.fold

    if config.trans=='linear':
        method = 'Neural_linear'
        dataset_train = EventDataBT(domain, mode='train', fold=fold)
        dataset_test = EventDataBT(domain, mode='test', fold=fold)
    elif config.trans=='rnn':
        method = 'Neural_rnn'
        dataset_train = EventDataBT(domain, mode='train', fold=fold)
        dataset_test = EventDataBT(domain, mode='test', fold=fold)
    elif config.trans=='time':
        method = 'Neural_time'
        dataset_train = EventData(domain, mode='train', fold=fold)
        dataset_test = EventData(domain, mode='test', fold=fold)
    else:
        raise Exception('Error in run_Neural.py')
    
    ndims = dataset_train.nvec
    nmod = dataset_train.nmod
    
    #cprint('g', ndims)
    #cprint('g', nmod)
    
    res_path = os.path.join(
        '__res__', 
        dataset_train.domain, 
        method,
        'rank'+str(config.R),
        'fold{}'.format(dataset_train.fold)
    )

    log_path = os.path.join(
        '__log__', 
        dataset_train.domain, 
        method,
        'rank'+str(config.R),
        'fold{}'.format(dataset_train.fold)
    )
    create_path(res_path)
    create_path(log_path)
    
    logger = get_logger(logpath=os.path.join(log_path, 'exp.log'), displaying=config.verbose)
    logger.info(config)
    
    batch_size = config.batch_size
    R = config.R
    
    if config.trans=='linear':
        model = Neural_linear(
            nmod = dataset_train.nmod,
            nvec = dataset_train.nvec,
            R = R,
            nFF=config.nFF,
            batch_size=batch_size
        )
    elif config.trans=='rnn':
        model = Neural_rnn(
            nmod = dataset_train.nmod,
            nvec = dataset_train.nvec,
            R = R,
            nFF=config.nFF,
            batch_size=batch_size
        )
    elif config.trans=='time':
        model = Neural_time(
            nmod = dataset_train.nmod,
            nvec = dataset_train.nvec,
            R = R,
            nFF=config.nFF,
            batch_size=batch_size
        )
    else:
        raise Exception('Error in model run_Neural.py')
    

    model.todev(device)

    max_epochs = config.max_epochs
    learning_rate = config.learning_rate
    
    perform_meters = PerformMeters(save_path=res_path, logger=logger, test_interval=config.test_interval)
 
    model.train(dataset_train, dataset_test, max_epochs, learning_rate, perform_meters)


if __name__=='__main__':
    fire.Fire(evaluation)
    