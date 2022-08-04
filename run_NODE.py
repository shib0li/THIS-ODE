import numpy as np
import torch
from torchdiffeq import odeint
import torch.autograd.functional as F 
import random

import torch.nn as nn
import os
import pickle
import time
import fire

from tqdm.auto import tqdm, trange
from torch.utils.data import Dataset, DataLoader

from core.model import *
from core.NODE import *
from core.NODE_noise import *
from core.NODE_auto import *

from data.real_events import EventData

from infrastructure.misc import *
from infrastructure.configs import *

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)


def evaluation(**kwargs):

    config = NodeExpConfig()
    config.parse(kwargs)

    device = torch.device(config.device)
    
    domain = config.domain
    fold = config.fold

    dataset_train = EventData(domain, mode='train', fold=fold)
    dataset_test = EventData(domain, mode='test', fold=fold)
    
    if config.est == 'pt':
        method = 'NODE'
    elif config.est == 'auto':
        method = 'NODE_auto'
    else:
        method = 'NODE_noise'
    
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
    
    if config.est == 'pt':
        node = NODE(
            nmod=dataset_train.nmod, 
            nvec=dataset_train.nvec, 
            R=R, 
            batch_size=batch_size, 
            int_steps=config.int_steps,
            nFF=config.nFF, 
            solver=config.solver
        )
    elif config.est == 'auto':
        node = NODE_auto(
            nmod=dataset_train.nmod, 
            nvec=dataset_train.nvec, 
            R=R, 
            nFF_init=config.nFF, 
            nFF_dynamic=config.nFF, 
            batch_size=batch_size, 
            steps=config.int_steps, 
            solver=config.solver
        )
    else:
        node = NODE_noise(
            nmod=dataset_train.nmod, 
            nvec=dataset_train.nvec, 
            R=R, 
            batch_size=batch_size, 
            int_steps=config.int_steps,
            nFF=config.nFF, 
            solver=config.solver
        )
    
    node.todev(device)
    
    max_epochs = config.max_epochs
    learning_rate = config.learning_rate
    
    perform_meters = PerformMeters(save_path=res_path, logger=logger, test_interval=config.test_interval)

    node.train(dataset_train, dataset_test, max_epochs, learning_rate, perform_meters)


if __name__=='__main__':
    fire.Fire(evaluation)
    
    
    