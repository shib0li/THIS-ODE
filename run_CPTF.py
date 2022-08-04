import numpy as np
import torch
# import torch.distributions as distributions
# from torch.optim import Adam
# from torch.optim import LBFGS
# from sklearn.cluster import KMeans
import random
import pickle
import fire
from tqdm.auto import tqdm, trange

# from baselines.kernels import KernelRBF, KernelARD
from baselines.CPTF_linear import *
from baselines.CPTF_rnn import *
from baselines.CPTF_time import *
from data.real_events import EventData, EventDataBT

from torch.utils.data import Dataset, DataLoader

from infrastructure.misc import *
from infrastructure.configs import *


np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

        
def evaluation(**kwargs):

    config = CPExpConfig()
    config.parse(kwargs)

    device = torch.device(config.device)
    
    domain = config.domain
    fold = config.fold
    
    #cprint('g', ndims)
    #cprint('g', nmod)
    
    if config.trans=='linear':
        method = 'CPTF_linear'
        dataset_train = EventDataBT(domain, mode='train', fold=fold)
        dataset_test = EventDataBT(domain, mode='test', fold=fold)
    elif config.trans=='rnn':
        method = 'CPTF_rnn'
        dataset_train = EventDataBT(domain, mode='train', fold=fold)
        dataset_test = EventDataBT(domain, mode='test', fold=fold)
    elif config.trans=='time':
        method = 'CPTF_time'
        dataset_train = EventData(domain, mode='train', fold=fold)
        dataset_test = EventData(domain, mode='test', fold=fold)
    else:
        raise Exception('Error in run_CPTF.py')
        
    ndims = dataset_train.nvec
    nmod = dataset_train.nmod
    
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
    nepoch = config.max_epochs
    m = config.m
    
    lr = config.learning_rate
    
    dataloader_train = DataLoader(dataset_train, batch_size=len(dataset_train), shuffle=False)
    dataloader_test = DataLoader(dataset_test, batch_size=len(dataset_train), shuffle=False)
    
    train_ind, train_time, train_y = next(iter(dataloader_train))
    train_ind = train_ind.data.numpy()
    train_time = train_time.reshape([-1,1]).data.numpy()
    train_y = train_y.reshape([-1,1]).data.numpy()
    
    test_ind, test_time, test_y = next(iter(dataloader_test))
    test_ind = test_ind.data.numpy()
    test_time = test_time.reshape([-1,1])
    test_y = test_y.reshape([-1,1])
    
    U = []
    for i in range(len(ndims)):
        U.append(np.random.rand(ndims[i],R))
    
    perform_meters = PerformMeters(save_path=res_path, logger=logger, test_interval=config.test_interval)
    
    if config.trans=='linear':
        model = CPTF_linear(train_ind, train_y, U, batch_size, torch.device('cpu'))
        model.train(test_ind, test_y, lr, nepoch, perform_meters)
    elif config.trans=='rnn':
        model = CPTF_rnn(train_ind, train_y, U, batch_size, torch.device('cpu'))
        model.train(test_ind, test_y, lr, nepoch, perform_meters)
    elif config.trans=='time':
        model = CPTF_time(train_ind, train_y, train_time, U, batch_size, torch.device('cpu'))
        model.train(test_ind, test_y, test_time, lr, nepoch, perform_meters)
    else:
        raise Exception('Error in model run_CPTF.py')

    #model.train(test_ind, test_y, lr, nepoch, perform_meters)

if __name__=='__main__':
    fire.Fire(evaluation)
    
    