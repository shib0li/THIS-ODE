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
from baselines.Tucker import *
from data.real_events import EventData, EventDataBT

from torch.utils.data import Dataset, DataLoader

from infrastructure.misc import *
from infrastructure.configs import *


np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

        
def evaluation(**kwargs):

    config = TuckerExpConfig()
    config.parse(kwargs)

    device = torch.device(config.device)
    
    domain = config.domain
    fold = config.fold

    dataset_train = EventDataBT(domain, mode='train', fold=fold)
    dataset_test = EventDataBT(domain, mode='test', fold=fold)

    ndims = dataset_train.nvec
    nmod = dataset_train.nmod
    
    #cprint('g', ndims)
    #cprint('g', nmod)
    
#     if config.trans=='linear':
#         method = 'CPTF_linear'
#     elif config.trans=='rnn':
#         method = 'CPTF_rnn'
#     else:
#         raise Exception('Error in run_CPTF.py')

    method = 'Tucker'
    
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

    model = Tucker(train_ind, train_y, U, batch_size, torch.device('cpu'))

    model.train(test_ind, test_y, lr, nepoch, perform_meters)

if __name__=='__main__':
    fire.Fire(evaluation)
    
    