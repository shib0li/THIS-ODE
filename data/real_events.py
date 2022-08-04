import os
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm, trange
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

import torch
from torch.utils.data import Dataset, DataLoader

from infrastructure.misc import *

np.random.seed(0)
random.seed(0)

def load_processed_folds(domain, mode, fold):
    
    pickle_name = domain+'.pickle'
    
    save_path = os.path.join('data', 'processed', pickle_name)
    
    with open(save_path, 'rb') as handle:
        D = pickle.load(handle)
    #
    
    #print(D.keys())
    
    if mode == 'train':
        data = D['train_folds'][fold]
    elif mode == 'test':
        data = D['test_folds'][fold]
    else:
        raise Exception('Error data mode')
    #
    
    nvec = D['nvec']
    ind  = data[:,:-2]
    t    = data[:,-2]
    obs  = data[:,-1]
    t_min = D['t_min']
    t_max = D['t_max']
    
    return ind, t, obs, nvec, t_min, t_max
    
#

class EventData(Dataset):
    
    def __init__(self, domain, mode, fold):
        super().__init__()
        
        self.domain = domain
        self.mode = mode
        self.fold = fold
        
        self.ind_n, self.t_n, self.y_n, self.nvec, self.t_min, self.t_max = \
            load_processed_folds(self.domain, self.mode, self.fold)
        
        self.nmod = len(self.nvec)
        
        
    def __getitem__(self, index):
        
        indices = self.ind_n[index].astype(int)
        t = self.t_n[index].astype(float)
        obs = self.y_n[index].astype(float)
  
        return indices, t, obs
    
    def __len__(self,):
        return self.ind_n.shape[0]
    
class EventDataBT(Dataset):
    
    def __init__(self, domain, mode, fold, Kbins_time=50):
        super().__init__()
        
        self.domain = domain
        self.mode = mode
        self.fold = fold
        
        self.ind_n, self.t_n, self.y_n, self.nvec, self.t_min, self.t_max = \
            load_processed_folds(self.domain, self.mode, self.fold)
        
        
        #cprint('r', self.t_min)
        #cprint('r', self.t_max)
        
        bins_time = np.linspace(
            start=self.t_min,
            stop=self.t_max,
            num=Kbins_time+1
        )[1:-1]
        
        #print(bins_time)
        
        bin_t_n = np.digitize(self.t_n, bins=bins_time).reshape([-1,1])
        
        # append binned time to ind_n
        self.ind_n = np.hstack([self.ind_n, bin_t_n])
        self.nvec.append(Kbins_time)
        self.nmod = len(self.nvec)
                
    def __getitem__(self, index):
        
        indices = self.ind_n[index].astype(int)
        t = self.t_n[index].astype(float)
        obs = self.y_n[index].astype(float)
  
        return indices, t, obs
    
    def __len__(self,):
        return self.ind_n.shape[0]

# domain = 'Server'
# dataset_train = EventData(domain, mode='train', fold=1)
# cprint('r', dataset_train.nvec)
# cprint('b', dataset_train.nmod)
# print(len(dataset_train))
# dataset_test = EventData(domain, mode='test', fold=1)
# cprint('r', dataset_test.nvec)
# cprint('b', dataset_test.nmod)
# print(len(dataset_test))

# dataloader = DataLoader(dataset_train, batch_size=50, shuffle=False)
# b_ind, b_t, b_obs = next(iter(dataloader))

# print(b_ind)
# print(b_t)
# print(b_obs)

# dataloader = DataLoader(dataset_test, batch_size=50, shuffle=False)
# b_ind, b_t, b_obs = next(iter(dataloader))

# print(b_ind)
# print(b_t)
# print(b_obs)

# domain = 'Server'
# dataset_train = EventDataBT(domain, mode='train', fold=1)
# cprint('r', dataset_train.nvec)
# cprint('b', dataset_train.nmod)
# print(len(dataset_train))
# dataset_test = EventDataBT(domain, mode='test', fold=1)
# cprint('r', dataset_test.nvec)
# cprint('b', dataset_test.nmod)
# print(len(dataset_test))

# dataloader = DataLoader(dataset_train, batch_size=50, shuffle=False)
# b_ind, b_t, b_obs = next(iter(dataloader))

# print(b_ind)
# # print(b_t)
# print(b_obs)

# dataloader = DataLoader(dataset_test, batch_size=50, shuffle=False)
# b_ind, b_t, b_obs = next(iter(dataloader))

# print(b_ind)
# # print(b_t)
# print(b_obs)
