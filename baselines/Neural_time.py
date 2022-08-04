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

from baselines.RFF import *

# from data.real_events import EventData

# from infrastructure.configs import *

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)


class Neural_time(nn.Module):

    def __init__(self, nmod, nvec, R, nFF, batch_size):
        
        super().__init__()
        
        self.nmod = nmod
        self.nvec = nvec
        assert self.nmod == len(self.nvec)
        
        self.R = R
        self.nFF = nFF
        
        self.B = batch_size
        
        self.register_buffer('dummy', torch.tensor([]))
        
        self.Ulist = nn.ParameterList([nn.Parameter(torch.randn(self.nvec[i], R)) for i in range(self.nmod)])
        self.init_model = RFF(num_ff=self.nFF, input_dim=self.R*self.nmod+1)
        
    def todev(self, device):
        self.to(device)
        for i in range(len(self.Ulist)):
            self.Ulist[i] = self.Ulist[i].to(device)
        self.init_model.to(device)
        
    def _extract_Uvec(self, b_i_n):    
        Uvec = []
        for i_n in b_i_n:
            v_i_n = []
            for i in range(self.nmod):
                v_i_n.append(self.Ulist[i][i_n[i]])
            #
            v_i_n = torch.cat(v_i_n)
            Uvec.append(v_i_n.unsqueeze(0))
        #
        Uvec = torch.cat(Uvec, dim=0) 
        return Uvec
    
    def forward_init(self, b_i_n, b_t_n):
        Uvec = self._extract_Uvec(b_i_n)
        #cprint('r', Uvec.shape)
        #cprint('r', b_t_n.shape)
        inputs = torch.hstack([Uvec, b_t_n.reshape([-1,1])])
        inputs = inputs.float()
        #cprint('r', inputs.shape)
        y = self.init_model(inputs)
        return y
    
    def test(self, dataset):
        dataloader_test = DataLoader(dataset, batch_size=self.B, shuffle=False, drop_last=True)
        
        soln_all = []
        ground_all = []
        
        for b_i_n, b_t_n, b_obs in dataloader_test:
            
            b_i_n, b_t_n, b_obs = \
                b_i_n.to(self.dummy.device), \
                b_t_n.to(self.dummy.device), \
                b_obs.to(self.dummy.device)
            
            ground_all.append(b_obs)
            
            pred = self.forward_init(b_i_n, b_t_n)
            soln_all.append(pred.squeeze())
        #
        
        soln_all = torch.cat(soln_all)
        obs = torch.cat(ground_all).to(self.dummy.device)
        
        rmse = torch.sqrt(torch.mean(torch.square(obs-soln_all)))
        #nrmse = torch.sqrt(torch.mean(torch.square(obs-soln_all)))/ torch.linalg.norm(obs)
        
        nrmse = torch.sqrt(torch.mean(torch.square(obs-soln_all)))/ torch.sqrt(torch.square(obs).mean())
        
        return rmse.item(), nrmse.item()  
    
    def train(self, dataset_train, dataset_test, max_epochs, learning_rate, perform_meters):
        
        cprint('b', '@@@@@@@@@@  Neural Time is being trained @@@@@@@@@@')
        
        dataloader_train = DataLoader(dataset_train, batch_size=self.B, shuffle=True, drop_last=True)
        
#         save_path = os.path.join(
#             '__dump__', 
#             'neural-init',
#             dataset_train.domain, 
#             'fold{}'.format(dataset_train.fold)
#         )
#         create_path(save_path)
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
#         hist_rmse_tr = []
#         hist_rmse_te = []
#         hist_nrmse_tr = []
#         hist_nrmse_te = []
        
#         rmse_tr, nrmse_tr = self.test(dataset_train)
#         rmse_te, nrmse_te = self.test(dataset_test)
        
#         cprint('c', '-------------------------------------------------------')
#         cprint('r', 'init,     rmse_tr={:.6f},  nrmse_tr={:.6f}'.format(rmse_tr, nrmse_tr))
#         cprint('g', '          rmse_te={:.6f},  nrmse_te={:.6f}'.format(rmse_te, nrmse_te))
        
#         hist_rmse_tr.append(rmse_tr.item())
#         hist_rmse_te.append(rmse_te.item())
#         hist_nrmse_tr.append(nrmse_tr.item())
#         hist_nrmse_te.append(nrmse_te.item())

        rmse_tr, nrmse_tr = self.test(dataset_train)
        rmse_te, nrmse_te = self.test(dataset_test)
        
        perform_meters.add_by_epoch(rmse_tr, rmse_te, nrmse_tr, nrmse_te)
        perform_meters.add_by_step(rmse_tr, rmse_te, nrmse_tr, nrmse_te)
        
        steps = 0
        
        for epoch in tqdm(range(max_epochs)):
            
            for b, (b_i_n, b_t_n, b_obs) in enumerate(dataloader_train):
                
                t_start = time.time()
                
                b_i_n, b_t_n, b_obs = \
                    b_i_n.to(self.dummy.device), \
                    b_t_n.to(self.dummy.device), \
                    b_obs.to(self.dummy.device)
                
                pred = self.forward_init(b_i_n, b_t_n)
                loss = torch.mean(torch.square(pred.squeeze()-b_obs))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                steps += 1
                ###### Test steps #######
                if perform_meters.test_interval > 0 and steps % perform_meters.test_interval == 0:
                    rmse_tr, nrmse_tr = self.test(dataset_train)
                    rmse_te, nrmse_te = self.test(dataset_test)

                    perform_meters.add_by_step(rmse_tr, rmse_te, nrmse_tr, nrmse_te)
                    perform_meters.save()
                ##########################
                
                t_end = time.time()
                
                perform_meters.logger.info('({}-batch takes {:.5f} seconds)'.format(b, t_end-t_start))
                
            #
            
            rmse_tr, nrmse_tr = self.test(dataset_train)
            rmse_te, nrmse_te = self.test(dataset_test)
            
            perform_meters.add_by_epoch(rmse_tr, rmse_te, nrmse_tr, nrmse_te)
            perform_meters.save()
            
#             rmse_tr, nrmse_tr = self.test(dataset_train)
#             rmse_te, nrmse_te = self.test(dataset_test)
            
#             hist_rmse_tr.append(rmse_tr.item())
#             hist_rmse_te.append(rmse_te.item())
#             hist_nrmse_tr.append(nrmse_tr.item())
#             hist_nrmse_te.append(nrmse_te.item())
            
#             cprint('c', '-------------------------------------------------------')
#             print('epoch = {}'.format(epoch))
#             cprint('r', 'rmse_tr={:.6f},  nrmse_tr={:.6f}'.format(rmse_tr, nrmse_tr))
#             cprint('g', 'rmse_te={:.6f},  nrmse_te={:.6f}'.format(rmse_te, nrmse_te))
            
#             res = {}
#             res['rmse_tr'] = np.array(hist_rmse_tr)
#             res['rmse_te'] = np.array(hist_rmse_te)
#             res['nrmse_tr'] = np.array(hist_nrmse_tr)
#             res['nrmse_te'] = np.array(hist_nrmse_te)

#             with open(os.path.join(save_path, 'error.pickle'), 'wb') as handle:
#                 pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
#             #

        #