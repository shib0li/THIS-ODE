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


class Neural_rnn(nn.Module):

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
        self.init_model = RFF(num_ff=self.nFF, input_dim=self.R*self.nmod)
        
#         self.W = torch.zeros([self.R,self.R], requires_grad=True)
#         torch.nn.init.xavier_normal_(self.W)
#         self.b = torch.zeros(self.R, requires_grad=True)
#         self.log_v = torch.tensor(0.0, requires_grad=True)
        #cprint('r', self.W.shape)
        #cprint('r', self.b.shape)
        #cprint('r', self.log_v.shape)
        
    def todev(self, device):
        self.to(device)
        for i in range(len(self.Ulist)):
            self.Ulist[i] = self.Ulist[i].to(device)
        self.init_model.to(device)
        self.W = torch.zeros([self.R,self.R], requires_grad=True, device=device)
        torch.nn.init.xavier_normal_(self.W)
        self.b = torch.zeros(self.R, requires_grad=True, device=device)
        self.log_v = torch.tensor(0.0, requires_grad=True, device=device)
        
    def trans_prior(self,):
        T = self.Ulist[-1].float()
        
        trans_mu = torch.sigmoid(torch.matmul(T, self.W) + self.b)
        I = torch.eye(T.shape[1]).to(self.dummy.device)
        trans_std = torch.exp(self.log_v)*I
        
        T = T[1:, :]
        trans_mu = trans_mu[:-1, :]
        
        #cprint('r', T.shape)
        #cprint('r', trans_mu.shape)
        
        prior_dist = torch.distributions.MultivariateNormal(loc=trans_mu, covariance_matrix=trans_std)
        
        log_prior = prior_dist.log_prob(T)
#         print(log_prior.sum())
        
#         buff = []
#         for t in range(T.shape[0]):
#             ut = T[t,:]
#             dist = torch.distributions.MultivariateNormal(loc=trans_mu[t,:], covariance_matrix=trans_std)
#             log_prob = dist.log_prob(ut)
#             buff.append(log_prob)
#         #
#         print(sum(buff))

        return log_prior.sum()
        
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
    
    def forward_init(self, b_i_n):
        Uvec = self._extract_Uvec(b_i_n)
        y = self.init_model(Uvec)
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
            
            pred = self.forward_init(b_i_n)
            soln_all.append(pred.squeeze())
        #
        
        soln_all = torch.cat(soln_all)
        obs = torch.cat(ground_all).to(self.dummy.device)
        
        rmse = torch.sqrt(torch.mean(torch.square(obs-soln_all)))
        #nrmse = torch.sqrt(torch.mean(torch.square(obs-soln_all)))/ torch.linalg.norm(obs)
        
        nrmse = torch.sqrt(torch.mean(torch.square(obs-soln_all)))/ torch.sqrt(torch.square(obs).mean())
        
        return rmse.item(), nrmse.item() 
    
    def eval_loss(self, b_i_n, b_obs, Nall):
        pred = self.forward_init(b_i_n)
        
        trans_log_prob = self.trans_prior()
        
        loss = -0.5*(Nall/self.B)*torch.sum(torch.square(pred.squeeze()-b_obs))+trans_log_prob
        return -loss.squeeze()
          
    def train(self, dataset_train, dataset_test, max_epochs, learning_rate, perform_meters):
        
        cprint('b', '@@@@@@@@@@  Neural RNN is being trained @@@@@@@@@@')
        
        dataloader_train = DataLoader(dataset_train, batch_size=self.B, shuffle=True, drop_last=True)
        
        params = []
        for param in self.parameters():
            if param.requires_grad:
                 params.append(param)
            #
        #
        
        params = params + [self.W, self.b, self.log_v]
        
        optimizer = optim.Adam(params, lr=learning_rate)
        
        

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
                
#                 pred = self.forward_init(b_i_n)
#                 loss = torch.mean(torch.square(pred.squeeze()-b_obs))

                loss = self.eval_loss(b_i_n, b_obs, len(dataset_train))
                
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
            

        #