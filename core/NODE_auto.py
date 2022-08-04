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

from data.real_events import EventData

from infrastructure.misc import *
from infrastructure.configs import *

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

torch.set_default_tensor_type(torch.DoubleTensor)
        
        
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

from data.real_events import EventData


from infrastructure.configs import *

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

torch.set_default_tensor_type(torch.DoubleTensor)

class RFF(nn.Module):
    
    def __init__(self, num_ff, input_dim):
        
        super().__init__()
        
        self.num_ff = num_ff*2
        self.input_dim = input_dim
        
        self.linear1 = torch.nn.Linear(in_features=self.input_dim, out_features=self.num_ff)
        self.linear2 = torch.nn.Linear(in_features=self.num_ff, out_features=1)
        
        
    def forward(self, X):
        h = self.linear1(X)
        h = torch.tanh(h)
        y = self.linear2(h)
        return y
    
# class NODE_auto(nn.Module):
    

#     def __init__(self, nmod, nvec, R, nFF_init, nFF_dynamic, batch_size, steps, solver):
        
#         super().__init__()
        
#         self.nmod = nmod
#         self.nvec = nvec
#         assert self.nmod == len(self.nvec)
        
#         self.R = R
#         self.nFF_init = nFF_init
#         self.nFF_dynamic = nFF_dynamic
        
#         self.B = batch_size
        
#         self.steps = steps
#         self.solver = solver
        
#         self.register_buffer('dummy', torch.tensor([]))
        
#         self.register_buffer('a0', torch.tensor(1e-3))
#         self.register_buffer('b0', torch.tensor(1e-3))
#         self.gamma_prior = torch.distributions.Gamma(self.a0, self.b0)
        

#         self.Ulist = nn.ParameterList([nn.Parameter(torch.rand(self.nvec[i], R)) for i in range(self.nmod)])
        
#         #self.init_model = RFF(num_ff=self.nFF_init, input_dim=self.R*self.nmod)
#         self.dynamic_model = RFF(num_ff=self.nFF_dynamic, input_dim=self.R*self.nmod+2)
        
#         self.log_tau = nn.Parameter(torch.tensor(0.0))
        
#         self.scaleT = None
        
#     def todev(self, device):
#         self.to(device)
#         for i in range(len(self.Ulist)):
#             self.Ulist[i] = self.Ulist[i].to(device)
#         #self.init_model.to(device)
#         self.dynamic_model.to(device)
        
# #     def _extract_Uvec(self, b_i_n):    
# #         Uvec = []
# #         for i_n in b_i_n:
# #             v_i_n = []
# #             for i in range(self.nmod):
# #                 v_i_n.append(self.Ulist[i][i_n[i]])
# #             #
# #             v_i_n = torch.cat(v_i_n)
# #             Uvec.append(v_i_n.unsqueeze(0))
# #         #
# #         Uvec = torch.cat(Uvec, dim=0) 
# #         return Uvec

#     def _extract_Uvec(self, b_i_n):    
#         Uvec = []
#         for i_n in b_i_n:
#             v_i_n = []
#             for i in range(self.nmod):
#                 v_i_n.append(self.Ulist[i][i_n[i]])
#             #
#             v_i_n = torch.vstack(v_i_n)
#             Uvec.append(v_i_n.unsqueeze(0))
#         #
#         Uvec = torch.cat(Uvec, dim=0) 
#         #cprint('g', Uvec.shape)
#         return Uvec
    
# #     def forward_init(self, b_i_n):
# #         Uvec = self._extract_Uvec(b_i_n)
# #         y = self.init_model(Uvec)
# #         return y

#     def forward_init(self, b_i_n):
#         Uvec = self._extract_Uvec(b_i_n)
#         #cprint('g', Uvec.shape)
#         beta = torch.prod(Uvec, dim=1).sum(1).view([-1,1]) # CP, B by 1
#         #cprint('r', beta.shape)
#         return beta
    
#     def dynamics(self, t, x, Uvec):
#         # Uvec: N by (nmod*R)
#         # x: m(t) N by 1
#         t_coln_scaled = self.scaleT.view(self.B, 1)
#         inputs = torch.hstack([t_coln_scaled*t, x, Uvec])
#         x_t = self.dynamic_model.forward(inputs)*self.scaleT.view(self.B, 1)
#         return x_t
        
# #     def solve(self, b_i_n, b_t_n):
# #         ####### Important #####
# #         self.scaleT = b_t_n
# #         #######################
        
# #         x0 = self.forward_init(b_i_n)
# #         Uvec = self._extract_Uvec(b_i_n)
        
# #         self.dynamics(0.0, x0, Uvec)
        
# #         t_span = torch.linspace(
# #             start=torch.tensor(0.0), end=torch.tensor(1.0), steps=self.steps).to(self.dummy.device)
        
        
        
# #         soln = odeint(
# #             func=lambda t, x:self.dynamics(t, x, Uvec), 
# #             y0=x0, 
# #             t=t_span.to(self.dummy.device), 
# #             method=self.solver
# #         )
        
# #         x_t = soln[-1]
        
# #         return x_t

#     def solve(self, b_i_n, b_t_n):
#         ####### Important #####
#         self.scaleT = b_t_n
#         #######################
        
#         x0 = self.forward_init(b_i_n)
#         Uvec = self._extract_Uvec(b_i_n)
#         Uvec_flat = Uvec.reshape([self.B, -1])
        
#         self.dynamics(0.0, x0, Uvec_flat)
        
#         t_span = torch.linspace(
#             start=torch.tensor(0.0), end=torch.tensor(1.0), steps=self.steps).to(self.dummy.device)
        
        
        
#         soln = odeint(
#             func=lambda t, x:self.dynamics(t, x, Uvec_flat), 
#             y0=x0, 
#             t=t_span.to(self.dummy.device), 
#             method=self.solver
#         )
        
#         x_t = soln[-1]
        
#         return x_t
    
#     def test(self, dataset):
#         dataloader_test = DataLoader(dataset, batch_size=self.B, shuffle=False, drop_last=True)
        
#         soln_all = []
#         ground_all = []
        
#         ind_all = []
#         t_all = []
        
#         for b_i_n, b_t_n, b_obs in dataloader_test:
            
#             ind_all.append(b_i_n.data.cpu().numpy())
#             t_all.append(b_t_n.data.cpu().numpy())
            
#             b_i_n, b_t_n, b_obs = \
#                 b_i_n.to(self.dummy.device), \
#                 b_t_n.to(self.dummy.device), \
#                 b_obs.to(self.dummy.device)
            
#             ground_all.append(b_obs)
            
#             pred = self.solve(b_i_n, b_t_n)
#             soln_all.append(pred.squeeze())
#         #
        
#         soln_all = torch.cat(soln_all)
#         obs = torch.cat(ground_all).to(self.dummy.device)
        
#         rmse = torch.sqrt(torch.mean(torch.square(obs-soln_all)))
#         #nrmse = torch.sqrt(torch.mean(torch.square(obs-soln_all)))/ torch.linalg.norm(obs)
        
#         nrmse = torch.sqrt(torch.mean(torch.square(obs-soln_all)))/torch.sqrt(torch.square(obs).mean())
        
#         return rmse.item(), nrmse.item()
    
    
#     def eval_loss(self, b_i_n, b_t_n, b_obs, N):
# #         pred = self.solve(b_i_n, b_t_n)
# #         loss = torch.mean(torch.square(pred.squeeze()-b_obs))

#         Uvec = self._extract_Uvec(b_i_n)
#         Ureg = 0.5*torch.sum(torch.square(Uvec))

#         pred = self.solve(b_i_n, b_t_n)
        
#         loss = Ureg + 0.5*N*self.log_tau - 0.5*torch.exp(self.log_tau)*(N/self.B)* \
#             torch.sum(torch.square(pred.squeeze()-b_obs)) +\
#             self.gamma_prior.log_prob(torch.exp(self.log_tau))
    
#         return -loss.squeeze()
        
#     def train(self, dataset_train, dataset_test, max_epochs, learning_rate, perform_meters):
        
#         cprint('r', '@@@@@@@@@@  NODE Auto is being trained @@@@@@@@@@')
        
#         dataloader_train = DataLoader(dataset_train, batch_size=self.B, shuffle=True, drop_last=True)
               
#         optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
#         rmse_tr, nrmse_tr = self.test(dataset_train)
#         rmse_te, nrmse_te = self.test(dataset_test)
        
#         tau = torch.exp(self.log_tau).item()
        
#         perform_meters.add_by_epoch(rmse_tr, rmse_te, nrmse_tr, nrmse_te, tau)
#         perform_meters.add_by_step(rmse_tr, rmse_te, nrmse_tr, nrmse_te, tau)
        
#         steps = 0
        
#         for epoch in tqdm(range(max_epochs)):
            
#             for b, (b_i_n, b_t_n, b_obs) in enumerate(dataloader_train):
                
#                 t_start = time.time()
                
#                 b_i_n, b_t_n, b_obs = \
#                     b_i_n.to(self.dummy.device), \
#                     b_t_n.to(self.dummy.device), \
#                     b_obs.to(self.dummy.device)
                
# #                 pred = self.solve(b_i_n, b_t_n)
# #                 loss = torch.mean(torch.square(pred.squeeze()-b_obs))

#                 loss = self.eval_loss(b_i_n, b_t_n, b_obs, len(dataset_train))
                
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
                
#                 steps += 1
#                 ###### Test steps #######
#                 if perform_meters.test_interval > 0 and steps % perform_meters.test_interval == 0:
#                     rmse_tr, nrmse_tr = self.test(dataset_train)
#                     rmse_te, nrmse_te = self.test(dataset_test)

#                     perform_meters.add_by_step(rmse_tr, rmse_te, nrmse_tr, nrmse_te)
#                     perform_meters.save()
#                 ##########################
                
#                 t_end = time.time()
                
#                 perform_meters.logger.info('({}-batch takes {:.5f} seconds)'.format(b, t_end-t_start))
                
#             #
            
# #             rmse_tr, nrmse_tr = self.test(dataset_train)
# #             rmse_te, nrmse_te = self.test(dataset_test)
            
# #             perform_meters.add_by_epoch(rmse_tr, rmse_te, nrmse_tr, nrmse_te)
# #             perform_meters.save()

#             rmse_tr, nrmse_tr = self.test(dataset_train)
#             rmse_te, nrmse_te = self.test(dataset_test)
            
#             tau = torch.exp(self.log_tau).item()
            
#             perform_meters.add_by_epoch(rmse_tr, rmse_te, nrmse_tr, nrmse_te, tau)
#             perform_meters.save()
            
            

class NODE_auto(nn.Module):
    

    def __init__(self, nmod, nvec, R, nFF_init, nFF_dynamic, batch_size, steps, solver):
        
        super().__init__()
        
        self.nmod = nmod
        self.nvec = nvec
        assert self.nmod == len(self.nvec)
        
        self.R = R
        self.nFF_init = nFF_init
        self.nFF_dynamic = nFF_dynamic
        
        self.B = batch_size
        
        self.steps = steps
        self.solver = solver
        
        self.register_buffer('dummy', torch.tensor([]))
        
        self.register_buffer('a0', torch.tensor(1e-3))
        self.register_buffer('b0', torch.tensor(1e-3))
        self.gamma_prior = torch.distributions.Gamma(self.a0, self.b0)
        
#         self.Ulist = nn.ParameterList([nn.Parameter(torch.randn(self.nvec[i], R)) for i in range(self.nmod)])

        self.Ulist = nn.ParameterList([nn.Parameter(torch.rand(self.nvec[i], R)) for i in range(self.nmod)])
        
        self.init_model = RFF(num_ff=self.nFF_init, input_dim=self.R*self.nmod)
        self.dynamic_model = RFF(num_ff=self.nFF_dynamic, input_dim=self.R*self.nmod+2)
        
        self.log_tau = nn.Parameter(torch.tensor(0.0))
        
        self.scaleT = None
        
    def todev(self, device):
        self.to(device)
        for i in range(len(self.Ulist)):
            self.Ulist[i] = self.Ulist[i].to(device)
        self.init_model.to(device)
        self.dynamic_model.to(device)
        
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
    
    def dynamics(self, t, x, Uvec):
        # Uvec: N by (nmod*R)
        # x: m(t) N by 1
        t_coln_scaled = self.scaleT.view(self.B, 1)
        inputs = torch.hstack([t_coln_scaled*t, x, Uvec])
        x_t = self.dynamic_model.forward(inputs)*self.scaleT.view(self.B, 1)
        return x_t
        
    def solve(self, b_i_n, b_t_n):
        ####### Important #####
        self.scaleT = b_t_n
        #######################
        
        x0 = self.forward_init(b_i_n)
        Uvec = self._extract_Uvec(b_i_n)
        
        self.dynamics(0.0, x0, Uvec)
        
        t_span = torch.linspace(
            start=torch.tensor(0.0), end=torch.tensor(1.0), steps=self.steps).to(self.dummy.device)
        
        
        
        soln = odeint(
            func=lambda t, x:self.dynamics(t, x, Uvec), 
            y0=x0, 
            t=t_span.to(self.dummy.device), 
            method=self.solver
        )
        
        x_t = soln[-1]
        
        return x_t
    
    def test(self, dataset):
        dataloader_test = DataLoader(dataset, batch_size=self.B, shuffle=False, drop_last=True)
        
        soln_all = []
        ground_all = []
        
        ind_all = []
        t_all = []
        
        for b_i_n, b_t_n, b_obs in dataloader_test:
            
            ind_all.append(b_i_n.data.cpu().numpy())
            t_all.append(b_t_n.data.cpu().numpy())
            
            b_i_n, b_t_n, b_obs = \
                b_i_n.to(self.dummy.device), \
                b_t_n.to(self.dummy.device), \
                b_obs.to(self.dummy.device)
            
            ground_all.append(b_obs)
            
            pred = self.solve(b_i_n, b_t_n)
            soln_all.append(pred.squeeze())
        #
        
        soln_all = torch.cat(soln_all)
        obs = torch.cat(ground_all).to(self.dummy.device)
        
        rmse = torch.sqrt(torch.mean(torch.square(obs-soln_all)))
        #nrmse = torch.sqrt(torch.mean(torch.square(obs-soln_all)))/ torch.linalg.norm(obs)
        
        nrmse = torch.sqrt(torch.mean(torch.square(obs-soln_all)))/torch.sqrt(torch.square(obs).mean())
        
        return rmse.item(), nrmse.item()
    
    
    def eval_loss(self, b_i_n, b_t_n, b_obs, N):
#         pred = self.solve(b_i_n, b_t_n)
#         loss = torch.mean(torch.square(pred.squeeze()-b_obs))

        Uvec = self._extract_Uvec(b_i_n)
        Ureg = 0.5*torch.sum(torch.square(Uvec))

        pred = self.solve(b_i_n, b_t_n)
        
        loss = Ureg + 0.5*N*self.log_tau - 0.5*torch.exp(self.log_tau)*(N/self.B)* \
            torch.sum(torch.square(pred.squeeze()-b_obs)) +\
            self.gamma_prior.log_prob(torch.exp(self.log_tau))
    
        return -loss.squeeze()
        
    def train(self, dataset_train, dataset_test, max_epochs, learning_rate, perform_meters):
        
        cprint('r', '@@@@@@@@@@  NODE Auto is being trained @@@@@@@@@@')
        
        dataloader_train = DataLoader(dataset_train, batch_size=self.B, shuffle=True, drop_last=True)
               
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        rmse_tr, nrmse_tr = self.test(dataset_train)
        rmse_te, nrmse_te = self.test(dataset_test)
        
        tau = torch.exp(self.log_tau).item()
        
        perform_meters.add_by_epoch(rmse_tr, rmse_te, nrmse_tr, nrmse_te, tau)
        perform_meters.add_by_step(rmse_tr, rmse_te, nrmse_tr, nrmse_te, tau)
        
        steps = 0
        
        for epoch in tqdm(range(max_epochs)):
            
            for b, (b_i_n, b_t_n, b_obs) in enumerate(dataloader_train):
                
                t_start = time.time()
                
                b_i_n, b_t_n, b_obs = \
                    b_i_n.to(self.dummy.device), \
                    b_t_n.to(self.dummy.device), \
                    b_obs.to(self.dummy.device)
                
#                 pred = self.solve(b_i_n, b_t_n)
#                 loss = torch.mean(torch.square(pred.squeeze()-b_obs))

                loss = self.eval_loss(b_i_n, b_t_n, b_obs, len(dataset_train))
                
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
            
#             rmse_tr, nrmse_tr = self.test(dataset_train)
#             rmse_te, nrmse_te = self.test(dataset_test)
            
#             perform_meters.add_by_epoch(rmse_tr, rmse_te, nrmse_tr, nrmse_te)
#             perform_meters.save()

            rmse_tr, nrmse_tr = self.test(dataset_train)
            rmse_te, nrmse_te = self.test(dataset_test)
            
            tau = torch.exp(self.log_tau).item()
            
            perform_meters.add_by_epoch(rmse_tr, rmse_te, nrmse_tr, nrmse_te, tau)
            perform_meters.save()
            

        
        