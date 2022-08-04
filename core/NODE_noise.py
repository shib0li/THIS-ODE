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
        
        
class NODE_noise(nn.Module):
    def __init__(self, nmod, nvec, R, nFF, batch_size, solver='dopri5', int_steps=2):

        super().__init__()
        
        self.nmod = nmod
        self.nvec = nvec
        assert self.nmod == len(self.nvec)
        
        self.M = nFF
        self.R = R
        #self.d = batch_size
        self.B = batch_size
        self.solver = solver
        self.int_steps = int_steps
        
        self.register_buffer('dummy', torch.tensor([]))
        
        self.model = FFGP(num_ff=self.M, input_dim=self.nmod*R+2) #(flat_vi, m, t)
        
        # embeddings of latent ODE
        #self.Ulist = [torch.rand([self.nvec[i], self.R]) for i in range(self.nmod)]
        
        if self.nmod == 2:
            Uinit = [np.random.rand(self.nvec[0],R), np.random.rand(self.nvec[1],R)]
        elif self.nmod == 3:
            Uinit = [np.random.rand(self.nvec[0],R), np.random.rand(self.nvec[1],R), np.random.rand(self.nvec[2],R)]

        self.Ulist = [torch.tensor(U.copy()) for U in Uinit]
        
        #cprint('r',  self.Ulist)
        
#         self.log_tau = torch.tensor([0.0], requires_grad=True)
        self.log_tau = torch.tensor([0.0])

        self.Uvec = None # B by nmod by R embeddings for current batch
        self.beta = None # B by 1
        self.scaleT = None
        
        
        # Gamma prior
        self.register_buffer('a0', torch.tensor(1e-3))
        self.register_buffer('b0', torch.tensor(1e-3))
        self.gamma_prior = torch.distributions.Gamma(self.a0, self.b0)
        
        #print(self.gamma_prior.log_prob(torch.exp(self.log_tau)))

    def todev(self, device):
        self.to(device)
        for i in range(len(self.Ulist)):
            self.Ulist[i] = self.Ulist[i].to(device)
        self.model.todev(device)
        self.log_tau = self.log_tau.to(device)

    def extra_repr(self,)->str:
        info = ''
        info += 'Ulist:\n'
        for i in range(len(self.Ulist)):
            info += '(U{}).shape = {}\n'.format(i, self.Ulist[i].shape)
        info += '(U)device = {}'.format(self.Ulist[0].device)
        return info
    
    def _extract_Uvec(self, b_i_n):    
        Uvec = []
        for i_n in b_i_n:
            v_i_n = []
            for i in range(self.nmod):
                v_i_n.append(self.Ulist[i][i_n[i]])
            #
            v_i_n = torch.vstack(v_i_n)
            Uvec.append(v_i_n.unsqueeze(0))
        #
        Uvec = torch.cat(Uvec, dim=0) 
        beta = torch.prod(Uvec, dim=1).sum(1).view([-1,1]) # CP, B by 1
        return Uvec, beta
    
    def pack_theta(self, b_i_n):
        self.Uvec, self.beta  = self._extract_Uvec(b_i_n)
        theta = torch.cat([
            self.beta.view(-1),    # ODE init
            self.Uvec.view(-1),    # tensor embeddings
            self.model.S.view(-1), # frequencies
            self.model.w.view(-1), # weights
            self.log_tau.view(-1),
        ])
        self.theta_size = theta.numel()
        return theta
    
    def unpack_theta(self, theta):
        idx = 0
        beta = theta[idx:idx+self.B].view(self.B,1)
        idx = idx + self.B
        #
        Uvec = theta[idx:idx+self.B*self.nmod*self.R].view(self.B, self.nmod, self.R)
        idx = idx + self.B * self.nmod * self.R
        #
        S = theta[idx:idx + self.model.num_ff*self.model.input_dim].view(self.model.num_ff, self.model.input_dim)
        idx = idx + self.model.num_ff * self.model.input_dim
        #
        w = theta[idx:idx + 2*self.model.num_ff].view(2*self.model.num_ff, 1)
        idx = idx + 2*self.model.num_ff
        # 
        log_tau = theta[idx:idx+1]
        idx = idx + 1
        torch._assert(idx == theta.numel(), 'unpacking incorrectly')
        return (beta, Uvec, S, w, log_tau)

    
    def forward_scaled(self, t, x, theta):
        beta, Uvec, S, w, log_tau = self.unpack_theta(theta)
        self.Uvec = Uvec
        self.model.S = S
        self.model.w = w  
        
        t_coln_scaled = self.scaleT.view(self.B, 1) # b_t_n.T B by 1
        inputs = torch.hstack([t_coln_scaled*t, x, Uvec.flatten(start_dim=1,end_dim=-1)]) # B by (1+1+R*nmod)
  
        x_t = self.model.forward(inputs)*self.scaleT.view(self.B, 1)
        
        return x_t
    
    #params including both x & g (i.e., sensitivity); states is a vector
    def ODE_dynamics_scaled(self, t, states, params_theta): # B by (1+theta.size)
        x = states[:,0].view([-1,1]) # B by 1
        g = states[:,1:].view([self.B,-1]) # B by theta.size
        dx = self.forward_scaled(t, x, params_theta) # B by 1
        jac = F.jacobian(self.forward_scaled, (t, x, params_theta))
        #print(jac[1].squeeze().shape)
        #print(jac[2].squeeze().shape)
        dg = torch.matmul(jac[1].squeeze(), g) + jac[2].squeeze()
        dtotal = torch.hstack([dx, dg])
        return dtotal

    def unpack_batch_augment_states(self, augment_states):
        idx = 0
        #
        soln_x = augment_states[:,idx:idx+1].view([-1,1])
        idx = idx + 1
        #
        grad_beta = augment_states[:,idx:idx+self.B]
        idx = idx + self.B
        #
        grad_Uvec_flat = augment_states[:,idx:idx+self.B*self.nmod*self.R]
        idx = idx + self.B*self.nmod*self.R
        #
        grad_eta = augment_states[:,idx:]
        idx = idx + grad_eta.shape[1]
        #
        #print(1+self.B+self.Bself.nmod*self.R+self.model.num_ff*self.model.input_dim+2*self.model.num_ff)
        ### Dont't forget the log_tau
        torch._assert(
            idx == 1+self.B+self.B*self.nmod*self.R+self.model.num_ff*self.model.input_dim+2*self.model.num_ff+1, 
            'augment_states unpacking incorrectly'
        )
        
        return (soln_x, grad_beta, grad_Uvec_flat, grad_eta)
        
        
    
    def batch_grad_scaled(self, b_i_n, b_t_n, b_obs):
        params_theta = self.pack_theta(b_i_n)
        
        x0 = self.beta
        #cprint('r',x0)
        g0 = torch.zeros(self.B, self.theta_size).to(self.dummy.device)
        g0[:self.B, :self.B] = torch.eye(self.B).to(self.dummy.device)
        
        ####### Important #####
        self.scaleT = b_t_n
        #######################

        def CP(Uvec):
            return torch.prod(Uvec, dim=1).sum(1).view([-1,1])
        #
        
        
        #t_start = time.time()
        init_state = torch.hstack([x0, g0])

        t_span = torch.linspace(start=torch.tensor(0.0), end=torch.tensor(1.0),steps=self.int_steps)

        soln = odeint(
            func=lambda t, states:self.ODE_dynamics_scaled(t, states, params_theta), 
            y0=init_state, 
            t=t_span.to(self.dummy.device), 
            method=self.solver
        )
        #print(soln.shape)
        
        #cprint('r', self.theta_size)
        
        soln_x, grad_beta, grad_Uvec_flat, grad_eta = self.unpack_batch_augment_states(soln[-1])
        
        #print(soln_x.shape)
        #print(grad_beta.shape)
        #print(grad_Uvec_flat.shape)
        #print(grad_eta.shape)
        
        def CP_beta(Uvec):
            return torch.prod(Uvec, dim=1).sum(1).view([-1,1])
        #
 
        dUvec_beta = F.jacobian(CP_beta, self.Uvec)

        ##stupid implementation
        #dUvec_x2 = []
        #for b in range(self.B):
        #    d_beta_b = grad_beta[b,b]
        #    d_Uvec_b = dUvec_beta[b,:]
        #    dUvec_x2.append(d_beta_b*d_Uvec_b)
        ##
        #dUvec_x2 = torch.cat(dUvec_x2, dim=0)
        
        # vectorized implementation, this is the derivative of Uvec through beta, dx/dUvec=(dx/dbeta)(dbeta/duvec)
        #dUvec_x_beta = (
        #    (torch.diag(grad_beta).view([-1,1]))*\
        #    (dUvec_beta.reshape([self.B,-1]))
        #).reshape([self.B, self.B, self.nmod, self.R])
        
        dUvec_x_beta = (
            (torch.diag(grad_beta).view([-1,1]))*\
            (dUvec_beta.reshape([self.B,-1]))
        ) # deruvatuve if Uvec through beta

        dUvec_x_dynamics = grad_Uvec_flat # deruvatuve if Uvec through dynamics
        
        #dUvec_total = dUvec_x_beta + dUvec_x_dynamics
        
        #cprint('p', time.time()-t_start)
        
        #t_start = time.time()
        
        # add prior to embeddings
        def log_prior(Uvec):
            return -0.5*torch.sum(torch.square(Uvec), dim=(1,2), keepdim=True).squeeze(1)
        
        dUvec_prior = F.jacobian(log_prior, self.Uvec).reshape([self.B, self.B*self.nmod*self.R])
        
        dUvec_total = dUvec_x_beta + dUvec_x_dynamics + dUvec_prior
        
        #cprint('g', time.time()-t_start)
        
        
        # tau in the likelihood term
        b_obs = b_obs.view([-1,1])
        
        
        #def log_loss(log_tau):
        #    return torch.exp(log_tau)*(soln_x - b_obs).square()
        
        #dlog_tau_llh2 = F.jacobian(log_loss, self.log_tau)
        
        #cprint('r', dlog_tau_llh2)
        
        
        #t_start = time.time()
        dlog_tau_llh = torch.exp(self.log_tau)*(soln_x - b_obs).square()
        
        #cprint('b', dlog_tau_llh)
        #cprint('g', (dlog_tau_llh.squeeze()-dlog_tau_llh2.squeeze()).square().sum())
        
        log_tau_copy = self.log_tau.clone().detach().to(self.dummy.device)
        log_tau_copy.requires_grad = True
        
        log_tau_prior = self.gamma_prior.log_prob(torch.exp(log_tau_copy))
        dlog_tau_prior = torch.autograd.grad(log_tau_prior, log_tau_copy)[0] *\
            (torch.ones([self.B, 1]).to(self.dummy.device))
        
        dlog_tau_total = dlog_tau_llh + dlog_tau_prior
        #dlog_tau_total = dlog_tau_llh
        
        #cprint('r', time.time()-t_start)
        
        #cprint('r', grad_eta)
        
        # put dlog_tau into grad_eta
        grad_eta[:,-1] = dlog_tau_total.squeeze()
        
        #cprint('b', grad_eta)

        grad_x = torch.exp(self.log_tau)*(soln_x - b_obs).unsqueeze(1)
        
        
        dtheta = torch.hstack([grad_beta, dUvec_total, grad_eta]) # insert back
        dtheta = dtheta.unsqueeze(1)

#         #b_obs = b_obs.view([-1,1])
        grad_x = torch.exp(self.log_tau)*(soln_x - b_obs).unsqueeze(1)

        #print(grad_x.shape, dtheta.shape)
        grad_total = torch.bmm(grad_x, dtheta).sum(0).squeeze(0)/self.B
        #print(grad_total.shape)
        
        #cprint('r', grad_total)
        
        return grad_total, params_theta
        
        
        
#     def debug(self, dataset):
#         dataloader_train = DataLoader(dataset, batch_size=self.B, shuffle=True, drop_last=True)
        
#         b_i_n, b_t_n, b_obs = next(iter(dataloader_train))
        
#         b_i_n, b_t_n, b_obs = \
#             b_i_n.to(self.dummy.device), \
#             b_t_n.to(self.dummy.device), \
#             b_obs.to(self.dummy.device)
        
# #         print(b_i_n.shape)
# #         print(b_t_n.shape)
# #         print(b_obs.shape)
        
# #         self.batch_grad_scaled(b_i_n, b_t_n, b_obs)


#         grad_total, params_theta = self.batch_grad_scaled(b_i_n, b_t_n, b_obs)
    
# #         cprint('r', grad_total.shape)
# #         cprint('b', params_theta.shape)
        

    def stack_Ulist(self, Ulist):
        stacked = []
        for U in Ulist:
            stacked.append(U)
        #
        return torch.vstack(stacked)

    def unstack_Ulist(self, Ustack):
        idx = 0
        Ulist = []
        for v in self.nvec:
            U = Ustack[idx:idx+v, :]
            Ulist.append(U)
            idx += v
        #
        return Ulist
  
    def aggregate_grad_Uvec_test(self, dUvec_flat, b_i_n):
        
        dUvec = dUvec_flat.view(self.B, self.nmod, self.R)
        
        Ustack = self.stack_Ulist(self.Ulist)
        
        def _wrap_Uvec(Ustack):
            Ulist = self.unstack_Ulist(Ustack)  
            Uvec = []
            for i_n in b_i_n:
                v_i_n = []
                for i in range(self.nmod):
                    v_i_n.append(Ulist[i][i_n[i]])
                #
                v_i_n = torch.vstack(v_i_n)
                Uvec.append(v_i_n.unsqueeze(0))
            #
            Uvec = torch.cat(Uvec, dim=0) 
            return Uvec
        #
        
        dUstack = F.jacobian(_wrap_Uvec, Ustack)
        
        grad = torch.tensordot(dUvec, dUstack, dims=3)
        dUlist = self.unstack_Ulist(grad)
        
        return dUlist
        
    def aggregate_grad_Uvec(self, dUvec_flat, b_i_n):
        
        dUvec = dUvec_flat.view(self.B, self.nmod, self.R)
        
        dUlist = [torch.zeros([self.nvec[i], self.R]).to(self.dummy.device) for i in range(self.nmod)]
        
        #print(b_i_n.shape)
        
        for b in range(self.B):
            i_n = b_i_n[b]
            for v in range(self.nmod):
                dUlist[v][i_n[v]] += dUvec[b, v]
            #
        #
        
        return dUlist
     
#     def update_model_params(self, params):
#         _, _, S, w = self.unpack_theta(params)  
#         self.model.S = S
#         self.model.w = w
        
#     def update_Ulist(self, dUlist, learning_rate):
#         for n in range(self.nmod):
#             self.Ulist[n] = self.Ulist[n] - learning_rate*dUlist[n]
#         #
        
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
            
            ####### Important #####
            self.scaleT = b_t_n
            #######################
            
            params_theta = self.pack_theta(b_i_n)
        
            t_span = torch.linspace(start=torch.tensor(0.0), end=torch.tensor(1.0), steps=self.int_steps)
            
            soln = odeint(
                func=lambda t, states:self.forward_scaled(t, states, params_theta), 
                y0=self.beta, 
                t=t_span.to(self.dummy.device), 
                method=self.solver
            )
            
            soln_all.append(soln[-1])
            
        #
        
        soln_all = torch.vstack(soln_all).squeeze()
        obs = torch.cat(ground_all).to(self.dummy.device)
        
        #_, _, obs = next(iter(DataLoader(dataset, batch_size=len(dataset), shuffle=False)))
        #obs = obs.to(self.dummy.device)
        rmse = torch.sqrt(torch.mean(torch.square(obs-soln_all)))
        #nrmse = torch.sqrt(torch.mean(torch.square(obs-soln_all)))/ torch.linalg.norm(obs)
        
        nrmse = torch.sqrt(torch.mean(torch.square(obs-soln_all)))/ torch.sqrt(torch.square(obs).mean())
        
        return rmse.item(), nrmse.item()
    
#     def test_err_U(self, dataset):

#         Ustack = torch.vstack(self.Ulist).data.cpu().numpy()
#         Ustack_ground = np.vstack(dataset.Uinit)
        
#         rmse = np.sqrt(np.mean(np.square(Ustack-Ustack_ground)))
#         nrmse = rmse/np.linalg.norm(Ustack_ground)
        
#         return rmse, nrmse
    
    def flat_Ulist(self, Ulist):
        Ustack = []
        for U in Ulist:
            Ustack.append(U)
        #
        Ustack = torch.vstack(Ustack)
        Uflat = Ustack.view(-1)
        return Uflat
    
    def update_model(self, new_params):
        
        idx = 0
        
        for n in range(self.nmod):
            v = self.nvec[n]
            Uflat = new_params[idx:idx+v*self.R]
            U = Uflat.reshape([v, self.R])
            idx = idx + v*self.R
            self.Ulist[n] = U
        #
        
        S = new_params[idx:idx+self.model.num_ff*self.model.input_dim].view([self.model.num_ff, self.model.input_dim])
        idx = idx + self.model.num_ff*self.model.input_dim
        self.model.S = S
        
        w = new_params[idx:idx+2*self.model.num_ff].view([2*self.model.num_ff,1])
        idx = idx + 2*self.model.num_ff
        self.model.w = w
        
        # update tau
        new_log_tau = new_params[-1]
        idx = idx + 1
        self.log_tau = new_log_tau
        
        torch._assert(
            idx == sum(self.nvec)*self.R + self.model.num_ff*self.model.input_dim + 2*self.model.num_ff + 1, 
            'update params incorrectly'
        )
        
        

    def train(self, dataset_train, dataset_test, max_epochs, learning_rate, perform_meters):
        
        cprint('r', '@@@@@@@@@@  NODE with noise is being trained @@@@@@@@@@')
        
        dataloader_train = DataLoader(dataset_train, batch_size=self.B, shuffle=True, drop_last=True)
        
        # Adam parameters
        beta_1 = 0.9
        beta_2 = 0.999
        eps_stable= 1e-8
        m_t = 0
        v_t = 0
        step = 0
        
        rmse_tr, nrmse_tr = self.test(dataset_train)
        rmse_te, nrmse_te = self.test(dataset_test)
        tau = torch.exp(self.log_tau).item()
        
        perform_meters.add_by_epoch(rmse_tr, rmse_te, nrmse_tr, nrmse_te, tau)
        perform_meters.add_by_step(rmse_tr, rmse_te, nrmse_tr, nrmse_te, tau)

        
        for epoch in tqdm(range(max_epochs)):
        
            for b, (b_i_n, b_t_n, b_obs) in enumerate(dataloader_train):
                
                t_start = time.time()

                b_i_n, b_t_n, b_obs = \
                    b_i_n.to(self.dummy.device), \
                    b_t_n.to(self.dummy.device), \
                    b_obs.to(self.dummy.device)

 
                grad_total, params_theta = self.batch_grad_scaled(b_i_n, b_t_n, b_obs)

                dUvec_flat = grad_total[self.B: self.B+self.B*self.nmod*self.R]
                
                dUlist = self.aggregate_grad_Uvec(dUvec_flat, b_i_n)
                #dUlist_test = self.aggregate_grad_Uvec_test(dUvec_flat, b_i_n)
                #
                #for dU, dUtest in zip(dUlist, dUlist_test):
                #    err = (dU-dUtest).square().sum()
                #    cprint('a', err)
                
                dUflat = self.flat_Ulist(dUlist)
                deta = grad_total[self.B+self.B*self.nmod*self.R:]
                
                grad = torch.cat([dUflat, deta])
                
                #run adam to update 
                step = step + 1
                m_t = beta_1*m_t + (1.0-beta_1)*grad
                v_t = beta_2*v_t + (1.0-beta_2)*(grad*grad)
                m_cap = m_t/(1-(beta_1**step))
                v_cap = v_t/(1-(beta_2**step))
                
                adam_grad = m_cap/(torch.sqrt(v_cap) + eps_stable)
                
                #print(adam_grad)
                
                ###### Test steps #######
                if perform_meters.test_interval > 0 and step % perform_meters.test_interval == 0:
                    rmse_tr, nrmse_tr = self.test(dataset_train)
                    rmse_te, nrmse_te = self.test(dataset_test)
                    tau = torch.exp(self.log_tau).item()

                    perform_meters.add_by_step(rmse_tr, rmse_te, nrmse_tr, nrmse_te, tau)
                    perform_meters.save()
                ##########################
                
                params_eta = params_theta[self.B+self.B*self.nmod*self.R:]
                params_Uflat = self.flat_Ulist(self.Ulist)
                
                params_interest = torch.cat([params_Uflat, params_eta])
                
                new_params_interets = params_interest - learning_rate*adam_grad
                
                self.update_model(new_params_interets)
                
                t_end = time.time()
                
                perform_meters.logger.info('({}-batch takes {:.5f} seconds)'.format(b, t_end-t_start))
                
            #
            
            rmse_tr, nrmse_tr = self.test(dataset_train)
            rmse_te, nrmse_te = self.test(dataset_test)
            tau = torch.exp(self.log_tau).item()
            
            perform_meters.add_by_epoch(rmse_tr, rmse_te, nrmse_tr, nrmse_te, tau)
            perform_meters.save()

        #
        
        
        