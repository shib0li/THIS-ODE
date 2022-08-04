import numpy as np
import torch
import torch.distributions as distributions
from torch.optim import Adam
from torch.optim import LBFGS
from sklearn.cluster import KMeans
import random
import pickle
import fire
from tqdm.auto import tqdm, trange

from baselines.kernels import KernelRBF, KernelARD
from data.real_events import EventData

from torch.utils.data import Dataset, DataLoader

from infrastructure.misc import *
from infrastructure.configs import *


np.random.seed(0)
torch.manual_seed(0)
random.seed(0)



#sparse variational GP for tensor factorization, the same performace with the TensorFlow version

class CPTF_rnn:
    #Uinit: init U
    #m: #pseudo inputs
    #ind: entry indices
    #y: observed tensor entries
    #B: batch-size
    def __init__(self, ind, y, Uinit, B, device, jitter=1e-3):
        self.device = device
        self.Uinit = Uinit
        self.y = torch.tensor(y.reshape([y.size,1]), device=self.device)
        self.ind = ind
        self.B = B
        self.nmod = len(self.Uinit)
        self.U = [torch.tensor(self.Uinit[k].copy(), device=self.device, requires_grad=True) for k in range(self.nmod)]
        self.N = y.size
        #variational posterior
        self.log_tau = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        R = self.U[0].shape[1]
        self.W = torch.zeros([R,R], device=self.device, requires_grad=True)
        torch.nn.init.xavier_normal_(self.W)
        self.b = torch.zeros(R, device=self.device, requires_grad=True)
        self.log_v = torch.tensor(0.0, device=self.device, requires_grad=True)
#         cprint('r', self.W)
#         cprint('r', self.b)
#         cprint('r', self.log_v)

    def trans_prior(self,):
        T = self.U[-1].float()
        
        trans_mu = torch.sigmoid(torch.matmul(T, self.W) + self.b)
        I = torch.eye(T.shape[1]).to(self.device)
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
        
        
    #batch neg ELBO
    def nELBO_batch(self, sub_ind):
        U_sub = [self.U[k][self.ind[sub_ind, k],:] for k in range(self.nmod)]
        y_sub = self.y[sub_ind]

        U_prod = U_sub[0]
        Ureg = -0.5*torch.sum(torch.square(self.U[0]))
        for k in range(1, self.nmod):
            U_prod = U_prod * U_sub[k]
            Ureg = Ureg - 0.5*torch.sum(torch.square(self.U[k]))
        pred = torch.sum(U_prod, 1, keepdim=True)
        
        trans_log_prob = self.trans_prior()
        

        L = Ureg + 0.5*self.N*self.log_tau -0.5*torch.exp(self.log_tau)*self.N/self.B*torch.sum(torch.square(y_sub - pred)) +\
        trans_log_prob
 
        return -torch.squeeze(L)

    def pred(self, test_ind):
        inputs = [self.U[k][test_ind[:,k],:]  for k in range(self.nmod)]
        U_prod = inputs[0]
        for k in range(1, self.nmod):
            U_prod = U_prod * inputs[k]
        pred = torch.sum(U_prod, 1, keepdim=True)
        return pred


    def _callback(self, ind_te, yte):
        with torch.no_grad():
            tau = torch.exp(self.log_tau)
            pred_mean = self.pred(ind_te)
            #err_tr = torch.sqrt(torch.mean(torch.square(pred_mu_tr-ytr)))
            pred_tr = self.pred(self.ind)
            rmse_tr = torch.sqrt(torch.mean(torch.square(pred_tr - self.y)))
            rmse_te = torch.sqrt(torch.mean(torch.square(pred_mean - yte)))
#             nrmse_tr = torch.sqrt(torch.mean(torch.square(pred_tr - self.y)))/ torch.linalg.norm(self.y)
#             nrmse_te = torch.sqrt(torch.mean(torch.square(pred_mean - yte)))/ torch.linalg.norm(yte)

            nrmse_tr = torch.sqrt(torch.mean(torch.square(pred_tr - self.y)))/ torch.sqrt(torch.square(self.y).mean())
            nrmse_te = torch.sqrt(torch.mean(torch.square(pred_mean - yte)))/ torch.sqrt(torch.square(yte).mean())
        
#             print('ls=%.5f, tau=%.5f, train_err = %.5f, test_err=%.5f' %\
#                  (ls, tau, err_tr, err_te))
#             with open('sparse_gptf_res.txt','a') as f:
#                 f.write('%g '%err_te)
                
            return rmse_tr.item(), rmse_te.item(), nrmse_tr.item(), nrmse_te.item(), tau.item()
            
    
    def train(self, ind_te, yte, lr, max_epochs, perform_meters):
        
        cprint('p', '@@@@@@@@@@  CPTF RNN is being trained @@@@@@@@@@')
        
        #yte = torch.tensor(yte.reshape([yte.size,1]), device=self.device)
        yte = yte.reshape([-1, 1])
        #time_te = torch.tensor(time_te.reshape([time_te.size, 1]), device=self.device)
        paras = self.U + [self.log_tau, self.W, self.b, self.log_v]
        
        rmse_tr, rmse_te, nrmse_tr, nrmse_te, tau = self._callback(ind_te, yte)
        
        perform_meters.add_by_epoch(rmse_tr, rmse_te, nrmse_tr, nrmse_te, tau)
        perform_meters.add_by_step(rmse_tr, rmse_te, nrmse_tr, nrmse_te, tau)
        
        minimizer = Adam(paras, lr=lr)
        
        steps = 0
        
        for epoch in trange(max_epochs):
            curr = 0
            while curr < self.N:
                batch_ind = np.random.choice(self.N, self.B, replace=False)
                minimizer.zero_grad()
                loss = self.nELBO_batch(batch_ind)
                loss.backward(retain_graph=True)
                minimizer.step()
                curr = curr + self.B
            #print('epoch %d done'%epoch)
#             if epoch%5 == 0:
#                 self._callback(ind_te, yte, time_te)

                steps += 1
                ###### Test steps #######
                if perform_meters.test_interval > 0 and steps % perform_meters.test_interval == 0:
                    rmse_tr, rmse_te, nrmse_tr, nrmse_te, tau = self._callback(ind_te, yte)
                    perform_meters.add_by_step(rmse_tr, rmse_te, nrmse_tr, nrmse_te, tau)
                    perform_meters.save()
                ##########################

            rmse_tr, rmse_te, nrmse_tr, nrmse_te, tau = self._callback(ind_te, yte)
    
            perform_meters.add_by_epoch(rmse_tr, rmse_te, nrmse_tr, nrmse_te, tau)
            perform_meters.save()
            
                
#         self._callback(ind_te, yte, time_te)
#         print(self.mu)
#         print(self.L)

#         print('U0 diff = %g'%( np.mean(np.abs(self.Uinit[0] - self.U[0].detach().numpy())) ))
#         print('U1 diff = %g'%( np.mean(np.abs(self.Uinit[1] - self.U[1].detach().numpy())) ))
#         print('U0')
#         print(self.U[0])
#         print('U1')
#         print(self.U[1])