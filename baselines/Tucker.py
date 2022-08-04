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

def kronecker_product_einsum_batched(A: torch.Tensor, B: torch.Tensor): 
    """ 
    Batched Version of Kronecker Products 
    :param A: has shape (b, a, c) 
    :param B: has shape (b, k, p) 
    :return: (b, ak, cp) 
    """ 
    assert A.dim() == 3 and B.dim() == 3 

    res = torch.einsum('bac,bkp->bakcp', A, B).view(A.size(0), 
                                                    A.size(1)*B.size(1), 
                                                    A.size(2)*B.size(2) 
                                                    ) 
    return res 

#sparse variational GP for tensor factorization, the same performace with the TensorFlow version

class Tucker:
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
        
#         R = self.U[0].shape[1]
#         self.W = torch.zeros([R,R], device=self.device, requires_grad=True)
#         torch.nn.init.xavier_normal_(self.W)
#         self.b = torch.zeros(R, device=self.device, requires_grad=True)
#         self.log_v = torch.tensor(0.0, device=self.device, requires_grad=True)
# #         cprint('r', self.W)
# #         cprint('r', self.b)
# #         cprint('r', self.log_v)
        
        R = self.U[0].shape[1]
        self.gamma_size = np.power(R, self.nmod) # R_U for CP, (R_U)^K for tucker
        self.gamma = torch.ones((self.gamma_size,1),requires_grad=True,device=self.device, dtype=torch.double)
        
        #cprint('r', self.gamma)
        
        # prior of noise
        self.v = 1. # prior varience of embedding (scaler)
        self.v_time = 1. # prior varience of time-mode embedding (scaler)
        self.all_modes = [i for i in range(self.nmod)]

        self.batch_product = kronecker_product_einsum_batched
        
    def moment_produc_U(self,ind):
        # computhe first and second moments of 
        # \kronecker_prod_{k \in given modes} u_k -Tucker
        # \Hadmard_prod_{k \in given modes} u_k -CP
        last_mode = self.all_modes[-1]
        # print(ind.shape)
        E_z = self.U[last_mode][ind[:,last_mode]] # N*R_u
        E_z = E_z.unsqueeze(2)
        
        #cprint('r', E_z.shape)

        for mode in reversed(self.all_modes[:-1]):
            E_u = self.U[mode][ind[:,mode]] # N*R_u
            E_u = E_u.unsqueeze(2)
            E_z = self.batch_product(E_z,E_u)

        return E_z
        
    #batch neg ELBO
#     def nELBO_batch(self, sub_ind):
#         U_sub = [self.U[k][self.ind[sub_ind, k],:] for k in range(self.nmod)]
#         y_sub = self.y[sub_ind]

#         U_prod = U_sub[0]
#         Ureg = -0.5*torch.sum(torch.square(self.U[0]))
#         for k in range(1, self.nmod):
#             U_prod = U_prod * U_sub[k]
#             Ureg = Ureg - 0.5*torch.sum(torch.square(self.U[k]))
#         pred = torch.sum(U_prod, 1, keepdim=True)
        
#         trans_log_prob = self.trans_prior()
        

#         L = Ureg + 0.5*self.N*self.log_tau -0.5*torch.exp(self.log_tau)*self.N/self.B*torch.sum(torch.square(y_sub - pred)) +\
#         trans_log_prob
 
#         return -torch.squeeze(L)

    def nELBO_batch(self, batch_ind):

        B_size = batch_ind.shape[0]
        ind_x = self.ind[batch_ind]

        y_pred = self.pred(ind_x).squeeze() 
        y_true = self.y[batch_ind].squeeze()

        tau = torch.exp(self.log_tau)
        ELBO = 0.5*self.N*self.log_tau \
            - 0.5*tau*self.N/B_size*torch.sum(torch.square(y_true - y_pred))

        return -torch.squeeze(ELBO)

#     def pred(self, test_ind):
#         inputs = [self.U[k][test_ind[:,k],:]  for k in range(self.nmod)]
#         U_prod = inputs[0]
#         for k in range(1, self.nmod):
#             U_prod = U_prod * inputs[k]
#         pred = torch.sum(U_prod, 1, keepdim=True)
#         return pred

    def pred(self,x_ind):
        E_z = self.moment_produc_U(x_ind).squeeze(-1) # N* gamma_size
        pred_y = torch.mm(E_z,self.gamma) # N*1
        return pred_y

    def _callback(self, ind_te, yte):
        with torch.no_grad():
            tau = torch.exp(self.log_tau)
            pred_mean = self.pred(ind_te)
            #err_tr = torch.sqrt(torch.mean(torch.square(pred_mu_tr-ytr)))
            pred_tr = self.pred(self.ind)
            rmse_tr = torch.sqrt(torch.mean(torch.square(pred_tr - self.y)))
            rmse_te = torch.sqrt(torch.mean(torch.square(pred_mean - yte)))
            nrmse_tr = torch.sqrt(torch.mean(torch.square(pred_tr - self.y)))/ torch.linalg.norm(self.y)
            nrmse_te = torch.sqrt(torch.mean(torch.square(pred_mean - yte)))/ torch.linalg.norm(yte)
#             print('ls=%.5f, tau=%.5f, train_err = %.5f, test_err=%.5f' %\
#                  (ls, tau, err_tr, err_te))
#             with open('sparse_gptf_res.txt','a') as f:
#                 f.write('%g '%err_te)
                
            return rmse_tr.item(), rmse_te.item(), nrmse_tr.item(), nrmse_te.item(), tau.item()
            
    
    def train(self, ind_te, yte, lr, max_epochs, perform_meters):
        
        cprint('p', '@@@@@@@@@@  PTucker is being trained @@@@@@@@@@')
        
        #yte = torch.tensor(yte.reshape([yte.size,1]), device=self.device)
        yte = yte.reshape([-1, 1])
        #time_te = torch.tensor(time_te.reshape([time_te.size, 1]), device=self.device)
        paras = self.U + [self.gamma, self.log_tau]
        
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
                #print(loss)
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
            
                
