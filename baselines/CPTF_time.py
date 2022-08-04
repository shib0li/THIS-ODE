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

from torchcubicspline import(natural_cubic_spline_coeffs, 
                             NaturalCubicSpline)

from infrastructure.misc import *
from infrastructure.configs import *


np.random.seed(0)
torch.manual_seed(0)
random.seed(0)



#sparse variational GP for tensor factorization, the same performace with the TensorFlow version

class CPTF_time:
    #Uinit: init U
    #m: #pseudo inputs
    #ind: entry indices
    #y: observed tensor entries
    #B: batch-size
    def __init__(self, ind, y, time_points, Uinit, B, device, jitter=1e-3, t_max=10.0):
        self.device = device
        self.Uinit = Uinit
        self.y = torch.tensor(y.reshape([y.size,1]), device=self.device)
        self.ind = ind
        self.time_points = torch.tensor(time_points.reshape([time_points.size,1]), device=self.device)
        self.B = B
        self.nmod = len(self.Uinit)
        self.U = [torch.tensor(self.Uinit[k].copy(), device=self.device, requires_grad=True) for k in range(self.nmod)]
        self.N = y.size
        #variational posterior
        self.log_tau = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        spline_knots = 100
        spline_order = 3
        
        self.spline_params_list = []
        self.t_span = torch.linspace(0,t_max,spline_knots+1)
        #cprint('r', self.t_span)

        for i in range(spline_order+1):
            spline_param = torch.zeros([spline_knots,1], device=self.device, requires_grad=True)
            torch.nn.init.xavier_normal_(spline_param)
            self.spline_params_list.append(spline_param)
            #cprint('r', spline_param.shape)
        #
        
        
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

        L = Ureg + 0.5*self.N*self.log_tau -0.5*torch.exp(self.log_tau)*self.N/self.B*torch.sum(torch.square(y_sub - pred))
 
        return -torch.squeeze(L)

    def pred(self, test_ind, test_time):
        inputs = [self.U[k][test_ind[:,k],:]  for k in range(self.nmod)]
        U_prod = inputs[0]
        for k in range(1, self.nmod):
            U_prod = U_prod * inputs[k]
        CP = torch.sum(U_prod, 1, keepdim=True)
        
        #cprint('b', CP.shape)
        
        spline_coeff = tuple([self.t_span] + self.spline_params_list)
        #print(spline_coeff)
        
        spline_model = NaturalCubicSpline(spline_coeff)
        
        #print(test_time.shape)
        
        lamda_t = spline_model.evaluate(test_time.squeeze())
        
        pred = CP * lamda_t
        
        #print(pred.shape)
        
        return pred


    def _callback(self, ind_te, yte, time_te):
        with torch.no_grad():
            tau = torch.exp(self.log_tau)
            pred_mean = self.pred(ind_te, time_te)
            #err_tr = torch.sqrt(torch.mean(torch.square(pred_mu_tr-ytr)))
            pred_tr = self.pred(self.ind, self.time_points)
            rmse_tr = torch.sqrt(torch.mean(torch.square(pred_tr - self.y)))
            rmse_te = torch.sqrt(torch.mean(torch.square(pred_mean - yte)))
#             nrmse_tr = torch.sqrt(torch.mean(torch.square(pred_tr - self.y)))/ torch.linalg.norm(self.y)
#             nrmse_te = torch.sqrt(torch.mean(torch.square(pred_mean - yte)))/ torch.linalg.norm(yte)

            nrmse_tr = torch.sqrt(torch.mean(torch.square(pred_tr - self.y)))/ torch.sqrt(torch.square(self.y).mean())
            nrmse_te = torch.sqrt(torch.mean(torch.square(pred_mean - yte)))/ torch.sqrt(torch.square(yte).mean())
                
            return rmse_tr.item(), rmse_te.item(), nrmse_tr.item(), nrmse_te.item(), tau.item()
            
    
    def train(self, ind_te, yte, time_te, lr, max_epochs, perform_meters):
        
        cprint('r', '@@@@@@@@@@  CPTF Time is being trained @@@@@@@@@@')
        

        #yte = torch.tensor(yte.reshape([yte.size,1]), device=self.device)
        yte = yte.reshape([-1, 1])
        #time_te = torch.tensor(time_te.reshape([time_te.size, 1]), device=self.device)
        paras = self.U + [self.log_tau] + self.spline_params_list
        #print(paras)
        
        rmse_tr, rmse_te, nrmse_tr, nrmse_te, tau = self._callback(ind_te, yte, time_te)
        
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
                    rmse_tr, rmse_te, nrmse_tr, nrmse_te, tau = self._callback(ind_te, yte, time_te)
                    perform_meters.add_by_step(rmse_tr, rmse_te, nrmse_tr, nrmse_te, tau)
                    perform_meters.save()
                ##########################

            rmse_tr, rmse_te, nrmse_tr, nrmse_te, tau = self._callback(ind_te, yte, time_te)
    
            perform_meters.add_by_epoch(rmse_tr, rmse_te, nrmse_tr, nrmse_te, tau)
            perform_meters.save()
            
            
                
