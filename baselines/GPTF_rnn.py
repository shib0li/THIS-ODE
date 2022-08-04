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

class GPTF_rnn:
    #Uinit: init U
    #m: #pseudo inputs
    #ind: entry indices
    #y: observed tensor entries
    #B: batch-size
    def __init__(self, ind, y, time_points, Uinit, m, B, device, jitter=1e-4):
        self.device = device
        self.Uinit = Uinit
        self.m = m
        
        self.y = torch.tensor(y.reshape([y.size,1]), device=self.device)

        self.ind = ind
        self.time_points = torch.tensor(time_points.reshape([time_points.size,1]), device=self.device)
        
        self.B = B
        self.nmod = len(self.Uinit)
        self.U = [torch.tensor(self.Uinit[k].copy(), device=self.device, requires_grad=True) for k in range(self.nmod)]
        
        
#         #======================================================================#
#         self.nvec = nvec
#         self.R = R
#         self.U = [torch.rand([self.nvec[k], self.R]) for k in range(self.nmod)]
#         #
        
        #dim. of pseudo input
        self.d = 0
        for k in range(self.nmod):
            self.d = self.d + Uinit[k].shape[1]
        #init mu, L, Z
        #Zinit = self.init_pseudo_inputs()
        Zinit = np.random.rand(self.m, self.d)
        self.Z = torch.tensor(Zinit, device=self.device, requires_grad=True)
        self.N = y.size
        #variational posterior
        self.mu = torch.tensor(np.zeros([m,1]), device=self.device, requires_grad=True)
        self.L = torch.tensor(np.eye(m), device=self.device, requires_grad=True)
        #kernel parameters
        #self.log_amp = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.log_ls = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.log_tau = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.jitter = torch.tensor(jitter, device=self.device)
        self.kernel = KernelRBF(self.jitter)
        
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
        input_emb = torch.cat([self.U[k][self.ind[sub_ind, k],:] for k in range(self.nmod)], 1)
        #time_log = torch.log(self.time_points[sub_ind] + 1e-4)
#         input_emb = torch.cat([input_emb, time_log], 1)
        input_emb = torch.cat([input_emb], 1)
        y_sub = self.y[sub_ind]
        
        Kmm = self.kernel.matrix(self.Z, torch.exp(self.log_ls))
        Kmn = self.kernel.cross(self.Z, input_emb, torch.exp(self.log_ls))
        Knm = Kmn.T
        Ltril = torch.tril(self.L)
        KnmKmmInv = torch.linalg.solve(Kmm, Kmn).T
        KnmKmmInvL = torch.matmul(KnmKmmInv, Ltril)
        tau = torch.exp(self.log_tau)
        ls = torch.exp(self.log_ls)
        
        trans_log_prob = self.trans_prior()

        hh_expt = torch.matmul(Ltril, Ltril.T) + torch.matmul(self.mu, self.mu.T)
        ELBO = -0.5*torch.logdet(Kmm) - 0.5*torch.trace(torch.linalg.solve(Kmm, hh_expt)) + 0.5*torch.sum(torch.log(torch.square(torch.diag(Ltril)))) \
                + 0.5*self.N*self.log_tau - 0.5*tau*self.N/self.B*torch.sum(torch.square(y_sub - torch.matmul(KnmKmmInv, self.mu))) \
                - 0.5*tau*( self.N*(1.0+self.jitter) - self.N/self.B*torch.sum(KnmKmmInv*Knm) + self.N/self.B*torch.sum(torch.square(KnmKmmInvL)) ) \
                + 0.5*self.m - 0.5*self.N*torch.log(2.0*torch.tensor(np.pi, device=self.device)) +\
        trans_log_prob
               

        return -torch.squeeze(ELBO)

    def init_pseudo_inputs(self):
        part = [None for k in range(self.nmod)]
        for k in range(self.nmod):
            part[k] = self.Uinit[k][self.ind[:,k], :]
        X = np.hstack(part)

        X = X[np.random.randint(X.shape[0], size=self.m * 100), :]
        print(X.shape)

        kmeans = KMeans(n_clusters=self.m, random_state=0).fit(X)
        return kmeans.cluster_centers_


    def pred(self, test_ind, test_time):
        inputs = torch.cat([self.U[k][test_ind[:,k],:]  for k in range(self.nmod)], 1)
#         inputs = torch.cat([inputs, test_time],1)
        inputs = torch.cat([inputs],1)
        #test_time_log = torch.log(test_time + 1e-4)
        #inputs = torch.cat([inputs, test_time_log],1)
        Knm = self.kernel.cross(inputs, self.Z, torch.exp(self.log_ls))
        Kmm = self.kernel.matrix(self.Z, torch.exp(self.log_ls))
        pred_mean = torch.matmul(Knm, torch.linalg.solve(Kmm, self.mu))
        return pred_mean


    def _callback(self, ind_te, yte, time_te):
        with torch.no_grad():
            ls = torch.exp(self.log_ls)
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
#             print('ls=%.5f, tau=%.5f, train_err = %.5f, test_err=%.5f' %\
#                  (ls, tau, err_tr, err_te))
#             with open('sparse_gptf_res.txt','a') as f:
#                 f.write('%g '%err_te)
                
            return rmse_tr.item(), rmse_te.item(), nrmse_tr.item(), nrmse_te.item(), tau.item()
            
    
    def train(self, ind_te, yte, time_te, lr, max_epochs, perform_meters):
        
        cprint('b', '@@@@@@@@@@  GPTF RNN is being trained @@@@@@@@@@')
        
        #yte = torch.tensor(yte.reshape([yte.size,1]), device=self.device)
        yte = yte.reshape([-1, 1])
        #time_te = torch.tensor(time_te.reshape([time_te.size, 1]), device=self.device)
        time_te = time_te.reshape([-1,1])
        paras = self.U + [self.Z, self.mu, self.L, self.log_ls, self.log_tau, self.W, self.b, self.log_v]
        
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
            
                
#         self._callback(ind_te, yte, time_te)
#         print(self.mu)
#         print(self.L)

#         print('U0 diff = %g'%( np.mean(np.abs(self.Uinit[0] - self.U[0].detach().numpy())) ))
#         print('U1 diff = %g'%( np.mean(np.abs(self.Uinit[1] - self.U[1].detach().numpy())) ))
#         print('U0')
#         print(self.U[0])
#         print('U1')
#         print(self.U[1])