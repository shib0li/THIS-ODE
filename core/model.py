import numpy as np
import torch
import random

import torch.nn as nn

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

torch.set_default_tensor_type(torch.DoubleTensor)

class FFGP(nn.Module):
    def __init__(self, num_ff, input_dim):
        super().__init__()
        
        self.num_ff = num_ff
        self.input_dim = input_dim

        self.S = torch.rand(self.num_ff, self.input_dim)
        self.w = torch.rand(2*self.num_ff, 1)
        torch.nn.init.xavier_normal_(self.w)
        torch.nn.init.xavier_uniform_(self.S)
        
    def forward(self, X):
        # N by d
        H = torch.matmul(X, self.S.T)
#         V = torch.cat([torch.cos(H), torch.sin(H)], 1)
        V = torch.cat([torch.tanh(H), torch.tanh(H)], 1)
        y = torch.matmul(V, self.w)
        return y
    
    def extra_repr(self,)->str:
        info = ''
        info += 'freq components (S).shape = {}\n'.format(self.S.shape)
        info += 'weights (w).shape = {}\n'.format(self.w.shape)
        info += '(FFGP)device = {}'.format(self.w.device)
        return info
    
    def todev(self, device):
        self.S = self.S.to(device)
        self.w = self.w.to(device)