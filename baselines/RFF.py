import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

class RFF(nn.Module):
    
    def __init__(self, num_ff, input_dim):
        
        super().__init__()
        
        self.num_ff = num_ff
        self.input_dim = input_dim
        
        self.linear1 = torch.nn.Linear(in_features=self.input_dim, out_features=self.num_ff)
        self.linear2 = torch.nn.Linear(in_features=self.num_ff, out_features=1)
        
    def forward(self, X):
        h = self.linear1(X)
        h = torch.tanh(h)
        y = self.linear2(h)
        return y
    
# model = RFF(num_ff=100, input_dim=20)