import torch

class KernelRBF:
    def __init__(self, jitter):
        self.jitter = jitter
        
    def matrix(self, X, ls):
        K = self.cross(X, X, ls)
        Ijit = self.jitter*torch.eye(X.shape[0]).to(K.device)
        K = K + Ijit
        return K
        
        
    def cross(self, X1, X2, ls):
        norm1 = torch.reshape(torch.sum(torch.square(X1), dim=1), [-1,1])
        norm2 = torch.reshape(torch.sum(torch.square(X2), dim=1), [1,-1])
        K = norm1-2.0*torch.matmul(X1,X2.T) + norm2
        K = torch.exp(-1.0*K/ls)
        return K
    
class KernelARD:
    def __init__(self, jitter):
        self.jitter = jitter
        
    def matrix(self, X, ls):
        K = self.cross(X, X, ls)
        Ijit = self.jitter*torch.eye(X.shape[0]).to(K.device)
        K = K + Ijit
        return K
        
        
    def cross(self, X1, X2, ls):
        ls_sqrt = torch.sqrt(ls)
        X1 = X1/ls_sqrt
        X2 = X2/ls_sqrt
        norm1 = torch.reshape(torch.sum(torch.square(X1), dim=1), [-1,1])
        norm2 = torch.reshape(torch.sum(torch.square(X2), dim=1), [1,-1])
        K = norm1-2.0*torch.matmul(X1,X2.T) + norm2
        #K = amp*torch.exp(-1.0*K)
        K = torch.exp(-1.0*K)
        return K