

class Config(object):
        
    def parse(self, kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
        #
        
        print('=================================')
        print('*', self.config_name)
        print('---------------------------------')
        
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print('-', k, ':', getattr(self, k))
        print('=================================')
        
        
    def __str__(self,):
        
        buff = ""
        buff += '=================================\n'
        buff += ('*'+self.config_name+'\n')
        buff += '---------------------------------\n'
        
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                buff += ('-' + str(k) + ':' + str(getattr(self, k))+'\n')
            #
        #
        buff += '=================================\n'
        
        return buff    
    

# class NodeExpConfig(Config):
    
#     domain = None
#     device = 'cpu'
    
#     pt_est = None
    
#     fold = None
    
#     batch_size = 100
#     max_epochs = 500
#     test_interval = 0
#     learning_rate = 1e-3
    
#     R = 3
    
#     int_steps = 2
#     nFF = 50
#     solver = 'dopri5'
    
    
#     verbose = False
    
#     def __init__(self,):
#         super(NodeExpConfig, self).__init__()
#         self.config_name = 'NODE Config'

class NodeExpConfig(Config):
    
    domain = None
    device = 'cpu'
    
    est = None
    
    fold = None
    
    batch_size = 100
    max_epochs = 500
    test_interval = 0
    learning_rate = 1e-3
    
    R = 3
    
    int_steps = 2
    nFF = 50
    solver = 'dopri5'
    
    
    verbose = False
    
    def __init__(self,):
        super(NodeExpConfig, self).__init__()
        self.config_name = 'NODE Config'
        
class CPExpConfig(Config):
    
    domain = None
    device = 'cpu'
    
    fold = None
    
    trans='linear'
    
    batch_size = 100
    max_epochs = 500
    test_interval = 0
    learning_rate = 1e-3
    
    R = 3
    m = 50 # number of inducing points
    
    verbose = False
    
    def __init__(self,):
        super(CPExpConfig, self).__init__()
        self.config_name = 'CPTF Config'
        
class GPExpConfig(Config):
    
    domain = None
    device = 'cpu'
    
    fold = None
    
    trans='linear'
    
    batch_size = 100
    max_epochs = 500
    test_interval = 0
    learning_rate = 1e-3
    
    R = 3
    m = 50 # number of inducing points
    
    verbose = False
    
    def __init__(self,):
        super(GPExpConfig, self).__init__()
        self.config_name = 'GPTF Config'
        
class TuckerExpConfig(Config):
    
    domain = None
    device = 'cpu'
    
    fold = None
    
    batch_size = 100
    max_epochs = 500
    test_interval = 0
    learning_rate = 1e-3
    
    R = 3
    m = 50 # number of inducing points
    
    verbose = False
    
    def __init__(self,):
        super(TuckerExpConfig, self).__init__()
        self.config_name = 'CPTF Config'
        
class NeuralExpConfig(Config):
    
    domain = None
    device = 'cpu'
    
    fold = None
    
    trans='linear'
    
    batch_size = 100
    max_epochs = 500
    test_interval = 0
    learning_rate = 1e-3
    
    R = 3

#     int_steps = 100
    nFF = 51
#     solver = 'rk4'
    
    verbose = False
    
    def __init__(self,):
        super(NeuralExpConfig, self).__init__()
        self.config_name = 'Neural Init Config'
        
        
# class NinitExpConfig(Config):
    
#     domain = None
#     device = 'cpu'
    
#     fold = None
    
#     bin_time=False
    
#     batch_size = 100
#     max_epochs = 500
#     test_interval = 0
#     learning_rate = 1e-3
    
#     R = 3
    
#     int_steps = 100
#     nFF = 50
#     solver = 'rk4'
    
#     verbose = False
    
#     def __init__(self,):
#         super(NinitExpConfig, self).__init__()
#         self.config_name = 'Neural Init Config'
        
        
        
        
     

        