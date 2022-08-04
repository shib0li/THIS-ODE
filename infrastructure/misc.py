import os, sys
import pickle
import logging
import numpy as np

# import torch
# import torch.optim as optim
# import torch.nn as nn
# import torch.nn.functional as F


def create_path(path): 
    try:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        #
        print("Directory '%s' created successfully" % (path))
    except OSError as error:
        print("Directory '%s' can not be created" % (path))
    #
    
def get_logger(logpath, displaying=True, saving=True, debug=False, append=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        if append:
            info_file_handler = logging.FileHandler(logpath, mode="a")
        else:
            info_file_handler = logging.FileHandler(logpath, mode="w+")
        #
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    return logger
    
def cprint(color, text, **kwargs):
    if color[0] == '*':
        pre_code = '1;'
        color = color[1:]
    else:
        pre_code = ''
    code = {
        'a': '30',
        'r': '31',
        'g': '32',
        'y': '33',
        'b': '34',
        'p': '35',
        'c': '36',
        'w': '37'
    }
    print("\x1b[%s%sm%s\x1b[0m" % (pre_code, code[color], text), **kwargs)
    sys.stdout.flush()
    
    
class PerformMeters(object):
    
    def __init__(self, save_path, logger=None, test_interval=0):

        
        
        self.epochs_rmse_tr = []
        self.epochs_rmse_te = []
        self.epochs_nrmse_tr = []
        self.epochs_nrmse_te = []
        
        self.steps_rmse_tr = []
        self.steps_rmse_te = []
        self.steps_nrmse_tr = []
        self.steps_nrmse_te = []
        
        self.cnt_epochs = 0
        self.cnt_steps = 0
        
        self.save_path = save_path
        self.logger = logger
        
        self.test_interval = test_interval
        
    def add_by_epoch(self, rmse_tr, rmse_te, nrmse_tr, nrmse_te, tau=0.0):
        
        self.epochs_rmse_tr.append(rmse_tr)
        self.epochs_rmse_te.append(rmse_te)
        self.epochs_nrmse_tr.append(nrmse_tr)
        self.epochs_nrmse_te.append(nrmse_te)
        
        if self.logger is not None:    
            self.logger.info('=========================================')
            self.logger.info('                 Epoch{}               '.format(self.cnt_epochs))
            self.logger.info('=========================================')          
            self.logger.info('  # rmse_tr={:.6f},  nrmse_tr={:.6f}'.format(rmse_tr, nrmse_tr))
            self.logger.info('  # rmse_te={:.6f},  nrmse_te={:.6f}'.format(rmse_te, nrmse_te))
            self.logger.info('  # tau={:.6f}'.format(tau))
        #
            
        self.cnt_epochs += 1
        
    def add_by_step(self, rmse_tr, rmse_te, nrmse_tr, nrmse_te, tau=0.0):
        
        self.steps_rmse_tr.append(rmse_tr)
        self.steps_rmse_te.append(rmse_te)
        self.steps_nrmse_tr.append(nrmse_tr)
        self.steps_nrmse_te.append(nrmse_te)
        
        if self.logger is not None:         
            self.logger.info('---------------  Steps{} ---------------'.format(self.cnt_steps))
            self.logger.info('  - rmse_tr={:.6f},  nrmse_tr={:.6f}'.format(rmse_tr, nrmse_tr))
            self.logger.info('  - rmse_te={:.6f},  nrmse_te={:.6f}'.format(rmse_te, nrmse_te))
            self.logger.info('  - tau={:.6f}'.format(tau))
        #
            
        self.cnt_steps += self.test_interval
            

    def save(self,):
        
        res = {}
        
        res['epochs_rmse_tr'] = np.array(self.epochs_rmse_tr)
        res['epochs_rmse_te'] = np.array(self.epochs_rmse_te)
        res['epochs_nrmse_tr'] = np.array(self.epochs_nrmse_tr)
        res['epochs_nrmse_te'] = np.array(self.epochs_nrmse_te)
        
        res['steps_rmse_tr'] = np.array(self.steps_rmse_tr)
        res['steps_rmse_te'] = np.array(self.steps_rmse_te)
        res['steps_nrmse_tr'] = np.array(self.steps_nrmse_tr)
        res['steps_nrmse_te'] = np.array(self.steps_nrmse_te)
        

        with open(os.path.join(self.save_path, 'error.pickle'), 'wb') as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #

