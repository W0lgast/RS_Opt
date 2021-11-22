"""
Custom losses for training
"""

# ---------------------------------------------------------------------

import torch
import numpy as np

# ---------------------------------------------------------------------

def euclidean_loss(y_true, y_pred):
    res = torch.sqrt(torch.sum(torch.square(y_pred - y_true), axis=-1))
    return res

def cyclical_mae_rad(y_true, y_pred):
    ret = torch.mean(torch.min(torch.abs(y_pred - y_true),
                                torch.min(torch.abs(y_pred - y_true + 2*np.pi),
                                          torch.abs(y_pred - y_true - 2*np.pi))),
                      axis=-1)
    return ret

mse_torch = torch.nn.MSELoss(reduction='none')

def mse(y_true, y_pred):

    #ret = mse_torch(torch.squeeze(y_true), torch.squeeze(y_pred))
    #ret = mse_torch(y_true, y_pred)
    ret = torch.subtract(torch.squeeze(y_true), torch.squeeze(y_pred))
    ret = torch.square(ret)


    #ret = ret.float()
    #ret = torch.squeeze(ret)
    return ret

l1 = torch.nn.L1Loss(reduction='none')

def mae(y_true, y_pred):
    ret = torch.squeeze(l1(y_true, y_pred))
    return ret
