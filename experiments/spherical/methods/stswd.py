import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from tsw import SphericalTSW
from utils.s3w import unif_hypersphere

def stswd(X, Y, ntrees=250, nlines=4, p=2, delta=2, device='cuda', ftype='normal'):
    TW_obj = SphericalTSW(ntrees=ntrees, nlines=nlines, p=p, delta=delta, device=device, ftype=ftype)
    stswd = TW_obj(X, Y)
    return stswd

def stswd_unif(X, ntrees=250, nlines=4, p=2, delta=2, device='cuda', ftype='normal'):
    Y_unif = unif_hypersphere(X.shape, device=X.device) 
    stswd_unif = stswd(X, Y_unif, ntrees, nlines, p, delta, device, ftype)
    return stswd_unif