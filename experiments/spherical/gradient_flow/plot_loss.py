import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import os
import argparse
from itertools import cycle
from tqdm.auto import trange
import matplotlib.pyplot as plt
from train import run_exp

import sys
sys.path.append('../')
import utils.vmf as vmf_utils
from utils.func import set_seed
from methods import s3wd, sswd, stswd

def vmf_pdf(x, mu, kappa):
    kappa = torch.tensor(kappa, dtype=torch.float32, device=x.device)
    mu = mu.to(x.device)
    C_d = kappa / (2 * np.pi * (torch.exp(kappa) - torch.exp(-kappa)))
    return C_d * torch.exp(kappa * torch.matmul(x, mu.T))

def get_config(method): 
    if method == "stsw":
        d_func = stswd.stswd
        d_args = {'p': 2, 'ntrees': args.ntrees, 'nlines': args.nlines, 'device': device}
    elif method == "stsw_gen":
        d_func = stswd.stswd
        d_args = {'p': 2, 'ntrees': args.ntrees, 'nlines': args.nlines, 'delta': 10, 'device': device, 'type': 'generalized'}
    elif method == "ari_s3w":
        d_func = s3wd.ari_s3wd
        d_args = {'p': 2, 'n_projs': 1000, 'device': device, 'h': None, 'n_rotations': 30, 'pool_size': 1000}
    elif method == "s3w":
        d_func = s3wd.s3wd
        d_args = {'p': 2, 'n_projs': 1000, 'device': device, 'h': None}
    elif method == "ri_s3w_1":
        d_func = s3wd.ri_s3wd
        d_args = {'p': 2, 'n_projs': 1000, 'device': device, 'h': None, 'n_rotations': 1}
    elif method == "ri_s3w_5":
        d_func = s3wd.ri_s3wd
        d_args = {'p': 2, 'n_projs': 1000, 'device': device, 'h': None, 'n_rotations': 5}
    elif method == "ssw":
        d_func = sswd.sswd
        d_args = {'p': 2, 'num_projections': 1000, 'device': device}
    else:
        raise Exception(f"Loss function {args.d_func} is not supported")
    return d_func, d_args

if __name__ == "__main__":
    parser = argparse.ArgumentParser('gradient flow parameters')
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--ntrees', '-nt', type=int, default=200)
    parser.add_argument('--nlines', '-nl', type=int, default=5)
    parser.add_argument('--ntry', type=int, default=10)
    parser.add_argument('--epochs', '-ep', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--batch_size', type=int, default=2400)
    
    args = parser.parse_args()
    set_seed(2025)
    device = args.device
    os.makedirs('figures', exist_ok=True)

    phi = (1 + np.sqrt(5)) / 2
    vs = np.array([
        [-1,  phi,  0],
        [ 1,  phi,  0],
        [-1, -phi,  0],
        [ 1, -phi,  0],
        [ 0, -1,  phi],
        [ 0,  1,  phi],
        [ 0, -1, -phi],
        [ 0,  1, -phi],
        [ phi,  0, -1],
        [ phi,  0,  1],
        [-phi,  0, -1],
        [-phi,  0,  1]
    ])
    mus = F.normalize(torch.tensor(vs, dtype=torch.float), p=2, dim=-1)
    X = []
    kappa = 50 
    N = 200   
    for mu in mus:
        vmf = vmf_utils.rand_vmf(mu, kappa=kappa, N=N)
        X += list(vmf)
    X = torch.tensor(X, dtype=torch.float)
    Xt = X.clone().detach()
    trainloader = DataLoader(Xt, batch_size=args.batch_size, shuffle=True)
    dataiter = iter(cycle(trainloader))

    plt.figure(figsize=(8, 4))
    iterations = np.arange(args.epochs)
    methods = ['stsw', 'stsw_gen']
    for i, method in enumerate(methods):
        L = np.zeros((args.ntry, args.epochs))
        for i in range(args.ntry):
            d_func, d_args = get_config(method)
            loss = run_exp(dataiter, d_func, d_args, mus, batch_size=args.batch_size, n_steps=args.epochs, 
                                           lr=args.lr, kappa=kappa, device=device, eval_loss=True, random_seed=i)
            L[i] = np.log10(loss)

        m = np.mean(L, axis=0)
        s = np.std(L, axis=0)
        plt.plot(iterations, m, label=method, linewidth=2)
        plt.fill_between(iterations, m - s, m + s, alpha=0.5)
    
    # save results
    output_file = "figures/w2_loss.png"
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel(r'$\log_{10}(W_2)$', fontsize=13)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(output_file)
    plt.close()