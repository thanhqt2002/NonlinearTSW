import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import os
import argparse
import time
from itertools import cycle
from scipy.stats import gaussian_kde
from tqdm.auto import trange
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from train import run_exp

import sys
sys.path.append('../')
import utils.vmf as vmf_utils
import utils.plot as plot_utils
from utils.func import set_seed
from methods import s3wd, sswd, stswd

def plot_result(X, out_path):
    k = gaussian_kde(X.T)
    fig, ax = plt.subplots(1, 1, figsize=(10,10), subplot_kw={'projection': "mollweide"})
    plot_utils.projection_mollweide(lambda x: k.pdf(x.T), ax)
    plt.savefig(out_path)
    plt.close(fig)

def get_run_name(args):
    if "stsw" in args.d_func:
        return f"{args.d_func}-nt_{args.ntrees}-nl_{args.nlines}-delta_{args.delta}"
    return f"{args.d_func}"
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser('gradient flow parameters')
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--d_func', '-d', type=str, default="stsw")
    parser.add_argument('--ntry', type=int, default=5)
    parser.add_argument('--ntrees', '-nt', type=int, default=200)
    parser.add_argument('--nlines', '-nl', type=int, default=5)
    parser.add_argument('--delta', type=float, default=2)
    parser.add_argument('--epochs', '-ep', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--batch_size', type=int, default=2400)
    parser.add_argument('--seed', type=int, default=0)
    
    args = parser.parse_args()
    set_seed(args.seed)
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
    X = np.array(X)
    X = torch.tensor(X, dtype=torch.float)
    Xt = X.clone().detach()
    trainloader = DataLoader(Xt, batch_size=args.batch_size, shuffle=True)
    dataiter = iter(cycle(trainloader))

    if args.d_func == "stsw":
        d_func = stswd.stswd
        d_args = {'p': 2, 'ntrees': args.ntrees, 'nlines': args.nlines, 'device': device}
    elif args.d_func == "stsw_gen":
        d_func = stswd.stswd
        d_args = {'p': 2, 'ntrees': args.ntrees, 'nlines': args.nlines, 'delta': args.delta, 'device': device, 'type': 'generalized'}
    elif args.d_func == "ari_s3w":
        d_func = s3wd.ari_s3wd
        d_args = {'p': 2, 'n_projs': 1000, 'device': device, 'h': None, 'n_rotations': 30, 'pool_size': 1000}
    elif args.d_func == "s3w":
        d_func = s3wd.s3wd
        d_args = {'p': 2, 'n_projs': 1000, 'device': device, 'h': None}
    elif args.d_func == "ri_s3w_1":
        d_func = s3wd.ri_s3wd
        d_args = {'p': 2, 'n_projs': 1000, 'device': device, 'h': None, 'n_rotations': 1}
    elif args.d_func == "ri_s3w_5":
        d_func = s3wd.ri_s3wd
        d_args = {'p': 2, 'n_projs': 1000, 'device': device, 'h': None, 'n_rotations': 5}
    elif args.d_func == "ssw":
        d_func = sswd.sswd
        d_args = {'p': 2, 'num_projections': 1000, 'device': device}
    else:
        raise Exception(f"Loss function {args.d_func} is not supported")
    
    results = []
    for s in range(args.ntry):
        results.append(run_exp(dataiter, d_func, d_args, mus, batch_size=args.batch_size, n_steps=args.epochs, 
                                           lr=args.lr, kappa=kappa, device=device, random_seed=s))

    runtimes = [r[0] for r in results]
    nll= np.array([r[1] for r in results])
    w = np.array([r[2] for r in results])
    log_wd = np.log10(w)
    iter_nll = np.mean(nll, axis=0)
    iter_log_wd = np.mean(log_wd, axis=0)

    res_df = pd.read_csv("results.csv") if os.path.exists("results.csv") else pd.DataFrame()
    to_add = pd.DataFrame({
        "run_name": [get_run_name(args)],
        "log_wd_iter_50": [iter_log_wd[0]],
        "log_wd_iter_100": [iter_log_wd[1]],
        "log_wd_iter_150": [iter_log_wd[2]],
        "log_wd_iter_200": [iter_log_wd[3]],
        "log_wd_iter_250": [iter_log_wd[4]],
        "log_wd_iter_300": [iter_log_wd[5]],
        "mean_nll_iter_50": [iter_nll[0]],
        "mean_nll_iter_100": [iter_nll[1]],
        "mean_nll_iter_150": [iter_nll[2]],
        "mean_nll_iter_200": [iter_nll[3]],
        "mean_nll_iter_250": [iter_nll[4]],
        "mean_nll_iter_300": [iter_nll[5]],
        "mean_runtime": [np.mean(runtimes)],
        "lr": [args.lr],
        "seed": [args.seed],
    })
    res_df = pd.concat([res_df, to_add], ignore_index=True)
    res_df.to_csv("results.csv", index=False)
    
    # best = np.argmin(w)
    # X0_best = results[best][0]
    # plot_result(X0_best, f"figures/{get_run_name(args)}.png")