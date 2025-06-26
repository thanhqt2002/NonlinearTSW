import torch
import torch.nn.functional as F
import numpy as np
import time
from tqdm.auto import trange

import sys
sys.path.append('../')
from methods import wd
from utils.func import set_seed

def vmf_pdf(x, mu, kappa):
    kappa = torch.tensor(kappa, dtype=torch.float32, device=x.device)
    mu = mu.to(x.device)
    C_d = kappa / (2 * np.pi * (torch.exp(kappa) - torch.exp(-kappa)))
    return C_d * torch.exp(kappa * torch.matmul(x, mu.T))

def run_exp(dataiter, d_func, d_args, mus, batch_size=200, n_steps=500, lr=1e-2, kappa=50, device='cuda', eval_loss=False, random_seed=0):
    set_seed(random_seed)
    L_w = np.zeros(n_steps//50)
    L_nll = np.zeros(n_steps//50)

    X0 = torch.randn((batch_size, 3), device=device)
    X0 = F.normalize(X0, p=2, dim=-1)
    X0.requires_grad_(True)

    optimizer = torch.optim.Adam([X0], lr=lr)
    loss_w = []

    tic = time.time()
    pbar = trange(n_steps)
    for i in pbar:
        optimizer.zero_grad()
        Xt = next(dataiter).to(device)
        sw = d_func(Xt, X0, **d_args)
        sw.backward()
        optimizer.step()
        X0.data /= torch.norm(X0.data, dim=1, keepdim=True)
        pbar.set_description(f"Loss: {sw.item():.4f}")
        
        if (i + 1) % 50 == 0:
            w = wd.g_wasserstein(X0, Xt, p=2, device=device)    
            log_probs = torch.stack([vmf_pdf(X0, mu, kappa) for mu in mus])
            log_sum_probs = torch.logsumexp(log_probs, dim=0) - torch.log(torch.tensor(len(mus), device=device))
            nll = -torch.sum(log_sum_probs).item()
            L_w[i//50] = w.item()
            L_nll[i//50] = nll

        if eval_loss:
            with torch.no_grad():
                w = wd.g_wasserstein(X0, Xt, p=2, device=device)    
                loss_w.append(w.item())
    if eval_loss:
        return loss_w
            
    pbar.close()
    t = time.time() - tic
    
    return t, L_nll, L_w
