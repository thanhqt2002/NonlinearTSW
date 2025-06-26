import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

def svd_orthogonalize(matrix):
    U, _, _ = torch.linalg.svd(matrix, full_matrices=False)
    return U

def generate_trees_frames(ntrees, nlines, d, mean=128, std=0.1, device='cuda', gen_mode='gaussian_raw'):    
    # random root as gaussian distribution with given mean and std
    assert gen_mode in ['gaussian_raw', 'gaussian_orthogonal'], "Invalid gen_mode"
    root = torch.randn(ntrees, 1, d, device=device) * std + mean
    intercept = root
    
    if gen_mode == 'gaussian_raw':
        theta = torch.randn(ntrees, nlines, d, device=device)
        theta = theta / torch.norm(theta, dim=-1, keepdim=True)
    elif gen_mode == 'gaussian_orthogonal':
        assert nlines <= d, "Support dim should be greater than or equal to number of lines to generate orthogonal lines"
        theta = torch.randn(ntrees, d, nlines, device=device)
        theta = svd_orthogonalize(theta)
        theta = theta.transpose(-2, -1)
    
    return theta, intercept

def generate_spherical_trees_frames(ntrees, nlines, d, device='cuda'):
    root = MultivariateNormal(torch.zeros(d), torch.eye(d)).sample((ntrees, 1)).to(device)
    root = root / torch.norm(root, dim=-1, keepdim=True)
    intercept = MultivariateNormal(torch.zeros(d), torch.eye(d)).sample((ntrees, nlines)).to(device)
    intercept_proj = intercept @ root.transpose(1, 2) #(ntrees, nlines, 1)
    intercept = intercept - intercept_proj @ root #(ntrees, nlines, d)
    intercept = F.normalize(intercept, p=2, dim=-1)

    return root, intercept