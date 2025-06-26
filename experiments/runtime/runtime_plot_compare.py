import torch
import numpy as np
import matplotlib.pyplot as pl
import os
import time
from tqdm import tqdm


from TW_concurrent_lines import TWConcurrentLines, generate_trees_frames

# ------------------------------------------------------------------
# 1) Sliced-Wasserstein (SW) Implementation
# ------------------------------------------------------------------
def rand_projections(dim, num_projections=1000, device='cuda'):
    """Generate random directions on the unit sphere in `dim` dimensions."""
    projections = torch.randn((num_projections, dim), device=device)
    # Normalize each row to have unit norm
    projections = projections / torch.sqrt(torch.sum(projections**2, dim=1, keepdim=True))
    return projections

def one_dimensional_Wasserstein_prod(X, Y, theta, p=2):
    """
    Compute 1D Wasserstein distance (L^p) between X and Y 
    projected onto directions in theta.
    """
    X_proj = X @ theta.transpose(0, 1)  # shape (N, L)
    Y_proj = Y @ theta.transpose(0, 1)  # shape (N, L)

    # Sort the projections and compute absolute differences
    W = torch.abs(torch.sort(X_proj, dim=0)[0] - torch.sort(Y_proj, dim=0)[0])
    # L^p mean across samples, then average across projections
    W = torch.mean(W**p, dim=0)
    return torch.mean(W)

def SW_distance(X, Y, L=10, p=2, device="cuda"):
    """
    Main Sliced-Wasserstein distance function.
    Optionally includes a small warmup.
    """
    # --- Warmup (optional) ---
    # A small pass on a subset to "warm up" JIT or GPU kernels
    X_warm, Y_warm = X[:10], Y[:10]
    theta_warm = rand_projections(X_warm.size(1), num_projections=2, device=device)
    _ = one_dimensional_Wasserstein_prod(X_warm, Y_warm, theta_warm, p)

    # --- Actual SW Computation ---
    dim = X.size(1)
    theta = rand_projections(dim, num_projections=L, device=device)
    sw_val = one_dimensional_Wasserstein_prod(X, Y, theta, p)
    return sw_val

# ------------------------------------------------------------------
# 2) Timing/Memory Helpers for TW_Concurrent
# ------------------------------------------------------------------
def warmup_run_tw(TW_obj, N, d, ntree, nline):
    """Warms up by running a small forward pass twice."""
    theta, intercept = generate_trees_frames(ntree, nline, d, gen_mode="gaussian_raw")
    X = torch.rand(N, d, device="cuda")
    Y = torch.rand(N, d, device="cuda")
    TW_obj(X, Y, theta, intercept)

    theta, intercept = generate_trees_frames(ntree, nline, d, gen_mode="gaussian_raw")
    X = torch.rand(N, d, device="cuda")
    Y = torch.rand(N, d, device="cuda")
    TW_obj(X, Y, theta, intercept)

def run_tw(TW_obj, N, d, ntree, nline, gen_mode="gaussian_raw"):
    torch.cuda.reset_peak_memory_stats(device=None)
    X = torch.rand(N, d, device="cuda")
    Y = torch.rand(N, d, device="cuda")
    start = time.time()
    theta, intercept = generate_trees_frames(ntree, nline, d, gen_mode=gen_mode)
    tw_val = TW_obj(X, Y, theta, intercept)
    end = time.time()
    mem_usage = torch.cuda.max_memory_allocated(device=None)
    return end - start, mem_usage


# ------------------------------------------------------------------
# 3) Timing/Memory Helpers for SW
# ------------------------------------------------------------------
def warmup_run_sw(N, d, L=10):
    # Just do a dummy forward pass
    X = torch.rand(N, d, device="cuda")
    Y = torch.rand(N, d, device="cuda")
    _ = SW_distance(X, Y, L=L, p=2, device="cuda")

def run_sw(N, d, L=10):
    torch.cuda.reset_peak_memory_stats(device=None)
    X = torch.rand(N, d, device="cuda")
    Y = torch.rand(N, d, device="cuda")
    start = time.time()
    sw_val = SW_distance(X, Y, L=L, p=2, device="cuda")
    end = time.time()
    mem_usage = torch.cuda.max_memory_allocated(device=None)
    return end - start, mem_usage


# ------------------------------------------------------------------
# 4) Main script comparing SW (one line) vs. multiple TW ftypes
# ------------------------------------------------------------------
if __name__ == "__main__":

    # 5a) Define your experiment parameters
    nrun = 10                  # number of repetitions for averaging
    Ns = [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]  # x-axis: size of X, Y
    d = 3000                   # dimension (fixed)
    ntree = 2500               # some parameter for TW
    nline = 4                 # some parameter for TW

    # We'll collect results for: SW + 4 ftypes
    legends = ["SW",  "SpatialTSW", "CircularTSW", r"CircularTSW$_{r=0}$", "Db-TSW", r"Db-TSW$^\perp$", "TSW-SL"]
    metrics = ["SW", "pow", "circular", "circular_r0", "linear", "linear", "linear"]
    # metrics = ["SW", "pow"]
    colors = ["black", "blue", "orange", "red", "green", "purple", "teal"]

    # For storing runtime (seconds) and memory usage (bytes or MB)
    # shape = (num_metrics, len(Ns), nrun)
    runtimes = np.zeros((len(metrics), len(Ns), nrun))
    mems = np.zeros((len(metrics), len(Ns), nrun))

    # 5b) Loop over each metric
    for m_idx, metric in enumerate(metrics):
        print(f"--- Preparing metric: {legends[m_idx]} ---")
        if metric != 'SW':
            # ftype for TW
            if m_idx != 6:
                TW_obj = TWConcurrentLines(ntrees=ntree, ftype=metric, mass_division='distance_based')
            else:
                TW_obj = TWConcurrentLines(ntrees=ntree, ftype=metric, mass_division='uniform')
                
            TW_obj = torch.compile(TW_obj)  # compile once
            
        # 5c) Now loop over the different N values
        for n_idx, N in enumerate(Ns):
            # Create or warm up the object
            if metric == "SW":
                # Warmup for SW
                warmup_run_sw(N, d, L=ntree*nline)  # small warmup
            else:
                warmup_run_tw(TW_obj, N, d, ntree, nline)  # small warmup
                gen_mode = "gaussian_raw" if m_idx != 5 else "gaussian_orthogonal"
            for rep in range(nrun):
                if metric == "SW":
                    # measure SW
                    dur, mem_val = run_sw(N, d, L=ntree*nline)
                else:
                    dur, mem_val = run_tw(TW_obj, N, d, ntree, nline)

                runtimes[m_idx, n_idx, rep] = dur
                mems[m_idx, n_idx, rep] = mem_val

    # 5d) Post-processing: convert to ms and MB, then average
    runtimes_ms = runtimes * 1000.0
    avg_runtimes = np.mean(runtimes_ms, axis=2)
    std_runtimes = np.std(runtimes_ms, axis=2)

    mems_mb = mems / (1024.0 * 1024.0)
    avg_mems = np.mean(mems_mb, axis=2)
    std_mems = np.std(mems_mb, axis=2)

    # ------------------------------------------------------------------
    # 6) Plot: Runtime vs N
    # ------------------------------------------------------------------
    pl.figure(figsize=(10, 6))
    for m_idx, metric in enumerate(metrics):
        # pl.plot(Ns, avg_runtimes[m_idx], label=metric, color=colors[m_idx])
        pl.plot(
            Ns,
            avg_runtimes[m_idx],
            label=legends[m_idx],  # Use your defined legends here
            color=colors[m_idx]
        )


    # pl.title("Runtime Comparison: SW vs. TW_Concurrent (Different ftypes)", fontsize=16)
    pl.xlabel("Number of Supports $(n)$", fontsize=14)
    pl.ylabel("Runtime (ms)", fontsize=14)
    pl.legend(legends, fontsize=12)
    pl.grid(True)
    pl.tight_layout()

    # Save or show
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    pl.savefig(os.path.join(results_dir, "compare_runtime.png"))
    pl.savefig(os.path.join(results_dir, "compare_runtime.pdf"))
    pl.show()
    pl.clf()

    # ------------------------------------------------------------------
    # 7) Plot: Memory vs N
    # ------------------------------------------------------------------
    pl.figure(figsize=(10, 6))
    for m_idx, metric in enumerate(metrics):
        pl.plot(Ns, avg_mems[m_idx], label=metric, color=colors[m_idx])
        pl.fill_between(
            Ns,
            avg_mems[m_idx] - std_mems[m_idx],
            avg_mems[m_idx] + std_mems[m_idx],
            color=colors[m_idx],
            alpha=0.2
        )

    pl.title("Memory Usage Comparison: SW vs. TW_Concurrent (Different ftypes)", fontsize=16)
    pl.xlabel("Number of Supports (N)", fontsize=14)
    pl.ylabel("GPU Memory (MB)", fontsize=14)
    pl.legend(fontsize=12)
    pl.grid(True)
    pl.tight_layout()

    pl.savefig(os.path.join(results_dir, "compare_memory.png"))
    pl.savefig(os.path.join(results_dir, "compare_memory.pdf"))
    pl.show()
    pl.clf()
