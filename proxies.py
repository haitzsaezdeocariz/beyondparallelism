import numpy as np
import random
import torch
import torch.nn.functional as F
from typing import Optional

def _get_learned_head_weights(attn_module) -> np.ndarray:
    """
    Extract a non-normalized importance for each head from out_proj:
    Reshape W_out (E × E) → (E, H, head_dim), take L2 over (E, head_dim).
    E stands for the embedding dimension, H for the number of heads, and d for the head dimension.
    Returns an (H,) array.
    """
    W = attn_module.out_proj.weight.detach().cpu().numpy()  # (E, E)
    H = attn_module.num_heads
    d = attn_module.head_dim
    Wh = W.reshape(attn_module.embed_dim, H, d)             # (E, H, head_dim)
    # per-head raw norm, proxy to importance
    imp = np.linalg.norm(Wh, axis=(0,2))                    # (H,)
    return imp

def monte_carlo_mixing_time(
    attn, 
    num_simulations: int, 
    head_weights: Optional[np.ndarray] = None,
    max_steps: int = 100
) -> float:
    """
    Monte Carlo simulation to estimate the mixing time of a Markov chain defined by
    either a single attention matrix or multiple head matrices combined via weights.

    Args:
        attn (np.ndarray): Attention matrix of shape (T, T) or multiple matrices of shape (H, T, T).
        num_simulations (int): Number of random walks to average per start state.
        head_weights (np.ndarray, optional): Weights for combining multiple heads.
        max_steps (int): Maximum steps per simulation before stopping.

    Returns:
        float: Estimated expected mixing time averaged over all start states.
    """
    # Handle both single-matrix and multi-head inputs
    arr = np.array(attn)
    H, T, _ = arr.shape
    # Normalize or default head weights
    if head_weights is None:
        w = np.ones(H) / H
    else:
        w_arr = np.array(head_weights, dtype=float)
        w = w_arr / np.sum(w_arr)

    # combine according to per-head weights
    combined = np.tensordot(w, arr, axes=[0, 0])

    sink_idx = T-1
    W = combined.T                   # now columns sum to 1, matching Def. 2.3
    cumsum = np.cumsum(W, axis=-1)   # sample next from row-stochastic W

    mixing_times = []
    for start in range(T):
        steps_list = []
        for _ in range(num_simulations):
            current = start
            steps = 0
            while current != sink_idx and steps < max_steps:
                r = random.random()
                next_token = np.searchsorted(cumsum[current], r)
                current = min(next_token, T - 1)
                steps += 1
            steps_list.append(steps)
        mixing_times.append(np.mean(steps_list))
    # Return average over all start points
    return float(np.mean(mixing_times))

def minimax_diffusion_fidelity(
    attn: np.ndarray,
    head_weights: Optional[np.ndarray] = None,
    horizon: int = 100
) -> float:
    """
    Exact minimax diffusion fidelity over a `horizon`-step diffusion:
      φ_j = max_{1 ≤ t ≤ horizon} [P^t]_{j, sink}
      φ_min = min_j φ_j

    Args:
      attn          : (H, T, T) attention heads
      head_weights  : optional (H,) raw importances → convex weights
      horizon       : number of diffusion steps to consider

    Returns:
      float: the minimax fidelity φ_min
    """
    A = np.array(attn)           # (H, T, T)
    H, T, _ = A.shape
    sink = T - 1

    # 1) form convex weights
    if head_weights is None:
        w = np.ones(H) / H
    else:
        w_raw = np.array(head_weights, dtype=float)
        w = w_raw / np.sum(w_raw)

    # 2) combine heads → single T×T and build forward‐walk P
    combined = np.tensordot(w, A, axes=[0, 0])  # (T, T)
    P = combined.T                             # P[i→j] = combined[j,i]

    # 3) iterate powers of P, tracking the peak P^t(j→sink) for each j
    M = np.eye(T)      # starts as P^0
    phi = np.zeros(T)  # phi[j] = max_{t≤horizon} P^t(j→sink)
    for _ in range(horizon):
        M = M @ P                   # now M == P^t
        # for each start j, see P^t(j→sink) = M[j, sink]
        phi = np.maximum(phi, M[:, sink])

    # 4) minimax fidelity
    return float(np.min(phi))*100

def evaluate_metrics(model, dataloader, device, num_batches, num_simulations, num_layers = 4, use_learned_weights: bool = True):

    model.eval()
    mix_times = []
    fid_times = []
    # snr = []
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            inputs = inputs.to(device)
            logits, attn_weights_list = model(inputs, return_attn=True)

            mix_time = 0
            fid_time = 0
            for layer in range(num_layers):
                attn = attn_weights_list[layer][0].cpu().numpy()

                if use_learned_weights:
                    aw = _get_learned_head_weights(model.layers[layer].attn)
                else:
                    aw = None
                mix_time += monte_carlo_mixing_time(attn, num_simulations=num_simulations, head_weights=aw)
                fid_time += minimax_diffusion_fidelity(attn, head_weights=aw)
                
            mix_times.append(mix_time/num_layers)
            fid_times.append(fid_time/num_layers)

    return np.mean(mix_times), np.std(mix_times), np.mean(fid_times), np.std(fid_times)

def evaluate_metrics_per_layer(
    model,
    dataloader,
    device,
    num_batches,
    num_simulations,
    num_layers: int = 4,
    use_learned_weights: bool = True,
):
    """
    Returns:
      mix_means:   array of length num_layers, mean mixing time for each layer
      mix_stds:    array of length num_layers, std  mixing time for each layer
      fid_means:   array of length num_layers, mean fidelity   for each layer
      fid_stds:    array of length num_layers, std  fidelity   for each layer
    """
    model.eval()

    # Prepare per-layer accumulators
    mix_times_per_layer = [[] for _ in range(num_layers)]
    fid_times_per_layer = [[] for _ in range(num_layers)]

    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            inputs = inputs.to(device)
            logits, attn_weights_list = model(inputs, return_attn=True)

            for layer in range(num_layers):
                attn = attn_weights_list[layer][0].cpu().numpy()
                aw = (_get_learned_head_weights(model.layers[layer].attn)
                      if use_learned_weights else None)

                # compute both metrics just for this layer
                mix_t = monte_carlo_mixing_time(
                    attn,
                    num_simulations=num_simulations,
                    head_weights=aw
                )
                fid_t = minimax_diffusion_fidelity(
                    attn,
                    head_weights=aw
                )

                mix_times_per_layer[layer].append(mix_t)
                fid_times_per_layer[layer].append(fid_t)

    # Now compute mean & std per layer
    mix_means = np.array([np.mean(lst) for lst in mix_times_per_layer])
    mix_stds  = np.array([np.std(lst)  for lst in mix_times_per_layer])
    fid_means = np.array([np.mean(lst) for lst in fid_times_per_layer])
    fid_stds  = np.array([np.std(lst)  for lst in fid_times_per_layer])

    return mix_means, mix_stds, fid_means, fid_stds
