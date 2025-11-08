import numpy as np
from numpy.linalg import matrix_power
import pandas as pd
import matplotlib.pyplot as plt

def build_delta(edges, n):
    """
    Build a diffusion matrix for a graph on n nodes with self-loops + given directed edges.
    edges: list of (source, target) tuples.
    """
    D = np.zeros((n, n))
    # self-loops
    for i in range(n):
        D[i, i] = 1.0
    # add directed edges
    for src, tgt in edges:
        D[tgt, src] += 1.0
    # normalize rows (sum over sending nodes for each receiving node)
    for i in range(n):
        total = D[i].sum()
        if total > 0:
            D[i] /= total
    return D

# Node indices
# A=0, B=1, C=2, τ=3
n = 4
A, B, C, T = 0, 1, 2, 3

# Define head graphs
edges1 = [(A, B), (B, C), (C, T)]  # A->B->C->τ chain
edges2 = [(A,C), (B, C), (B, T), (C, T)]  # B->A->C->τ twisted chain

# Build diffusion matrices
Delta1 = build_delta(edges1, n)
Delta2 = build_delta(edges2, n)

# Combine with equal weights
beta1 = beta2 = 0.5
Delta_multi = beta1 * Delta1 + beta2 * Delta2

# Simulate signal propagation up to t=20
T_max = 20
records = []
for t in range(1, T_max + 1):
    D1 = matrix_power(Delta1, t)
    D2 = matrix_power(Delta2, t)
    DM = matrix_power(Delta_multi, t)
    records.append({
        't': t,
        'head1_A': D1[T, A],
        'head1_B': D1[T, B],
        'head1_C': D1[T, C],
        'head2_A': D2[T, A],
        'head2_B': D2[T, B],
        'head2_C': D2[T, C],
        'multi_A': DM[T, A],
        'multi_B': DM[T, B],
        'multi_C': DM[T, C],
    })

# Create a DataFrame and display
df = pd.DataFrame(records)
print("Signal at τ over time (t = 1..20):")
print(df.to_string(index=False))

# Compute node-fidelities and minimax fidelities
phi1 = min(df[['head1_A', 'head1_B', 'head1_C']].max())
phi2 = min(df[['head2_A', 'head2_B', 'head2_C']].max())
phi_multi = min(df[['multi_A', 'multi_B', 'multi_C']].max())

print(f"\nHead1 minimax fidelity φ_min(1) = {phi1:.3f}")
print(f"Head2 minimax fidelity φ_min(2) = {phi2:.3f}")
print(f"Multi-head minimax fidelity φ_min(multi) = {phi_multi:.3f}")



def build_delta(edges, n):
    """Build a diffusion matrix for a graph on n nodes with self-loops + given directed edges."""
    D = np.zeros((n, n))
    for i in range(n):
        D[i, i] = 1.0  # self-loops
    for src, tgt in edges:
        D[tgt, src] += 1.0
    for i in range(n):  # normalize rows
        total = D[i].sum()
        if total > 0:
            D[i] /= total
    return D

# Node indices
n = 4
u, v, w, T = 0, 1, 2, 3

# Define head graphs
edges1 = [(u, v), (v, w), (w, T)]
edges2 = [(u, w), (v, w), (v, T), (w, T)]

# Build diffusion matrices
Delta1 = build_delta(edges1, n)
Delta2 = build_delta(edges2, n)
beta1 = beta2 = 0.5
Delta_multi = beta1 * Delta1 + beta2 * Delta2

# Simulate signal propagation
T_max = 20
records = []
for t in range(1, T_max + 1):
    D1  = matrix_power(Delta1, t)
    D2  = matrix_power(Delta2, t)
    DM  = matrix_power(Delta_multi, t)
    records.append({
        't':        t,
        'head1_u':  D1[T, u], 'head1_v': D1[T, v], 'head1_w': D1[T, w],
        'head2_u':  D2[T, u], 'head2_v': D2[T, v], 'head2_w': D2[T, w],
        'multi_u':  DM[T, u], 'multi_v': DM[T, v], 'multi_w': DM[T, w],
    })

df = pd.DataFrame(records)

# Compute minimax-up-to-t
for prefix in ('head1', 'head2', 'multi'):
    for node in ('u', 'v', 'w'):
        df[f'{prefix}_{node}_max'] = df[f'{prefix}_{node}'].cummax()
    df[f'{prefix}_min'] = df[
        [f'{prefix}_u_max', f'{prefix}_v_max', f'{prefix}_w_max']
    ].min(axis=1)

# Plot function to save each figure
def plot_head(prefix, title, filename):
    plt.figure(figsize=(6, 4))
    plt.plot(df['t'], df[f'{prefix}_u'] * 100, label='u')
    plt.plot(df['t'], df[f'{prefix}_v'] * 100, label='v')
    plt.plot(df['t'], df[f'{prefix}_w'] * 100, label='w')
    plt.plot(df['t'], df[f'{prefix}_min'] * 100,
             linestyle='--', linewidth=2, color='red',
             label=r'$\phi_{min}$')
    plt.xlabel('Diffusion steps')
    plt.ylabel('Signal % at sink')
    plt.xticks(np.arange(1, df['t'].max() + 1, 1))
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# Save figures
plot_head('head1', 'Head 1', 'head1.png')
plot_head('head2', 'Head 2', 'head2.png')
plot_head('multi', 'Multi-Head', 'multi.png')

print("Saved head1.png, head2.png, and multi.png")
