# Standard imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os

# Custom imports
from transformer import Transformer
from sequence_datasets import CopyDataset, CycleDataset
from proxies import evaluate_metrics_per_layer
from train import train_model, save_checkpoint, load_checkpoint

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Hyperparameters
    NUM_SAMPLES = 5000
    SEQ_LENGTH = 100
    VOCAB_SIZE = 256
    BATCH_SIZE = 50
    EMBED_DIM = 64            # Total embedding dimension remains fixed
    NUM_LAYERS = 4
    MLP_HIDDEN_DIM = 128
    NUM_EPOCHS = 200
    LR = 1e-3
    NUM_SIMULATIONS = 500     # for Monte Carlo mixing time estimation
    HEAD_COUNTS = [1, 4, 8, 16]  # number of heads for evaluation
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    CHECKPOINT_DIR = "./checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Define datasets
    datasets = {
        "copy":  DataLoader(CopyDataset(NUM_SAMPLES, SEQ_LENGTH, VOCAB_SIZE), batch_size=BATCH_SIZE, shuffle=True),
        "cycle": DataLoader(CycleDataset(NUM_SAMPLES, SEQ_LENGTH, VOCAB_SIZE), batch_size=BATCH_SIZE, shuffle=True),
    }

    # Loop over head counts and tasks
    for h in HEAD_COUNTS:
        print(f"\n=== {h} heads ===")
        for name, loader in datasets.items():
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"{h}_heads_{name}.pt")
            print(f"\n-- Task: {name.capitalize()} --")

            # 1) initialize a fresh model for this head-count
            model = Transformer(
                VOCAB_SIZE, EMBED_DIM,
                num_heads=h, num_layers=NUM_LAYERS,
                mlp_hidden_dim=MLP_HIDDEN_DIM,
                max_seq_length=SEQ_LENGTH,
                causal=True
            ).to(DEVICE)

            # 2) load or train on this task
            if os.path.exists(ckpt_path):
                print(f"Loading checkpoint from {ckpt_path}")
                model = load_checkpoint(model, ckpt_path, device=DEVICE)
            else:
                print(f"Training on {name} dataset…")
                train_model(model, loader, num_epochs=NUM_EPOCHS, lr=LR, device=DEVICE)
                save_checkpoint(model, ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")

            # 3) evaluate per-layer mixing-time and fidelity
            mix_means, mix_stds, fid_means, fid_stds = evaluate_metrics_per_layer(
                model, loader, DEVICE,
                num_batches=BATCH_SIZE,
                num_simulations=NUM_SIMULATIONS,
                num_layers=NUM_LAYERS,
                use_learned_weights=True
            )

            # 4) print out each layer’s metrics
            for layer in range(NUM_LAYERS):
                print(
                    f"Layer {layer:>2} — "
                    f"mixing time: {mix_means[layer]:.4f} ± {mix_stds[layer]:.4f}, "
                    f"fidelity:      {fid_means[layer]:.2f}% ± {fid_stds[layer]:.2f}%"
                )
