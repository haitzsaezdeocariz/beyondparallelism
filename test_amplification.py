import torch
import numpy as np
from transformer import Transformer
from sequence_datasets import CopyDataset, CycleDataset
from proxies import minimax_diffusion_fidelity, _get_learned_head_weights
from train import load_checkpoint
from torch.utils.data import DataLoader

DEVICE = 'cuda:5' if torch.cuda.is_available() else 'cpu'
VOCAB_SIZE = 256
EMBED_DIM = 64
SEQ_LENGTH = 100
NUM_LAYERS = 4
MLP_HIDDEN_DIM = 128
BATCH_SIZE = 1  # Evaluating fidelity one sample at a time
HEAD_COUNTS = [4, 8, 16]
TASKS = {'copy': CopyDataset, 'cycle': CycleDataset}

for task_name, dataset_cls in TASKS.items():
    loader = DataLoader(dataset_cls(100, SEQ_LENGTH, VOCAB_SIZE), batch_size=BATCH_SIZE, shuffle=False)

    for num_heads in HEAD_COUNTS:
        checkpoint_path = f'./checkpoints/{num_heads}_heads_{task_name}.pt'

        model = Transformer(
            VOCAB_SIZE, EMBED_DIM, num_heads=num_heads, num_layers=NUM_LAYERS,
            mlp_hidden_dim=MLP_HIDDEN_DIM, max_seq_length=SEQ_LENGTH, causal=True
        ).to(DEVICE)
        model = load_checkpoint(model, checkpoint_path, device=DEVICE)

        model.eval()
        with torch.no_grad():
            inputs, _ = next(iter(loader))
            inputs = inputs.to(DEVICE)
            _, attn_weights_list = model(inputs, return_attn=True)

            for layer in range(NUM_LAYERS):
                attn = attn_weights_list[layer][0].cpu().numpy()  # (H, T, T)
                aw = _get_learned_head_weights(model.layers[layer].attn)
                H = attn.shape[0]

                fidelities_individual = [
                    minimax_diffusion_fidelity(attn[h:h+1, :, :]) for h in range(H)
                ]
                best_individual_fidelity = np.max(fidelities_individual)

                combined_fidelity = minimax_diffusion_fidelity(attn, head_weights=aw)

                print(f"Task: {task_name.capitalize()}, Heads: {num_heads}, Layer {layer+1} Results:")
                print(f"Best Individual Fidelity: {best_individual_fidelity:.2f}%")
                print(f"Combined Multi-Head Fidelity: {combined_fidelity:.2f}%")

                if combined_fidelity > best_individual_fidelity:
                    print("Multi-head combination successfully exceeds individual best fidelity!\n")
                else:
                    print("Multi-head combination does not exceed individual best fidelity.\n")