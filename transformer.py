import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CustomMultiHeadAttention(nn.Module):
    '''
    Custom Multi-Head Attention module with optional causal masking.
    This module computes the attention scores and applies them to the input tensor.
    It supports both causal and non-causal attention mechanisms.
    Parameters:
    - embed_dim (int): The dimension of the input embeddings.
    - num_heads (int): The number of attention heads.
    - dropout (float): Dropout probability for the attention weights.
    - causal (bool): If True, applies causal masking to the attention scores.
    '''
    def __init__(self, embed_dim, num_heads=1, dropout=0.1, causal=True):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attn=False):
        B, T, C = x.size()
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        if self.causal:
            mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
            scores = scores.masked_fill(~mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        if return_attn:
            return out, attn
        return out

class TransformerBlock(nn.Module):
    '''
    Transformer Block with Multi-Head Attention and Feed Forward Network.
    This block consists of a multi-head attention layer followed by a feed-forward network.
    Parameters:
    - embed_dim (int): The dimension of the input embeddings.
    - num_heads (int): The number of attention heads.
    - mlp_hidden_dim (int): The hidden dimension of the feed-forward network.
    - dropout (float): Dropout probability for the feed-forward network.
    - causal (bool): If True, applies causal masking to the attention scores.
    '''
    def __init__(self, embed_dim, num_heads, mlp_hidden_dim, dropout=0.1, causal=True):
        super().__init__()
        self.attn = CustomMultiHeadAttention(embed_dim, num_heads=num_heads, dropout=dropout, causal=causal)
        self.ln1 = nn.RMSNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.RMSNorm(embed_dim)
    
    def forward(self, x, return_attn=False):

        if return_attn:
            attn_out, attn_weights = self.attn(self.ln1(x), return_attn=True)
            x = x + attn_out
        else:
            x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        if return_attn:
            return x, attn_weights
        return x

class Transformer(nn.Module):
    '''
    Transformer model with multiple layers of multi-head attention and feed-forward networks.
    This model is designed for sequence-to-sequence tasks and can be used for various NLP applications.
    Parameters:
    - vocab_size (int): The size of the vocabulary.
    - embed_dim (int): The dimension of the input embeddings.
    - num_heads (int): The number of attention heads.
    - num_layers (int): The number of transformer layers.
    - mlp_hidden_dim (int): The hidden dimension of the feed-forward network.
    - max_seq_length (int): The maximum sequence length for positional embeddings.
    - dropout (float): Dropout probability for the feed-forward network.
    - causal (bool): If True, applies causal masking to the attention scores.
    '''
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, mlp_hidden_dim, max_seq_length, dropout=0.1, causal=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_length, embed_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_hidden_dim, dropout=dropout, causal=causal)
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(embed_dim, vocab_size)
        self.max_seq_length = max_seq_length
        
    def forward(self, x, return_attn=False):
        B, T = x.size()
        token_embeddings = self.token_emb(x)
        positions = torch.arange(0, T, device=x.device).unsqueeze(0).expand(B, T)
        pos_embeddings = self.pos_emb(positions)
        x = token_embeddings + pos_embeddings
        attn_weights_list = []
        for layer in self.layers:
            if return_attn:
                x, attn_weights = layer(x, return_attn=True)
                attn_weights_list.append(attn_weights)
            else:
                x = layer(x)
        logits = self.out_proj(x)
        if return_attn:
            return logits, attn_weights_list
        return logits