import os
import math
import numpy as np
import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from transformers import GPT2TokenizerFast
from tqdm import tqdm

class Config:
    def __init__(self, n_vocab=500, n_ctx=1024, n_embd=750, n_layer=10, n_head=10, dropout=0.1, tie_embeddings=True):
        self.n_vocab = n_vocab
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.dropout = dropout
        self.tie_embeddings = tie_embeddings

# Count model parameters        
def count_params(model):
    return sum(p.numel() for p in model.parameters())

# Softmax function, subtracting with max helps avoid overflow errors
def softmax(x, dim=-1):
    ex = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)
    return ex / torch.sum(x, dim=dim, keepdim=True)

# GeLU implementation
def gelu(x):
    return 0.5*x*(1+torch.tanh(np.sqrt(2/np.pi)*(x+0.044715*torch.pow(x, 3))))

# Normalization with diagonal affine transformation
def norm(x, *, dim=-1, epsilon=1e-5):
    n_state = x.shape[-1]
    g = nn.Parameter(torch.ones(n_state))
    b = nn.Parameter(torch.zeros(n_state))
    u = torch.mean(x, dim=dim, keepdim=True)
    s = torch.mean((x - u)**2, dim=dim, keepdim=True)
    x = (x - u) * torch.rsqrt(s + epsilon)
    x = x*g + b
    return x

# Reshape last dimension of X into [n, x.shape[-1]/n]
def split_states(x, n):
    *start, m = list(x.shape)
    return x.view(*start, n, m//n)

# Merge last two dimensions of x into a single dimension
def merge_states(x):
    *start, a, b = list(x.shape)
    return x.reshape(*start, a*b)

# Just a linear layer : c = w*x + b
def conv1d(x, nf, *, w_init_stdev=0.02):
    *start, nx = list(x.shape)
    w = nn.Parameter(torch.randn(1, nx, nf) * w_init_stdev)
    b = nn.Parameter(torch.zeros(nf))
    c = torch.matmul(x.view(-1, nx), w.view(-1, nf)) + b
    c = c.view(*start, nf)
    return c

# Attention mask with lower triangle 1's
def attention_mask(nd, ns, *, dtype):
    i = torch.arange(nd).view(-1, 1)
    j = torch.arange(ns)
    m = i >= (j - ns + nd)
    return m.to(dtype)

def attn(x, n_state, *, past, hparams):
    assert x.ndim == 3  # Should be [batch, sequence, features]
    assert n_state % hparams.n_head == 0
    if past is not None:
        assert past.shape.ndims == 5  # Should be [batch, [k, v], heads, sequence, features]

    # def split_heads(x):
    #     # From [batch, sequence, features] to [batch, heads, sequence, features]
    #     return split_states(x, hparams.n_head).permute(0, 2, 1, 3)

    # def merge_heads(x):
    #     # Reverse of split_heads
    #     return merge_states(x.permute(0, 2, 1, 3))

    def mask_attn_weights(w):
        # w has shape [batch, heads, dest_sequence, src_sequence], where information flows from src to dest.
        _, _, nd, ns = list(w.shape)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = b.view(1, 1, nd, ns)
        w = w*b - torch.tensor(1e10, dtype=w.dtype)*(1-b)
        return w

    def multihead_attn(q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = torch.matmul(q, k.transpose(-2, -1)) # tf.matmul(q, k, transpose_b=True)
        w = w * torch.rsqrt(torch.tensor(v.shape[-1], dtype=w.dtype))
        w = mask_attn_weights(w)
        w = softmax(w)
        a = torch.matmul(w, v)
        return a

    c = conv1d(x, n_state*3)
    batch, seq, qkv = c.shape
    features_per_head = qkv//(3*hparams.n_head)
    c = c.view(batch, seq, 3, hparams.n_head, features_per_head)
    c = c.permute(2, 0, 3, 1, 4)
    q, k, v = c[0], c[1], c[2]
    present = torch.stack((k, v), dim=1)
    if past is not None:
        pk, pv = torch.unbind(past, dim=1)
        k = torch.cat([pk, k], dim=-2)
        v = torch.cat([pv, v], dim=-2)
    a = multihead_attn(q, k, v)
    a = merge_states(a.permute(0, 2, 1, 3))
    a = conv1d(a, n_state)
    return a, present

def mlp(x, n_state, *, hparams):
    nx = x.shape[-1]
    h = gelu(conv1d(x, n_state))
    h2 = conv1d(h, nx)
    return h2

def block(x, *, past, hparams):
    nx = x.shape[-1]
    a, present = attn(norm(x), nx, past=past, hparams=hparams)
    x = x + a
    m = mlp(norm(x), nx*4, hparams=hparams)
    x = x + m
    return x, present

def past_shape(*, hparams, batch_size=None, sequence=None):
    return [batch_size, hparams.n_layer, 2, hparams.n_head, sequence, hparams.n_embd // hparams.n_head]

# Adds a new dimension of given size
def expand_tile(value, size):
    value = torch.as_tensor(value)
    ndims = value.dim()
    return value.unsqueeze(0).repeat(size, *[1]*ndims)

def positions_for(tokens, past_length):
    batch_size, nsteps = tokens.shape[:2]
    return expand_tile(past_length + torch.arange(nsteps, device=tokens.device), batch_size)

class Model(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.wte = nn.Parameter(torch.randn(hparams.n_vocab, hparams.n_embd) * 0.02)
        self.wpe = nn.Parameter(torch.randn(hparams.n_ctx, hparams.n_embd) * 0.01)
        self.ln_f = nn.LayerNorm(hparams.n_embd)
        
    def forward(self, X, past=None):
        results = {}
        batch, seq = X.shape
        past_len = 0 if past is None else [None] * self.hparams.n_layer
        h = self.wte[X] + self.wpe[positions_for(X, past_len)]
        
        # Transformer
        presents = []
        pasts = torch.unbind(past, dim=1) if past is not None else [None] * self.hparams.n_layer
        assert len(pasts) == self.hparams.n_layer
        for past in pasts:
            h, present = block(h, past=past, hparams=self.hparams)
            presents.append(present)
        results['present'] = torch.stack(presents, dim=1)
        h = norm(h)
        
        # Language model loss.  Do tokens <n predict token n?
        h_flat = h.view(batch*seq, self.hparams.n_embd)
        logits = torch.matmul(h_flat, self.wte.t())
        logits = logits.view(batch, seq, self.hparams.n_vocab)
        results['logits'] = logits
        return results

hparams = Config()

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.BpeTrainer(vocab_size=hparams.n_vocab, min_frequency=2)
tokenizer.train(["imdb_unsup.txt"], trainer)
ds = load_dataset("HuggingFaceTB/cosmopedia", "web_samples_v1")
print(ds)
print(dfds)
encoded = tokenizer.encode("Hi, how are you?")
print(encoded.ids, encoded.tokens)
decoded = tokenizer.decode(encoded.ids)
print(decoded)

model = Model(hparams)
outputs = model(torch.tensor([encoded.ids]), past=None)
print("Total trainable parameters:", model)
logits = outputs['logits']
next_token_probs = torch.softmax(logits, dim=-1)
next_tokens = torch.argmax(next_token_probs, dim=-1)
print(tokenizer.decode(list(next_tokens[0])))
print(logits.shape)
print(next_tokens.shape)
















# # Layer Normalization
# class LayerNorm(nn.Module):
#     def __init__(self, n_embd, eps=1e-5):
#         super().__init__()
#         self.ln = nn.LayerNorm(n_embd, eps=eps)
        
#     def forward(self, x):
#         return self.ln(x)
    
# class CausalSelfAttention(nn.Module):
#     """GPT-2 style multi-head causal self-attention with combined QKV linear."""
#     def __init__(self, config):
#         super().__init__()
#         n_embd = config.n_embd
#         self.n_head = config.n_head
#         assert n_embd % self.n_head == 0
#         self.head_dim = n_embd // self.n_head
#         self.scale = 1.0 / math.sqrt(self.head_dim)

#         # QKV projection and output projection
#         self.c_attn = nn.Linear(n_embd, 3 * n_embd)   # W_qkv
#         self.c_proj = nn.Linear(n_embd, n_embd)       # W_o

#         self.dropout = nn.Dropout(config.dropout)
