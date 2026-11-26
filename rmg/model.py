"""One-layer attention model for the running-maximum task.

A number (token) x_t is embedded by a trained table E (`We`), the embeddings are combined by one causal
self-attention layer whose query, key and value maps are fixed to the identity, and the result is read
out by a trained, bias-free table U (`Wd`). Token v stands for the number v + 1.
"""

import math

import torch
import torch.nn as nn


class TxformerSimple(nn.Module):
    """
    vocab_size:   number of tokens V (rows of E and U).
    embed_dim:    embedding dimension d.
    decoder_type: 'linear' -> logits z . U_j (bias-free linear readout; the baseline);
                  'dist'   -> logits (2 / sqrt(d)) z . U_j - |z|^2 - |U_j|^2 (distance readout).
    dist_attn:    False -> attention scores h_t . h_t' / sqrt(d) (the baseline);
                  True  -> attention scores -|h_t - h_t'|^2 / d (distance attention).
    """

    def __init__(self, vocab_size, embed_dim, decoder_type='linear', dist_attn=False):
        super().__init__()
        self.K, self.d = vocab_size, embed_dim
        self.We = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)   # E
        self.Wd = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)   # U
        self.decoder_type = decoder_type
        self.dist_attn = dist_attn
        self.attn_softmax = nn.Softmax(dim=-1)
        # query, key and value maps fixed to the identity (frozen)
        self.Wq = nn.Parameter(torch.eye(embed_dim), requires_grad=False)
        self.Wk = nn.Parameter(torch.eye(embed_dim), requires_grad=False)
        self.Wv = nn.Parameter(torch.eye(embed_dim), requires_grad=False)

    def embed(self, inputs):
        """inputs: integer tensor (B, T) -> embeddings (B, T, d)."""
        return self.We(inputs)

    def decode(self, x):
        """x: (B, T, d) -> logits (B, T, V)."""
        if self.decoder_type == 'linear':
            return x @ self.Wd.weight.T
        if self.decoder_type == 'dist':
            d = x.shape[-1]
            return (1 / math.sqrt(d)) * 2 * (x @ self.Wd.weight.T) - (x ** 2).sum(-1, keepdim=True) \
                - (self.Wd.weight ** 2).sum(-1)
        raise ValueError(f'unknown decoder_type {self.decoder_type}')

    def attn(self, x):
        """Causal self-attention with fixed identity maps: x (B, T, d) -> (B, T, d)."""
        B, T, d = x.shape
        Q, K, V = x @ self.Wq, x @ self.Wk, x @ self.Wv
        if self.dist_attn:
            tau = self.d
            scores = (2 * Q @ K.transpose(-1, -2) - (Q ** 2).sum(-1, keepdim=True)
                      - (K ** 2).sum(-1).unsqueeze(-2)) / tau
        else:
            tau = self.d ** 0.5
            scores = Q @ K.transpose(-1, -2) / tau
        causal_mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(causal_mask == 0, -torch.inf)
        return self.attn_softmax(scores) @ V

    def forward(self, x):
        """x: integer tensor (B, T) -> logits (B, V, T), the layout nn.CrossEntropyLoss expects."""
        return self.decode(self.attn(self.embed(x))).permute(0, 2, 1)
