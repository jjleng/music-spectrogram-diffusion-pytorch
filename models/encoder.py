import math
import torch
from torch import nn, Tensor
from typing import Optional
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_


def geglu(x):
    x, gates = x.chunk(2, dim=-1)
    return x * F.gelu(gates)


class MultiheadAttention(nn.Module):
    """Custom MultiheadAttention compatible with PyTorch 2.x"""
    def __init__(self, emb_dim, nhead, head_dim, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.head_dim = head_dim
        self.num_heads = nhead
        self.embed_dim = emb_dim

        self.q_proj = nn.Linear(emb_dim, head_dim * nhead, bias=False)
        self.k_proj = nn.Linear(emb_dim, head_dim * nhead, bias=False)
        self.v_proj = nn.Linear(emb_dim, head_dim * nhead, bias=False)
        self.emb_proj = nn.Linear(head_dim * nhead, emb_dim, bias=False)

        xavier_uniform_(self.q_proj.weight.data)
        xavier_uniform_(self.k_proj.weight.data)
        xavier_uniform_(self.v_proj.weight.data)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                attn_mask=None, average_attn_weights=True, is_causal=False):
        B, T, _ = query.size()
        Q = self.q_proj(query).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_score = Q @ K.transpose(-2, -1) / self.head_dim ** 0.5

        if attn_mask is not None:
            attn_score = attn_score + attn_mask

        attn_prob = F.softmax(attn_score, dim=-1)
        attn_prob = F.dropout(attn_prob, self.dropout, self.training)

        attn_vec = attn_prob @ V
        x = self.emb_proj(attn_vec.transpose(1, 2).reshape(B, T, -1))
        if not need_weights:
            return x, None
        if average_attn_weights:
            return x, attn_prob.mean(dim=2)
        return x, attn_prob


class EncoderLayer(nn.Module):
    """Standalone encoder layer without inheriting from TransformerEncoderLayer"""
    def __init__(self, emb_dim, nhead, head_dim, dropout=0.1, dim_feedforward=2048,
                 activation=geglu, layer_norm_eps=1e-5, norm_first=True, **kwargs):
        super().__init__()
        self.self_attn = MultiheadAttention(emb_dim, nhead, head_dim, dropout)

        # Feedforward - 2x for GeGLU
        self.linear1 = nn.Linear(emb_dim, dim_feedforward * 2)
        self.linear2 = nn.Linear(dim_feedforward, emb_dim)

        self.norm1 = nn.LayerNorm(emb_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(emb_dim, eps=layer_norm_eps)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = activation
        self.norm_first = norm_first

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor],
                  key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x, attn_mask=attn_mask,
                          key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.activation(self.linear1(x)))
        return self.dropout2(x)


class Encoder(nn.Module):
    """Standalone encoder without inheriting from TransformerEncoder"""
    def __init__(self, layers, emb_dim, nhead, head_dim, **kwargs) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(emb_dim, nhead, head_dim, activation=geglu, **kwargs)
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, src: Tensor, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output
