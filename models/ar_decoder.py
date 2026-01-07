import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Optional

from .encoder import geglu, Encoder, MultiheadAttention
from .utils import sinusoidal


class DecoderLayer(nn.Module):
    """Standalone decoder layer without inheriting from TransformerDecoderLayer"""
    def __init__(self, emb_dim, nhead, head_dim, dropout=0.1, dim_feedforward=2048,
                 activation=geglu, layer_norm_eps=1e-5, norm_first=True, **kwargs):
        super().__init__()
        self.self_attn = MultiheadAttention(emb_dim, nhead, head_dim, dropout)
        self.multihead_attn = MultiheadAttention(emb_dim, nhead, head_dim, dropout)

        # Feedforward - 2x for GeGLU
        self.linear1 = nn.Linear(emb_dim, dim_feedforward * 2)
        self.linear2 = nn.Linear(dim_feedforward, emb_dim)

        self.norm1 = nn.LayerNorm(emb_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(emb_dim, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(emb_dim, eps=layer_norm_eps)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = activation
        self.norm_first = norm_first

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            x = self.norm3(x + self._ff_block(x))
        return x

    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x, attn_mask=attn_mask,
                          key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        x = self.multihead_attn(x, mem, mem, attn_mask=attn_mask,
                               key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout2(x)

    def _ff_block(self, x):
        x = self.linear2(self.activation(self.linear1(x)))
        return self.dropout3(x)


class Decoder(nn.Module):
    """Standalone decoder without inheriting from TransformerDecoder"""
    def __init__(self, layers, emb_dim, nhead, head_dim, **kwargs) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(emb_dim, nhead, head_dim, activation=geglu, **kwargs)
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt
        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        memory_key_padding_mask=memory_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class Transformer(nn.Module):
    """Standalone transformer without inheriting from nn.Transformer"""
    def __init__(self, emb_dim, nhead, head_dim, num_encoder_layers, num_decoder_layers, **kwargs) -> None:
        super().__init__()
        self.encoder = Encoder(num_encoder_layers, emb_dim, nhead, head_dim, **kwargs)
        self.decoder = Decoder(num_decoder_layers, emb_dim, nhead, head_dim, **kwargs)
        self.d_model = emb_dim

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             memory_key_padding_mask=memory_key_padding_mask)
        return output

    def generate_square_subsequent_mask(self, sz, device=None):
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

    def autoregressive_infer(self, tgt, src=None, memory=None,
                             src_mask=None, tgt_mask=None, memory_mask=None,
                             src_key_padding_mask=None, tgt_key_padding_mask=None,
                             memory_key_padding_mask=None):
        if memory is None:
            memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        out = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                          tgt_key_padding_mask=tgt_key_padding_mask,
                          memory_key_padding_mask=memory_key_padding_mask)
        return out, memory


class MIDI2SpecAR(nn.Module):
    def __init__(self,
                 num_emb, output_dim,
                 max_input_length, max_output_length,
                 emb_dim, nhead, head_dim, num_encoder_layers, num_decoder_layers, **kwargs) -> None:
        super().__init__()
        self.emb = nn.Embedding(num_emb, emb_dim)
        self.register_buffer('in_pos_emb', sinusoidal(shape=(max_input_length, emb_dim)))
        self.register_buffer('out_pos_emb', sinusoidal(shape=(max_output_length, emb_dim)))
        self.transformer = Transformer(
            emb_dim, nhead, head_dim, num_encoder_layers, num_decoder_layers, **kwargs)
        self.linear_in = nn.Linear(output_dim, emb_dim)
        self.linear_out = nn.Linear(emb_dim, output_dim)

    def forward(self, midi_tokens, spec):
        batch_size, seq_len = midi_tokens.shape
        midi = self.emb(midi_tokens) + self.in_pos_emb[:seq_len]
        spec = self.linear_in(spec) + self.out_pos_emb[:spec.shape[1]]
        spec_tri_mask = self.transformer.generate_square_subsequent_mask(
            spec.shape[1], device=spec.device)
        x = self.transformer(midi, spec, tgt_mask=spec_tri_mask)
        x = self.linear_out(x)
        return x

    def infer(self, midi_tokens, max_len=512, dither_amount=0., verbose=True):
        batch_size, seq_len = midi_tokens.shape
        midi = self.emb(midi_tokens) + self.in_pos_emb[:seq_len]
        spec = midi.new_zeros((batch_size, 1, self.linear_out.weight.shape[0]))
        memory = None

        for i in tqdm(range(max_len), disable=not verbose):
            spec_emb = self.linear_in(spec) + self.out_pos_emb[:i+1]
            next_spec, memory = self.transformer.autoregressive_infer(
                spec_emb, src=midi, memory=memory)
            next_spec = self.linear_out(next_spec[:, -1:])
            if dither_amount > 0:
                next_spec = next_spec + dither_amount * torch.randn_like(next_spec)
            next_spec.clamp_(-1, 1)
            spec = torch.cat([spec, next_spec], dim=1)

        return spec[:, 1:]
