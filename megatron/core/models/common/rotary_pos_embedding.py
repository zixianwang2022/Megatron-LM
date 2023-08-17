# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import importlib.util
import torch

from torch import einsum, nn

__all__ = ['RotaryEmbedding', 'apply_rotary_pos_emb']

# >>>
def pax(a):
    from scripts import pax as _pax
    _pax(a)
# <<<


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        # >>>
        # pax({"dim": dim, "inv_freq": inv_freq, "inv_freq / self": self.inv_freq})
        # <<<

    def forward(self, max_seq_len, offset=0):
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # dim = 2*self.inv_freq.numel()
        # self.inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        # self.inv_freq = self.inv_freq.to(device="cuda")
        # pax({"inv_freq / self": self.inv_freq})
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        seq = torch.arange(max_seq_len, device=self.inv_freq.device) + offset
        freqs = einsum('i , j -> i j', seq.type_as(self.inv_freq), self.inv_freq)
        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        emb = torch.cat((freqs, freqs), dim=-1)
        # emb [seq_length, .., dim]
        # >>>
        # pax({
        #     "inv_freq" : self.inv_freq,
        #     "max_seq_len" : max_seq_len,
        #     "offset" : offset,
        #     "seq" : seq,
        #     "freqs" : freqs,
        #     # "freqs.32" : einsum('i , j -> i j', seq, self.inv_freq.float()),
        #     # "freqs / complex" : torch.view_as_complex(freqs.reshape(*freqs.shape[:-1], -1, 2)),
        #     "freqs / polar" : torch.polar(torch.ones_like(freqs), freqs),
        #     "emb" : emb,
        #     "emb / out" : emb[:, None, None, :],
        # })
        # <<<
        return emb[:, None, None, :]


def _rotate_half(x):
    """
    change sign so the last dimension becomes [-odd, +even]
    """
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs):
    """
    input tensor t is of shape [seq_length, ..., dim]
    rotary positional embeding tensor freqs is of shape [seq_length, ..., dim]
    check https://kexue.fm/archives/8265 for detailed formulas
    """
    rot_dim = freqs.shape[-1]
    # >>>
    # pax({
    #     "freqs" : freqs.transpose(0, 1),
    #     # "freqs / complex" : torch.view_as_complex(freqs.float().reshape(*freqs.shape[:-1], -1, 2)).transpose(0, 1),
    #     # "freqs / polar" : torch.polar(freqs),
    #     "t" : t.transpose(0, 1),
    #     "rot_dim": rot_dim,
    # })
    # <<<
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    # >>>
    # t = (t * freqs.cos()) + (_rotate_half(t) * freqs.sin())
    t = (t * freqs.cos()) + (t * freqs.sin())
    # <<<
    # >>>
    # pax({
    #     "freqs" : freqs.transpose(0, 1),
    #     "rot_dim" : rot_dim,
    #     "t" : t.transpose(0, 1),
    #     "t_pass" : t_pass.transpose(0, 1),
    #     "out" : torch.cat((t, t_pass), dim=-1).transpose(0, 1),
    # })
    # <<<
    return torch.cat((t, t_pass), dim=-1)
