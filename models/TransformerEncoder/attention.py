import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_qk: int,
                 d_v: int,
                 n_heads: int,
                 mask: bool = False):
        super().__init__()

        self.W_q = torch.nn.Linear(d_model, d_qk * n_heads)
        self.W_k = torch.nn.Linear(d_model, d_qk * n_heads)
        self.W_v = torch.nn.Linear(d_model, d_v * n_heads)

        self.W_o = torch.nn.Linear(d_v * n_heads, d_model)

        self.n_heads = n_heads
        self.d_qk = d_qk

        self.mask = mask
        self.score = None

    def forward(self, x, stage, device):
        # reorganize Q, K and V for parallel computation of attn heads
        Q = torch.cat(self.W_q(x).chunk(self.n_heads, dim=-1), dim=0)
        K = torch.cat(self.W_k(x).chunk(self.n_heads, dim=-1), dim=0)
        V = torch.cat(self.W_v(x).chunk(self.n_heads, dim=-1), dim=0)

        # scaled dot attn
        score = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_qk)
        self.score = score

        # masking future
        if self.mask and stage == 'train':
            mask = torch.ones_like(score[0])
            mask = torch.tril(mask, diagonal=0)
            masked = torch.Tensor([-2**32+1]).expand_as(score[0]).to(device)
            score = torch.where(mask > 0, score, masked)

        # calculate attn
        score = F.softmax(score, dim=-1)
        attention = torch.matmul(score, V)
        attention_heads = torch.cat(attention.chunk(self.n_heads, dim=0), dim=-1)
        self_attention = self.W_o(attention_heads)

        return self_attention, self.score