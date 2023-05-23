import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """positional encoding by Vaswani et al., code by Informer"""
    def __init__(self, d_input, d_model):
        super().__init__()
        padding_surplus = 20
        pe = torch.zeros(d_input + padding_surplus, d_model).float()  # matrix with zeros, account for conv padding
        pe.require_grad = False

        # Compute the positional encodings once in log space.
        position = torch.arange(0, d_input + padding_surplus).float().unsqueeze(1)  # position from 1 to d_input
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()  # denominator

        pe[:, 0::2] = torch.sin(position * div_term)  # fill in pe at correct spot, sin for even
        pe[:, 1::2] = torch.cos(position * div_term)  # and cos for uneven

        pe = pe.unsqueeze(0)  # add batch dim
        self.register_buffer('pe', pe)  # store params in state_dict, but not trained by optimizer

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]  # length may vary due to conv


class ConvEmbedding(nn.Module):
    def __init__(self, d_channel, d_model, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.conv_layer = nn.Conv1d(in_channels=d_channel,
                                    out_channels=d_model,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    dilation=dilation)

    def forward(self, x):
        x = x.transpose(1, 2)  # B, L, C -> B, C, L
        x = self.conv_layer(x)
        x = x.transpose(1, 2)  # -> B, L, C
        return x