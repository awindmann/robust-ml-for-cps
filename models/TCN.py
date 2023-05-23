import torch.nn as nn
from models.core import PLCore
from torch.nn.utils import weight_norm


class TCN(PLCore):
    """Pretty much the default TCN, with linear layer at the end
    to get the dimensions right

    The implementation is taken from here in large parts:
    https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
    """

    def __init__(
        self,
        in_channels: int = 3,
        seq_len_x: int = 250,
        seq_len_y: int = 50,
        kernel_size: int = 5,
        num_channels: list = [6, 12, 18, 12, 6],
        dropout: float = 0.2,
        ** kwargs,
    ):
        super().__init__(**kwargs)
        self.tcn = TemporalConvNet(
            kernel_size=kernel_size,
            num_inputs=in_channels,
            num_channels=list(num_channels)+[in_channels],
            dropout=dropout
        )
        self.linear = nn.Linear(seq_len_x, seq_len_y)
        self.in_channels = in_channels
        self.seq_len_x = seq_len_x
        self.seq_len_y = seq_len_y

    def _shared_step(self, x, y):
        x = x.reshape(-1, self.in_channels, self.seq_len_x)

        y_pred = self.tcn(x)
        y_pred = self.linear(y_pred)
        b_size = x.size(0)
        y_pred = y_pred.view(b_size, self.d_seq_out, self.d_features)

        return y_pred, y


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
