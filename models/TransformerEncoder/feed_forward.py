import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_hidden: int = 1024):
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_hidden)
        self.linear_2 = nn.Linear(d_hidden, d_model)

    def forward(self, x):

        x = self.linear_1(x)
        x = F.relu(x)
        x = self.linear_2(x)

        return x