import torch.nn as nn

from models.TransformerEncoder.feed_forward import FeedForward
from models.TransformerEncoder.attention import MultiHeadAttention


class EncoderBlock(nn.Module):
    """Encoder Transformer block, containing multi-head self-attention and feed-forward network.
    Args:
        d_model: dimension of latent space
        d_ff: dimension in FFN
        d_qk: dimension of Q and K
        d_v: dimension of V
        h: number of attn heads
        mask: apply masking during training
        dropout: percentage for dropout layer
    """
    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 d_qk: int,
                 d_v: int,
                 n_heads: int,
                 mask: bool = False,
                 dropout: float = 0.1):
        super().__init__()

        self.MHA = MultiHeadAttention(d_model=d_model, d_qk=d_qk, d_v=d_v, n_heads=n_heads, mask=mask)
        self.feedforward = FeedForward(d_model=d_model, d_hidden=d_ff)
        self.dropout = nn.Dropout(p=dropout)
        self.layerNormal_1 = nn.LayerNorm(d_model)
        self.layerNormal_2 = nn.LayerNorm(d_model)

    def forward(self, x, stage, device):
        residual = x
        x = self.layerNormal_1(x)  # normalize layer before attn to improve learning stability
        x, score = self.MHA(x, stage, device)
        x = self.dropout(x)
        x = x + residual

        residual = x
        x = self.layerNormal_2(x)
        x = self.feedforward(x)
        x = self.dropout(x)
        x = x + residual

        return x, score