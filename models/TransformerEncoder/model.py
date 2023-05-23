import torch
import torch.nn as nn

from models.core import PLCore
from models.TransformerEncoder.encoderblock import EncoderBlock
from models.TransformerEncoder.embedding import PositionalEncoding, ConvEmbedding


class TransformerEncoder(PLCore):
    """Transformer model.
    Args:
        d_model: dimension of latent space
        d_ff: dimension in FFN
        d_qk: dimension of Q and K
        d_v: dimension of V
        h: number of attn heads
        n_layers: number of encoder layers
        conv_kernel_size: kernel size for conv embedding
        conv_stride: stride for conv embedding
        conv_dilation: dilation for conv embedding
        dropout: percentage for dropout layer
        mask: apply masking during training
        use_pos_enc: use positional encoding
    """
    def __init__(
        self,
        d_model: int = 512,
        d_ff: int = 1028,
        d_qk: int = 8,
        d_v: int = 8,
        n_heads: int = 8,
        n_layers: int = 8,
        conv_kernel_size: int = 9,
        conv_stride: int = 1,
        conv_dilation: int = 1,
        dropout: float = 0.1,
        mask: bool = False,
        use_pos_enc=True,
        **kwargs
    ):
        super().__init__(**kwargs)

        # self.d_seq_in/out and self.d_features from parent class

        # encoder (Transformer blocks)
        self.encoder_list = nn.ModuleList([EncoderBlock(d_model=d_model,
                                                        d_ff=d_ff,
                                                        d_qk=d_qk,
                                                        d_v=d_v,
                                                        n_heads=n_heads,
                                                        mask=mask,
                                                        dropout=dropout)
                                           for _ in range(n_layers)])

        # embedding of the time series
        self.use_pos_enc = use_pos_enc
        self.positional_encoding = PositionalEncoding(self.d_seq_in, d_model)
        self.embedding_channel = ConvEmbedding(self.d_features, d_model, conv_kernel_size, conv_stride, conv_dilation)

        # output layer
        seq_length = int((self.d_seq_in - conv_dilation * (conv_kernel_size - 1) - 1) / conv_stride + 1)  # from conv embedding
        self.output_linear = nn.Linear(d_model * seq_length, self.d_seq_out * self.d_features)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # self.output_linear.bias.data.zero_()
        # self.output_linear.weight.data.uniform_(-0.1, 0.1)

    def _shared_step(self, x, y, stage):
        """Shared step used in training, validation and test."""
		# x1 = x, x2 = y
        b_size = x.shape[0]

        encoding = self.embedding_channel(x)

        if self.use_pos_enc:
            encoding = self.positional_encoding(encoding)
        for encoder in self.encoder_list:
            encoding, score_input = encoder(encoding, stage=stage, device=self.device)  # stage: masking in train only
        encoding = encoding.reshape(encoding.shape[0], -1)

        pred = self.output_linear(encoding)
        pred = pred.reshape(b_size, self.d_seq_out, self.d_features)

        return pred, y

    @torch.no_grad()
    def forward(self, x):
        """Forward pass for pytorch lightning.
        Should return the prediction (y_pred)."""
        return self._shared_step(x, None, stage="test")[0]
    
    def training_step(self, batch, batch_id):
        """Training step for pytorch lightning."""
        x1, x2 = batch
        pred, target = self._shared_step(x1, x2, stage="train")
        for name, metric in self.train_metrics.items():
            metric_loss = metric(pred, target)
            self.log("train_" + name, metric_loss)
            if name == self.loss_fct_key:
                # use this loss function for backpropagation
                loss = metric_loss
        return loss

    def validation_step(self, batch, batch_id, dataloader_idx):
        """Validation step for pytorch lightning."""
        x1, x2 = batch
        pred, target = self._shared_step(x1, x2, stage="test")
        for name, metric in self.val_metrics.items():
            metric_loss = metric(pred, target)
            self.log("val_" + name, metric_loss)
            if name == self.loss_fct_key and dataloader_idx == 0:
                # save epoch losses on standard dataset for logging
                self.validation_step_outputs.append(metric_loss)
        

    def test_step(self, batch, batch_id, dataloader_idx):
        """Test step for pytorch lightning."""
        x1, x2 = batch
        pred, target = self._shared_step(x1, x2, stage="test")
        for name, metric in self.test_metrics.items():
            self.log("test_" + name, metric(pred, target))
