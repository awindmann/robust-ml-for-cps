import torch
import torch.nn as nn
from models.core import PLCore
import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm


class TcnAe(PLCore):
    def __init__(
        self,
        enc_tcn1_in_dims: list = [3, 50, 40, 30],
        enc_tcn1_out_dims: list = [50, 40, 30, 10],
        enc_tcn2_in_dims: list = [10, 8, 6, 3],
        enc_tcn2_out_dims: list = [8, 6, 3, 1],
        latent_dim: int = 3,
        seq_len_x: int = 250,
        seq_len_y: int = 50,
        dec_y_tcn1_seq_len: int= 50,
        dec_y_tcn2_seq_len: int=50,
        dec_x_tcn1_seq_len: int= 100,
        dec_x_tcn2_seq_len: int=250,
        kernel_size: int = 5,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.save_hyperparameters()

        dec_tcn1_in_dims = enc_tcn2_out_dims[::-1]
        dec_tcn1_out_dims = enc_tcn2_in_dims[::-1]
        dec_tcn2_in_dims = enc_tcn1_out_dims[::-1]
        dec_tcn2_out_dims = enc_tcn1_in_dims[::-1]
        self.in_channels = enc_tcn1_in_dims[0]
        self.seq_len_x = seq_len_x
        self.seq_len_y = seq_len_y

        self.encoder = Encoder(
            tcn1_in_dims=enc_tcn1_in_dims,
            tcn1_out_dims=enc_tcn1_out_dims,
            tcn2_in_dims=enc_tcn2_in_dims,
            tcn2_out_dims=enc_tcn2_out_dims,
            kernel_size=kernel_size,
            latent_dim=latent_dim,
            seq_len=seq_len_x,
        )

        self.x_decoder = Decoder(
            tcn1_in_dims=dec_tcn1_in_dims,
            tcn1_out_dims=dec_tcn1_out_dims,
            tcn2_in_dims=dec_tcn2_in_dims,
            tcn2_out_dims=dec_tcn2_out_dims,
            kernel_size=kernel_size,
            latent_dim=latent_dim,
            tcn1_seq_len=dec_x_tcn1_seq_len,
            tcn2_seq_len=dec_x_tcn2_seq_len,
        )

        self.y_decoder = Decoder(
            tcn1_in_dims=dec_tcn1_in_dims,
            tcn1_out_dims=dec_tcn1_out_dims,
            tcn2_in_dims=dec_tcn2_in_dims,
            tcn2_out_dims=dec_tcn2_out_dims,
            kernel_size=kernel_size,
            latent_dim=latent_dim,
            tcn1_seq_len=dec_y_tcn1_seq_len,
            tcn2_seq_len=dec_y_tcn2_seq_len,
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.x_decoder(z), self.y_decoder(z)

    def forward(self, x):
        x = x.reshape(-1, self.in_channels, self.seq_len_x)
        z = self.encode(x)
        _, y_hat = self.decode(z)
        return y_hat.reshape(-1, self.seq_len_y, self.in_channels)

    @staticmethod
    def loss_function(x_hat, x, y_hat, y):
        y_loss = nn.MSELoss()(y_hat, y)
        x_loss = nn.MSELoss()(x, x_hat)
        loss = y_loss + x_loss
        return loss, x_loss, y_loss

    def _shared_step(self, x, y):
        x = x.reshape(-1, self.in_channels, self.seq_len_x)
        y = y.reshape(-1, self.in_channels, self.seq_len_y)
        z = self.encode(x)
        x_hat, y_hat = self.decode(z)
        loss, x_loss, y_loss = self.loss_function(x_hat, x, y_hat, y)
        return z, x_loss, y_loss, loss, y, y_hat

    def training_step(self, batch, batch_id):
        """Training step for pytorch lightning."""
        x, y = batch
        z, x_loss, y_loss, loss, target, pred = self._shared_step(x, y)
        for name, metric in self.train_metrics.items():
            metric_loss = metric(pred, target)
            self.log("train_" + name, metric_loss)
            if name == self.loss_fct_key:
                # use this loss function for backpropagation
                loss = metric_loss
        return loss

    def validation_step(self, batch, batch_id, dataloader_idx):
        """Validation step for pytorch lightning."""
        x, y = batch
        z, x_loss, y_loss, loss, target, pred = self._shared_step(x, y)
        for name, metric in self.val_metrics.items():
            metric_loss = metric(pred, target)
            self.log("val_" + name, metric_loss)
            if name == self.loss_fct_key and dataloader_idx == 0:
                # save epoch losses on standard dataset for logging
                self.validation_step_outputs.append(metric_loss)

    def test_step(self, batch, batch_id, dataloader_idx):
        """Test step for pytorch lightning."""
        x, y = batch
        z, x_loss, y_loss, loss, target, pred = self._shared_step(x, y)
        for name, metric in self.test_metrics.items():
            self.log("test_" + name, metric(pred, target))


class Conv1DResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, padding, stride, dilation):
        super(Conv1DResidualBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                in_dim,
                out_dim,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.activation = nn.ReLU()
        self.conv2 = weight_norm(
            nn.Conv1d(
                out_dim,
                out_dim,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.residual = nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        res = self.residual(x)
        return out + res


class TCN(nn.Module):
    def __init__(self, in_dims: list, out_dims: list, kernel_size: int):
        super(TCN, self).__init__()
        layers = nn.ModuleList()
        for i in range(len(in_dims)):
            dilation = 2**i
            layers.append(
                Conv1DResidualBlock(
                    in_dim=in_dims[i],
                    kernel_size=kernel_size,
                    out_dim=out_dims[i],
                    padding="same",
                    dilation=dilation,
                    stride=1,
                )
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(
        self,
        tcn1_in_dims: list,
        tcn1_out_dims: list,
        tcn2_in_dims: list,
        tcn2_out_dims: list,
        kernel_size: int = 15,
        latent_dim: int = 3,
        fc_in_out_dim: int = 10,
        tcn1_seq_len: int = 50,
        tcn2_seq_len: int = 50,
    ) -> None:
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim

        self.fc_in = nn.Linear(
            in_features=latent_dim,
            out_features=int(
                fc_in_out_dim * tcn1_in_dims[0],
            ),
        )

        # tcn1
        self.upsampler1 = torch.nn.Upsample(size=tcn1_seq_len, mode="nearest")
        self.tcn1 = TCN(
            in_dims=tcn1_in_dims,
            out_dims=tcn1_out_dims,
            kernel_size=kernel_size,
        )

        # tcn2
        self.upsampler2 = torch.nn.Upsample(size=tcn2_seq_len, mode="nearest")
        self.tcn2 = TCN(
            in_dims=tcn2_in_dims,
            out_dims=tcn2_out_dims,
            kernel_size=kernel_size,
        )

    def forward(self, z):
        # fc in
        out = self.fc_in(z).reshape(-1, 1, self.fc_in.out_features)
        out = self.upsampler1(out)
        out = self.tcn1(out)
        out = self.upsampler2(out)
        out = self.tcn2(out)
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        tcn1_in_dims: list,
        tcn1_out_dims: list,
        tcn2_in_dims: list,
        tcn2_out_dims: list,
        kernel_size: int = 15,
        latent_dim: int = 10,
        seq_len: int = 1000,
    ) -> None:
        super(Encoder, self).__init__()

        # TCN1
        self.tcn1 = TCN(
            in_dims=tcn1_in_dims, out_dims=tcn1_out_dims, kernel_size=kernel_size
        )
        self.max_pool1 = nn.MaxPool1d(kernel_size=2)

        # TCN2
        self.tcn2 = TCN(
            in_dims=tcn2_in_dims, out_dims=tcn2_out_dims, kernel_size=kernel_size
        )
        self.max_pool2 = nn.MaxPool1d(kernel_size=2)

        self.fc_out = nn.Linear(
            in_features=int(0.25 * seq_len * tcn2_out_dims[-1]), out_features=latent_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.tcn1(x)
        out = self.max_pool1(out)
        out = self.tcn2(out)
        out = self.max_pool2(out)
        out = out.flatten(start_dim=1)
        out = self.fc_out(out)
        return out
