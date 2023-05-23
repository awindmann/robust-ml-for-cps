import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import MeanSquaredError, MeanAbsoluteError
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import train_arguments as args



class PLCore(pl.LightningModule):
    """pytorch lightning core module
    This module is the base class for all models.
    It implements the training, validation and test steps.
    Args:
        d_seq_in (int): input sequence length
        d_features (int): number of features
        d_seq_out (int): output sequence length
        train_scenario (str): the scenario to train on
        lr (float): learning rate
        beta1 (float): beta1 for Adam optimizer
        beta2 (float): beta2 for Adam optimizer
        eps (float): epsilon for Adam optimizer
    """
    def __init__(self, d_seq_in=250, d_features=3, d_seq_out=50, 
                 train_scenario="standard",
                 lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__()

        self.d_seq_in = d_seq_in
        self.d_features = d_features
        self.d_seq_out = d_seq_out
        
        self.example_input_array = torch.rand(32, self.d_seq_in, self.d_features)  # 32 as example batch size

        self.save_hyperparameters()  # stores hyperparameters in self.hparams and allows logging
        self.visualization_device = "cpu"  # for visualizations, can be changed in model if necessary

        scenario_dict = {
            "standard": 0,
            "fault": 1,
            "noise": 2,
            "duration": 3,
            "scale": 4,
            "switch": 5,
            "q1+v3": 6,
            "q1+v3+rest": 7,
            "v12+v23": 8,
            "standard+": 9,
            "standard++": 10,
            "frequency": 11,
            "time_warp": 12
        }
        self.train_scenario_idx = scenario_dict[train_scenario]
        
        # metrics to keep track of
        metrics = {
            "MAE": MeanAbsoluteError(),
            "MSE": MeanSquaredError()
        }
        # the loss function to use for backpropagation
        self.loss_fct_key = args.LOSS_FCT  
        assert self.loss_fct_key in metrics.keys(), "loss function key should be in metrics"

        self.train_metrics = nn.ModuleDict({name: metric.clone() for name, metric in metrics.items()})
        self.val_metrics = nn.ModuleDict({name: metric.clone() for name, metric in metrics.items()})
        self.test_metrics = nn.ModuleDict({name: metric.clone() for name, metric in metrics.items()})
        self.validation_step_outputs = []
        self.min_epoch_val_loss = float("inf")

    def _shared_step(self, x, y):
        """Shared step used in training, validation and test step.
        Should return the prediction and the target (y_pred, y).
        """
        raise NotImplementedError("This should be implemented in the model that inherits from PLCore.")

    @torch.no_grad()
    def forward(self, x):
        """Forward pass for pytorch lightning.
        Should return the prediction (y_pred)."""
        return self._shared_step(x, None)[0]
    
    def training_step(self, batch, batch_id):
        """Training step for pytorch lightning.
        Only receives dataloader 0, which is the training dataloader.
        """
        x1, x2 = batch
        pred, target = self._shared_step(x1, x2)
        for name, metric in self.train_metrics.items():
            metric_loss = metric(pred, target)
            self.log("train_" + name, metric_loss, logger=True)
            if name == self.loss_fct_key:
                # use this loss function for backpropagation
                loss = metric_loss
        return loss

    def validation_step(self, batch, batch_id, dataloader_idx):
        """Validation step for pytorch lightning."""
        x1, x2 = batch
        pred, target = self._shared_step(x1, x2)
        for name, metric in self.val_metrics.items():
            metric_loss = metric(pred, target)
            self.log("val_" + name, metric_loss, logger=True)
            if name == self.loss_fct_key and dataloader_idx == self.train_scenario_idx:
                # save epoch losses on standard dataset for logging
                self.validation_step_outputs.append(metric_loss)

    def on_validation_epoch_end(self):
        """Validation epoch end for pytorch lightning.
        Only receives val losses from dataloader 0.
        """
        epoch_losses = torch.stack(self.validation_step_outputs)
        mean_loss = torch.mean(epoch_losses)
        self.log("ep_val_loss", mean_loss, prog_bar=True, logger=True)
        self.validation_step_outputs.clear()  # free memory
        # save if this is the best model so far
        if mean_loss < self.min_epoch_val_loss:
            self.min_epoch_val_loss = mean_loss

    def on_train_end(self):
        """Train end for pytorch lightning."""
        # log best val_loss
        if self.logger:
            self.logger.log_hyperparams(self.hparams, {"hp/min_epoch_val_loss": self.min_epoch_val_loss})

    def test_step(self, batch, batch_id, dataloader_idx):
        """Test step for pytorch lightning."""
        x1, x2 = batch
        pred, target = self._shared_step(x1, x2)
        for name, metric in self.test_metrics.items():
            self.log("test_" + name, metric(pred, target), logger=True)

    def configure_optimizers(self):
        """Configure optimizers for pytorch lightning."""
        optimizer = Adam(
            self.parameters(), 
            lr=self.hparams.lr,  
            betas=(self.hparams.beta1, self.hparams.beta2),
            eps=self.hparams.eps
            )
        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=25, min_lr=1e-5)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch", "monitor": f"val_{self.loss_fct_key}/dataloader_idx_0"}]
