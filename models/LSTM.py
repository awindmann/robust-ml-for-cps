import torch
import torch.nn as nn

from models.core import PLCore



class LSTM(PLCore):
	"""LSTM model.
	Args:
		d_hidden (int): hidden dimension
		n_layers (int): number of layers
		bidirectional (bool): whether to use bidirectional LSTM
		dropout (float): dropout rate
		autoregressive (bool): whether to predict one output at a time
	"""
	def __init__(self, d_hidden, n_layers=1, bidirectional=False, dropout=0.5, autoregressive=False, **kwargs):
		super().__init__(**kwargs)
		
		# self.d_features and self.d_seq_out from parent class

		self.lstm = nn.LSTM(
			input_size=self.d_features,
			hidden_size=d_hidden,
			num_layers=n_layers,
			bidirectional=bidirectional,
			batch_first=True,
			dropout=dropout if n_layers > 1 else 0
		)
		
		if not autoregressive:
			# Predict all outputs at once
			self.fc = nn.Linear(d_hidden * (bidirectional + 1), self.d_features * self.d_seq_out)
		else:
			# Predict one output at a time
			self.fc = nn.Linear(d_hidden * (bidirectional + 1), self.d_features)
		self.autoregressive = autoregressive

	def _shared_step(self, x, y):
		# x: (batch_size, d_seq_in, d_features)
		# y: (batch_size, d_seq_out, d_features)
		b_size = x.size(0)

		if not self.autoregressive:
			_, (h, _) = self.lstm(x)
			h = h[-1, :, :]
			y_pred = self.fc(h).view(b_size, self.d_seq_out, self.d_features)
		else:
			y_pred = []
			_, (h, c) = self.lstm(x)
			output = self.fc(h[-1, :, :]).unsqueeze(1)
			y_pred.append(output)

			# Autoregressive forecasting
			for _ in range(self.d_seq_out - 1):
				_, (h, c) = self.lstm(output, (h, c))
				output = self.fc(h[-1, :, :]).unsqueeze(1)
				y_pred.append(output)

			y_pred = torch.cat(y_pred, dim=1)

		return y_pred, y