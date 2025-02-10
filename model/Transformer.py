import math
import torch

import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create constant 'pe' matrix with values dependent on 
        # pos and i
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # For odd d_model, handle last dimension separately
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_length, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, seq_length, d_model=64, nhead=4, num_encoder_layers=3, dim_feedforward=128, dropout=0.1):
        """
        Args:
            input_dim: number of features in the input time series.
            output_dim: number of features to predict.
            seq_length: length of the input sequence.
            d_model: dimension of the embedding space.
            nhead: number of heads in the multiheadattention models.
            num_encoder_layers: number of transformer encoder layers.
            dim_feedforward: dimension of the feedforward network model.
            dropout: dropout value.
        """
        super(TimeSeriesTransformer, self).__init__()

        self.seq_length = seq_length
        self.d_model = d_model

        # Linear projection to embedding dimension
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_length, dropout=dropout)

        # Transformer Encoder Layers
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)

        # Final regression/classification head.
        # Here we aggregate over the sequence. One option is to take the last time step.
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, src):
        """
        Args:
            src: Tensor of shape (batch_size, seq_length, input_dim)
        Returns:
            out: Tensor of shape (batch_size, output_dim)
        """
        # Project input to d_model
        src = self.input_projection(src)  # shape (batch_size, seq_length, d_model)
        src = self.pos_encoder(src)
        # Pass through the transformer encoder
        memory = self.transformer_encoder(src)  # shape (batch_size, seq_length, d_model)
        # For forecasting/regression, one common approach is to use the last time step's representation.
        last_hidden = memory[:, -1, :]  # shape (batch_size, d_model)
        out = self.fc_out(last_hidden)
        return out