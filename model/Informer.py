import torch
import math

import torch.nn as nn
import torch.nn.functional as F


class ProbSparseAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, topk_factor=0.5):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.topk_factor = topk_factor
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # q, k, v shape: (batch, seq_len, d_model)
        bs, len_q, _ = q.size()
        bs, len_k, _ = k.size()

        Q = self.q_linear(q).view(bs, len_q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(k).view(bs, len_k, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(v).view(bs, len_k, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (bs, n_heads, len_q, len_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Approximate ProbSparse: select top-k keys for each query
        topk = max(1, int(self.topk_factor * scores.size(-1)))
        topk_scores, topk_indices = torch.topk(scores, topk, dim=-1)
        attn = torch.zeros_like(scores)
        attn.scatter_(-1, topk_indices, F.softmax(topk_scores, dim=-1))
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)  # (bs, n_heads, len_q, d_k)
        context = context.transpose(1, 2).contiguous().view(bs, len_q, self.n_heads * self.d_k)
        output = self.out_linear(context)
        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = ProbSparseAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention
        new_x = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(new_x))
        # Feed-forward network
        new_x = self.ff(x)
        x = self.norm2(x + self.dropout(new_x))
        return x


class Encoder(nn.Module):
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = ProbSparseAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = ProbSparseAttention(d_model, n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        # Self-attention in decoder
        new_x = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(new_x))
        # Cross-attention with encoder output
        new_x = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout(new_x))
        # Feed-forward network
        new_x = self.ff(x)
        x = self.norm3(x + self.dropout(new_x))
        return x


class Decoder(nn.Module):
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(num_layers)])

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x


class Informer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=512, n_heads=8, d_ff=2048,
                 num_encoder_layers=3, num_decoder_layers=2, dropout=0.1, max_len=5000):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Embedding layers for source and target
        self.enc_embedding = nn.Linear(input_dim, d_model)
        self.dec_embedding = nn.Linear(output_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)

        # Encoder and Decoder
        encoder_layer = EncoderLayer(d_model, n_heads, d_ff, dropout)
        self.encoder = Encoder(encoder_layer, num_encoder_layers)

        decoder_layer = DecoderLayer(d_model, n_heads, d_ff, dropout)
        self.decoder = Decoder(decoder_layer, num_decoder_layers)

        # Final projection layer to map to output dimensions
        self.projection = nn.Linear(d_model, output_dim)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        src: (batch, src_len, input_dim)
        tgt: (batch, tgt_len, output_dim)
        """
        # Encoder
        enc_input = self.enc_embedding(src)
        enc_input = self.pos_enc(enc_input)
        enc_out = self.encoder(enc_input, src_mask)

        # Decoder
        dec_input = self.dec_embedding(tgt)
        dec_input = self.pos_enc(dec_input)
        dec_out = self.decoder(dec_input, enc_out, src_mask, tgt_mask)

        output = self.projection(dec_out)
        return output


if __name__ == "__main__":
    # Dummy multivariate time series data
    batch_size = 16
    src_len = 96   # Length of input/past sequence
    tgt_len = 24   # Forecast horizon
    input_dim = 10  # Number of input variables
    output_dim = 10  # Number of output variables (forecast targets)

    # Generate random source and target sequences
    src = torch.randn(batch_size, src_len, input_dim)
    tgt = torch.randn(batch_size, tgt_len, output_dim)

    # Create Informer model instance (using a smaller model for demonstration)
    model = Informer(
        input_dim=input_dim,
        output_dim=output_dim,
        d_model=64,
        n_heads=4,
        d_ff=128,
        num_encoder_layers=2,
        num_decoder_layers=1,
        dropout=0.1,
        max_len=100
    )

    # Forward pass: the output shape should be (batch_size, tgt_len, output_dim)
    output = model(src, tgt)
    print("Output shape:", output.shape)