import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class HFTTransformerModel(nn.Module):
    def __init__(self, feature_dim, num_pairs=1, d_model=4, nhead=1, num_layers=1, dim_feedforward=8, dropout=0.0):
        super(HFTTransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.feature_dim = feature_dim

        # Linear layer to map input features to d_model
        self.input_linear = nn.Linear(feature_dim, d_model)

        self.pos_encoder = PositionalEncoding(d_model, dropout)
        # Using batch_first=True for performance.
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # Output layer (predicting single scalar: signal/OBI)
        self.output_linear = nn.Linear(d_model, 1)
        self.d_model = d_model

    def forward(self, src):
        # src shape with batch_first=True: (batch_size, seq_len, feature_dim)
        # However, our pos_encoder might expect (seq_len, batch_size, d_model)
        # Let's adjust PositionalEncoding or src dimensions.
        # Original pos_encoder expects (seq_len, batch_size, d_model)

        src = self.input_linear(src) * math.sqrt(self.d_model)
        # If input is (seq_len, batch_size, feature_dim), pos_encoder handles it:
        src = self.pos_encoder(src)
        # But we made batch_first=True, so transformer_encoder expects (batch_size, seq_len, d_model)
        src = src.transpose(0, 1) # Convert (seq_len, batch_size, d_model) to (batch_size, seq_len, d_model)

        output = self.transformer_encoder(src)

        # After transformer, output is (batch_size, seq_len, d_model)
        # Use the last time step's output to make the prediction
        output = self.output_linear(output[:, -1, :])
        return output
