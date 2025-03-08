# model.py
import torch
import torch.nn as nn
import math

# Positional Encoding class to add position information to token embeddings
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# Simple Transformer model for text generation
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=4, d_ff=1024, max_seq_len=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, n_heads, d_ff, dropout=0.1),
            num_layers=n_layers
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.max_seq_len = max_seq_len

    def forward(self, x, mask=None):
        # x: input tensor of shape (batch_size, seq_len)
        # mask: attention mask to prevent attending to future tokens
        batch_size, seq_len = x.size()
        x = self.embedding(x) * math.sqrt(self.d_model)  # Shape: (batch_size, seq_len, d_model)
        x = self.pos_encoder(x)  # Add positional encodings
        
        # Transpose x to (seq_len, batch_size, d_model) as expected by TransformerDecoder
        x = x.transpose(0, 1)  # Shape: (seq_len, batch_size, d_model)
        
        if mask is None:
            # Generate a causal mask of shape (seq_len, seq_len)
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Ensure mask is compatible with batch processing
        output = self.transformer(x, x, tgt_mask=mask)  # Shape: (seq_len, batch_size, d_model)
        
        # Transpose back to (batch_size, seq_len, d_model) for the output layer
        output = output.transpose(0, 1)  # Shape: (batch_size, seq_len, d_model)
        return self.fc_out(output)  # Shape: (batch_size, seq_len, vocab_size)