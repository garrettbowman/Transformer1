import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_size: int):
        super(InputEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding_size = embedding_size

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.embedding_size)
    
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size: int, dropout: float, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding_size = embedding_size

        pe = torch.zeros(max_len, embedding_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-math.log(10000.0) / embedding_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x * math.sqrt(self.embedding_size)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)