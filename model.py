#Making Transformer model
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
        pe = torch.zeros(max_len, embedding_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-math.log(10000.0) / embedding_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) 
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape(1), :].requires_grad_(False)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    def __init__(self, embedding_size: int, eps: float = 1e-6):
        super(LayerNormalization, self).__init__()
        self.alpha = nn.Parameter(torch.ones(embedding_size)) #multiplied
        self.beta = nn.Parameter(torch.zeros(embedding_size)) #added
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.beta
    
class FeedForwardBlock(nn.Module):
    def __init__(self, embedding_size: int, hidden_size: int, dropout: float):
        super(FeedForwardBlock, self).__init__()
        self.linear1 = nn.Linear(embedding_size, hidden_size) #W1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_size, embedding_size) #W2 and b2

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))