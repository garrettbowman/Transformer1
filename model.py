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
    
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, embedding_size: int, num_heads: int, dropout: float):
        super(MultiHeadAttentionBlock, self).__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        assert embedding_size % num_heads == 0,"embedding size not be divisible by num_heads"
        
        self.head_size = embedding_size // num_heads
        self.scale = math.sqrt(self.head_size)
        self.dropout = nn.Dropout(dropout)
        self.query = nn.Linear(embedding_size, embedding_size)
        self.key = nn.Linear(embedding_size, embedding_size)
        self.value = nn.Linear(embedding_size, embedding_size)
        self.output = nn.Linear(embedding_size, embedding_size)

    def forward(self, x):
        batch_size = x.shape[0]
        query = self.query(x).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        key = self.key(x).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        value = self.value(x).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        attention = self.attention(query, key, value)
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.embedding_size)
        return self.output(attention)

    @staticmethod
    def attention(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value)
    
class ResidualConnectionBlock(nn.Module):
    def __init__(self, embedding_size: int, dropout: float):
        super(ResidualConnectionBlock, self).__init__()
        self.layer_norm = LayerNormalization(embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.layer_norm(x)))
    
class EncoderBlock(nn.Module):
    def __init__(self, embedding_size: int, num_heads: int, hidden_size: int, dropout: float):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttentionBlock(embedding_size, num_heads, dropout)
        self.feed_forward = FeedForwardBlock(embedding_size, hidden_size, dropout)
        self.residual_connection = ResidualConnectionBlock(embedding_size, dropout)

    def forward(self, x, src_mask):
        x = self.residual_connection(x, lambda x: self.attention(x, x, x, src_mask))
        return self.residual_connection(x, self.feed_forward)

class Encoder(nn.Module):
    def __init__(self, layers, ModuleList) -> None:
        super.__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.Module([ResidualConnectionBlock(dropout) for _ in range(3)])
        
        def forward(self, x, encoder_output, src_mask, tgt_mask):
            x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
            x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
            return self.residual_connection[2](x, self.feed_forward_block)
        
        
class Decoder(nn.Module):
    def __init__(self, layers, ModuleList) -> None:
        super.__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)        

class ProjectionLayer(nn.Module):
    def __init__(self, embedding_size: int, vocab_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(embedding_size, vocab_size)
        
    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=-1)