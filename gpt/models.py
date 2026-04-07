import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from config import BaseConfig


class BaseModel(nn.Module, ABC):
    """Abstract base model class"""
    
    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config
    
    @abstractmethod
    def forward(self, x):
        """Forward pass - implement in subclass"""
        pass
    
    def get_num_params(self) -> int:
        """Count total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_params(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
class SelfAttention(nn.Module):
    def __init__(self, embd_size, heads):
        super(SelfAttention, self).__init__()

        self.embd_size = embd_size
        self.heads = heads
        self.heads_dim = embd_size // heads

        assert(self.heads_dim * heads == embd_size), "Embd size needs to be divisible by heads"

        self.values = nn.Linear(embd_size, embd_size, bias=False)
        self.query = nn.Linear(embd_size, embd_size, bias=False)
        self.keys = nn.Linear(embd_size, embd_size, bias=False)

        self.fc_out = nn.Linear(embd_size, embd_size)

    def forward(self, x, mask):

        batch_size = x.shape[0]
        seq_len = x.shape[1]

        values = self.values(x)
        keys = self.keys(x)
        query = self.query(x)

        # Reshaping keys, query, values matrix
        values = values.reshape(batch_size, seq_len, self.heads, self.heads_dim)
        query = query.reshape(batch_size, seq_len, self.heads, self.heads_dim)
        keys = keys.reshape(batch_size, seq_len, self.heads, self.heads_dim)

        query_t = query.permute(0, 2, 1, 3)
        key_t = keys.permute(0, 2, 3, 1)
        values_t = values.permute(0, 2, 1, 3)

        scores = torch.matmul(query_t, key_t) / self.heads_dim**0.5

        if mask is not None:
            scores = scores.masked_fill(mask==0, float("-1e20"))

        # Calculating attention score
        attention_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, values_t)

        output = context.reshape(batch_size, seq_len, self.heads*self.heads_dim)

        output = self.fc_out(output)

        return output
    
class TransformerBlock(nn.Module):
    def __init__(self, forward_expansion, embd_size, heads, dropout_p):
        super(TransformerBlock, self).__init__()

        self.attention = SelfAttention(embd_size=embd_size, heads=heads)
        self.norm1 = nn.LayerNorm(embd_size)
        self.norm2 = nn.LayerNorm(embd_size)

        self.ffn = nn.Sequential(
            nn.Linear(embd_size, forward_expansion*embd_size),
            nn.GELU(),
            nn.Linear(forward_expansion*embd_size, embd_size)
        )

        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x, mask=None):
        
        # pre-normalization 1
        attention_output = self.attention(self.norm1(x), mask)

        x = x + self.dropout(attention_output)

        # pre-normalization 2
        feed_forward_output = self.ffn(self.norm2(x))
        x = x + self.dropout(feed_forward_output)

        return x
    
class GPT(BaseModel):
    def __init__(self, config, vocab_size, embd_size, heads, dropout_p, forward_expansion, max_len, num_layers, device):
        super(GPT, self).__init__(config=config)

        self.device = device
        self.embd_size = embd_size
        self.max_len = max_len

        self.positional_embedding = nn.Embedding(max_len, embd_size)
        self.word_embedding = nn.Embedding(vocab_size, embd_size)
        self.layer_norm_final = nn.LayerNorm(embd_size)
        self.fc_out = nn.Linear(embd_size, vocab_size)
        self.dropout = nn.Dropout(dropout_p)

        self.layers = nn.ModuleList(
            [TransformerBlock(embd_size=embd_size, heads=heads, dropout_p=dropout_p, forward_expansion=forward_expansion) for _ in range(num_layers)]
        )

    def forward(self, x, mask):
        batch_size, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(batch_size, seq_len).to(self.device)
        
        x = self.dropout(self.word_embedding(x) + self.positional_embedding(positions))

        for layer in self.layers:
            x = layer(x, mask)

        x = self.layer_norm_final(x)
        out = self.fc_out(x)
        
        return out
    
    def create_causal_mask(self, seq_len):
        # Lower-triangular matrix: allow past + present, block future
        return torch.tril(torch.ones(seq_len, seq_len, device=self.device)).bool()