import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embd_size, heads):
        super(SelfAttention, self).__init__()

        self.embd_size = embd_size
        self.heads = heads
        self.heads_dim = embd_size // heads

        assert(self.heads_dim * heads == embd_size), "Embd size needs to be divisible by heads"

        # self.values = nn.Linear(self.heads_dim, self.heads_dim, bias=False)
        # self.query = nn.Linear(self.heads_dim, self.heads_dim, bias=False)
        # self.keys = nn.Linear(self.heads_dim, self.heads_dim, bias=False)

        # self.fc_out = nn.Linear(heads * self.heads_dim, embd_size)

        self.values = nn.Linear(embd_size, embd_size, bias=False)
        self.query = nn.Linear(embd_size, embd_size, bias=False)
        self.keys = nn.Linear(embd_size, embd_size, bias=False)

        self.fc_out = nn.Linear(embd_size, embd_size)

    def forward(self, x, mask, kv_cache=None):
        N = x.shape[0]
        
        # Query len = Key len = Value len
        seq_len = x.shape[1]

        values = self.values(x)
        keys = self.keys(x)
        query = self.query(x)

        # Splitting embeddings into self.heads pieces
        values = values.reshape(N, seq_len, self.heads, self.heads_dim)
        keys = keys.reshape(N, seq_len, self.heads, self.heads_dim)
        query = query.reshape(N, seq_len, self.heads, self.heads_dim)

        # attention = softmax( query . key // embd**0.5).value

        query_transpose = query.permute(0, 2, 1, 3)
        keys_transpose = keys.permute(0, 2, 3, 1)
        value_transpose = values.permute(0, 2, 1, 3)

        # KV Caching
        if kv_cache is not None:
            if "k" in kv_cache:
                # Concatenate new K, V with cached K, V from previous steps
                keys_transpose = torch.cat([kv_cache["k"], keys_transpose], dim=3)
                value_transpose = torch.cat([kv_cache["v"], value_transpose], dim=2)

            # Update cache with the full K, V (past + new)
            kv_cache["k"] = keys_transpose
            kv_cache["v"] = value_transpose


        energy = torch.matmul(query_transpose, keys_transpose)

        if mask is not None:
            energy = energy.masked_fill(mask==0, float("-1e20"))

        attention = torch.softmax(energy / self.embd_size**(1/2), dim=-1)

        output = torch.matmul(attention, value_transpose)

        output = output.reshape(N, seq_len, self.heads*self.heads_dim)

        output = self.fc_out(output) # we use this linear layer to allow different heads to interact with each other

        return output
    
class TransformerBlock(nn.Module):
    def __init__(self, embd_size, heads, dropout_p, forward_expansion):
        super(TransformerBlock, self).__init__()

        self.attention = SelfAttention(embd_size=embd_size, heads=heads)
        self.norm_1 = nn.LayerNorm(embd_size)
        self.norm_2 = nn.LayerNorm(embd_size)

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(embd_size, forward_expansion * embd_size),
            nn.GELU(),  # GELU is commonly used in modern transformers
            nn.Linear(forward_expansion * embd_size, embd_size)
        )

        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x, mask=None):

        # pre-normalisation 1
        attention_output = self.attention(self.norm_1(x), mask)

        x = x + self.dropout(attention_output)  

        # pre-normalisation 2
        ff_out = self.ffn(self.norm_2(x))
        x = x + self.dropout(ff_out)

        return x


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, embd_size, heads, dropout_p, forward_expansion, max_len, num_layers, device):
        super(DecoderOnlyTransformer, self).__init__()

        self.device = device
        self.embd_size = embd_size
        self.max_len = max_len

        # defining encodings
        self.positional_embedding = nn.Embedding(max_len, embd_size)
        self.word_embedding = nn.Embedding(vocab_size, embd_size)

        self.layers = nn.ModuleList(
            [TransformerBlock(embd_size=embd_size, heads=heads, forward_expansion=forward_expansion, dropout_p=dropout_p)
             for _ in range(num_layers)]
        )
        
        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(embd_size)

        self.fc_out = nn.Linear(embd_size, vocab_size, bias=False)
        self.dropout = nn.Dropout(dropout_p)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def create_causal_mask(self, N, seq_len):
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(self.device)
        return mask.unsqueeze(0).unsqueeze(0).expand(N, 1, seq_len, seq_len)

    def forward(self, x, mask, labels=None):
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)

        x = self.dropout(self.word_embedding(x) + self.positional_embedding(positions))

        for layer in self.layers:
            x = layer(x, mask)

        x = self.ln_f(x)
        out = self.fc_out(x)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            # Reshape for loss: (N * seq_len, vocab_size)
            loss = loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
            return loss, out

        return out