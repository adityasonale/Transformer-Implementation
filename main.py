import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embd_size, heads):
        super(SelfAttention, self).__init__()

        self.embd_size = embd_size
        self.heads = heads
        self.head_dim = embd_size // heads

        assert(self.head_dim * heads == embd_size), "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.fc_out = nn.Linear(heads*self.head_dim, embd_size)

    def forward(self, values, keys, queries, mask):

        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Split embeddings into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # energy = torch.einsum("nqhd, nkhd-->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, head_dim)
        # keys shape: (N, key_len, heads, head_dim)
        # values shape: (N, value_len, heads, head_dim)
        # energy shape: (N, heads, query_len, key_len)

        key_transposed = keys.permute(0, 2, 3, 1) # (n, h, d, k)
        
        queries_transposed = queries.permute(0, 2, 1, 3) # (n, h, q, d)

        energy = torch.matmul(queries_transposed, key_transposed)

        if mask is not None:
            energy = energy.masked_fill(mask==0, float("-1e20"))

        attention = torch.softmax(energy / (self.embd_size**(1/2)), dim=3) # (n, h , q, k)


        # attention @ value
        # (n, h, q, k) @ (n, k/v, h, d)

        value_transpose = values.permute(0, 2, 1, 3)

        output = torch.matmul(attention, value_transpose) # (n, h, q, d)

        output = output.reshape(N, query_len, self.heads*self.head_dim)

        output = self.fc_out(output)

        return output

class TransformerBlock(nn.Module):
    def __init__(self, embd_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()

        self.attention = SelfAttention(embd_size, heads)

        # Normalization
        self.norm1 = nn.LayerNorm(embd_size)
        self.norm2 = nn.LayerNorm(embd_size)

        # Feedforward
        self.feed_forward = nn.Sequential(
            nn.Linear(embd_size, forward_expansion * embd_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embd_size, embd_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention + query))

        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out
    
class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embd_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(Encoder, self).__init__()

        self.embed_size = embd_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embd_size)
        self.position_embedding = nn.Embedding(max_length, embd_size)

        self.layers = nn.ModuleList(
            [TransformerBlock(embd_size, heads, dropout, forward_expansion)
            for _ in range(num_layers)]
        )

        #You can think of nn.ModuleList as a specialized PyTorch data structure that:
        # Holds layers (or modules) just like a list,
        # But also registers them as part of your model.

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask) # Key, Query, Value all will be same

        return out
    
class DecoderBlock(nn.Module):
    def __init__(self, embd_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()

        self.attention = SelfAttention(embd_size, heads)
        self.norm = nn.LayerNorm(embd_size)
        self.transformer_block = TransformerBlock(
            embd_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x,x,x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        
        return out
    
class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embd_size, num_layers, heads, forward_expansion, dropout, device, max_length):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embd_size)
        self.position_embedding = nn.Embedding(max_length, embd_size)

        self.layers = nn.ModuleList(
            [DecoderBlock(embd_size, heads, forward_expansion, dropout, device)
             for _ in range(num_layers)]
        )

        self.fc_out = nn.Linear(embd_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)

        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)

        return out


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, embd_size=256, num_layers=6, forward_expansion=4, heads=8, dropout=0, device="cuda", max_len=100):
        super(Transformer, self).__init__()

        self.encoder = Encoder(src_vocab_size,embd_size, num_layers, heads, device, forward_expansion, dropout, max_len)
        self.decoder = Decoder(trg_vocab_size, embd_size, num_layers, heads, forward_expansion, dropout, device, max_len)

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        return src_mask.to(self.device)
    
    def make_trg_mask(self, trg):
        N, trg_len = trg.shape

        trg_mask = torch.tril((torch.ones(trg_len, trg_len))).expand(N, 1, trg_len, trg_len)

        return trg_mask.to(self.device)
    
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, src_mask, trg_mask)
        return output
    

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 2, 8, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)

    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(device)

    out = model(x, trg[:, :-1])
    print(out.shape)