import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import default_data_collator, AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset


class BlockManager:
    def __init__(self, num_blocks, block_size, num_layers, num_heads, head_dim, device):
        self.block_size = block_size   # how many tokens each page can hold
        self.num_blocks = num_blocks   # total pages in the pool
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device

        # pool of free page indices — all pages are free at the start
        # e.g. if num_blocks=8 → free_blocks = [0, 1, 2, 3, 4, 5, 6, 7]
        self.free_blocks = list(range(num_blocks))


        # page table — maps each user (sequence id) to their list of page indices
        # e.g. page_table = {0: [0, 3], 1: [1, 4, 7], 2: [2]}
        self.page_table = {}

        # physical memory — the actual storage for K and V
        # shape: (num_layers, num_blocks, 2, num_heads, block_size, head_dim)
        #                                 ↑
        #                          2 = one for K, one for V

        self.kv_storage = torch.zeros(
            num_layers, num_blocks, 2, num_heads, block_size, head_dim,
            device=device
        )

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

    def forward(self, x, mask, page_kv_cache=None):
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
        if page_kv_cache is not None:
            block_manager = page_kv_cache["block_manager"]
            seq_id = page_kv_cache["seq_id"]
            layer_id      = page_kv_cache["layer_id"]
            current_len   = page_kv_cache["current_len"]

            # write current token's K,V into the correct page and slot
            # keys shape after permute:  (N, heads, heads_dim, seq_len)
            # values shape after permute: (N, heads, seq_len, heads_dim)
            # we squeeze seq_len=1 dimension for writing single token
            k_to_write = keys_transpose[0, :, :, 0]    # (heads, heads_dim)
            v_to_write = value_transpose[0, :, 0, :]   # (heads, heads_dim)

            # write into physical storage via block manager
            block_manager.write_kv(seq_id, layer_id, current_len, k_to_write, v_to_write)


        energy = torch.matmul(query_transpose, keys_transpose)

        if mask is not None:
            energy = energy.masked_fill(mask==0, float("-1e20"))

        attention = torch.softmax(energy / self.embd_size**(1/2), dim=-1)

        output = torch.matmul(attention, value_transpose)

        output = output.reshape(N, seq_len, self.heads*self.heads_dim)

        output = self.fc_out(output) # we use this linear layer to allow different heads to interact with each other

        return output, page_kv_cache
    
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

    def forward(self, x, mask=None, page_kv_cache=None):

        # update layer_id in page_kv_cache so SelfAttention
        # writes/reads from the correct layer's storage in BlockManager
        if page_kv_cache is not None:
            page_kv_cache["layer_id"] = page_kv_cache.get("layer_id", 0)

        # pass page_kv_cache down to SelfAttention, get updated cache back
        attention_output, page_kv_cache = self.attention(self.norm_1(x), mask, page_kv_cache)

        x = x + self.dropout(attention_output)  

        # pre-normalisation 2
        ff_out = self.ffn(self.norm_2(x))
        x = x + self.dropout(ff_out)

        return x, page_kv_cache


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

    def forward(self, x, mask, labels=None, page_kv_cache=None):
        N, seq_len = x.shape

        # positional embedding — two cases:
        # 1. no cache or empty cache → full sequence, positions start from 0
        # 2. cache has past tokens   → only new token, position continues from current_len
        if page_kv_cache is not None and page_kv_cache["current_len"] > 0:
            # current_len tells us how many tokens are already cached
            # new token's position must continue from where cache left off
            past_len = page_kv_cache["current_len"]
            positions = torch.arange(past_len, past_len + seq_len).expand(N, seq_len).to(self.device)
        else:
            # no cache — training or first inference step
            positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)

        x = self.dropout(self.word_embedding(x) + self.positional_embedding(positions))

        for i, layer in enumerate(self.layers):
            # update layer_id before each layer so BlockManager
            # writes/reads from correct layer's storage
            if page_kv_cache is not None:
                page_kv_cache["layer_id"] = i

            x, page_kv_cache = layer(x, mask, page_kv_cache)

        # update current_len after all layers are done
        # so next generation step knows how many tokens are cached
        if page_kv_cache is not None:
            page_kv_cache["current_len"] += seq_len

        x = self.ln_f(x)
        out = self.fc_out(x)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
            return loss, out, page_kv_cache

        return out, page_kv_cache
    
def train_decoder_only_transformer(model, dataset, tokenizer, device, batch_size=32, epochs=1, lr=5e-5, max_len=128):
    model.to(device)
    model.train()

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Loss function
    criterion = CrossEntropyLoss()

    # Dataloader (dataset should already be tokenized and grouped into blocks of max_len)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, collate_fn=default_data_collator)

    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

        for batch in progress_bar:
            input_ids = torch.tensor(batch['input_ids']).to(device=device)
            labels = input_ids.clone()

            # create causal mask
            mask = model.create_causal_mask(N=input_ids.shape[0], seq_len=input_ids.shape[1])

            # Forward pass
            loss, logits, _ = model(input_ids, mask=mask, labels=labels)

            # Backward Pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), "model.pth")
        print("Model saved to model.pth")

def generate_with_paged_attention(model, prompt, tokenizer, device, max_new_tokens=50,
                                   num_blocks=128, block_size=16):
    model.eval()
    model.to(device)

    # tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()

    with torch.no_grad():
        generated = input_ids
        seq_id = 0   # user id

        # initialize BlockManager
        block_manager = BlockManager(
            num_blocks=num_blocks,
            block_size=block_size,
            num_layers=len(model.layers),
            num_heads=model.layers[0].attention.heads,
            head_dim=model.layers[0].attention.heads_dim,
            device=device
        )

        # allocate first page for this user
        block_manager.allocate(seq_id)

        # initialize page_kv_cache
        page_kv_cache = {
            "block_manager": block_manager,
            "seq_id":        seq_id,
            "layer_id":      0,
            "current_len":   0
        }

        for i in range(max_new_tokens):
            if page_kv_cache["current_len"] == 0:
                # first step — pass full prompt
                N, seq_len = generated.shape
                mask = model.create_causal_mask(N=N, seq_len=seq_len)
                out, page_kv_cache = model(generated, mask, page_kv_cache=page_kv_cache)
            else:
                # every step after — pass only last token
                last_token = generated[:, -1:]
                N, seq_len = last_token.shape
                mask = model.create_causal_mask(N=N, seq_len=1)

                # check if we need a new page for next token
                block_manager.can_append(seq_id, page_kv_cache["current_len"])

                out, page_kv_cache = model(last_token, mask, page_kv_cache=page_kv_cache)

            # pick most likely next token
            next_token = out[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated = torch.cat([generated, next_token], dim=1)

        # free pages when done
        block_manager.free(seq_id)

    end_time.record()
    torch.cuda.synchronize()

    elapsed = start_time.elapsed_time(end_time)

    decoded = tokenizer.decode(generated[0], skip_special_tokens=True)

    print(f"\n--- With Paged Attention ---")
    print(f"Generated text  : {decoded}")
    print(f"Time taken      : {elapsed:.2f} ms")
    print(f"Tokens generated: {max_new_tokens}")
    print(f"Pages used      : {len(block_manager.page_table.get(seq_id, []))}")

    return decoded, elapsed

def tokenize(example):
    return tokenizer(example["text"])

def group_texts(examples, block_size=128):
    result = {}

    for key in examples.keys():
        concatenated = sum(examples[key], [])
        total_length = (len(concatenated) // block_size) * block_size
        result[key] = [concatenated[i:i+block_size] for i in range(0, total_length, block_size)]

    return result

if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    grouped = tokenized.map(group_texts, batched=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load trained model
    model = DecoderOnlyTransformer(
        vocab_size=tokenizer.vocab_size,
        embd_size=256,
        dropout_p=0.1,
        forward_expansion=4,
        max_len=128,
        heads=8,
        num_layers=5,
        device=device
    )

    train_decoder_only_transformer(model=model, dataset=grouped["train"], tokenizer=tokenizer, device=device)

    model.load_state_dict(torch.load("model.pth"))

    prompt = "The cat sat on the"

    decoded_paged, time_paged = generate_with_paged_attention(model=model, prompt=prompt, tokenizer=tokenizer, device=device, max_new_tokens=50)
    print(f"Paged attention time: {time_paged:.2f} ms")