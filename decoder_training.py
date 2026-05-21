import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import default_data_collator 
from decoder_only import DecoderOnlyTransformer

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

def tokenize(example):
    return tokenizer(example["text"])

def group_texts(examples, block_size=128):
    result = {}

    for key in examples.keys():
        concatenated = sum(examples[key], [])
        total_length = (len(concatenated) // block_size) * block_size
        result[key] = [concatenated[i:i+block_size] for i in range(0, total_length, block_size)]

    return result

def generate_without_cache(model, prompt, tokenizer, device, max_new_tokens=50):
    model.eval()
    model.to(device)

    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()

    with torch.no_grad():
        generated = input_ids  # start with prompt tokens

        for _ in range(max_new_tokens):
            # pass FULL sequence every step — grows every iteration
            N, seq_len = generated.shape
            mask = model.create_causal_mask(N=N, seq_len=seq_len)

            # full sequence goes in every step
            out, _ = model(generated, mask)

            # take the last token's logits and pick the most likely next token
            next_token = out[:, -1, :].argmax(dim=-1).unsqueeze(1)

            # append new token to sequence
            generated = torch.cat([generated, next_token], dim=1)

        end_time.record()
        torch.cuda.synchronize()

        elapsed = start_time.elapsed_time(end_time)  # milliseconds

        decoded = tokenizer.decode(generated[0], skip_special_tokens=True)

        print(f"\n--- Without KV Cache ---")
        print(f"Generated text : {decoded}")
        print(f"Time taken     : {elapsed:.2f} ms")
        print(f"Tokens generated: {max_new_tokens}")

        return decoded, elapsed
    
def generate_with_cache(model, prompt, tokenizer, device, max_new_tokens=50):
    model.eval()
    model.to(device)

    # tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()

    with torch.no_grad():
        generated = input_ids
        kv_cache = None

        for i in range(max_new_tokens):
            if kv_cache is None:
                # first step — pass full prompt, build cache from scratch
                N, seq_len = generated.shape
                mask = model.create_causal_mask(N=N, seq_len=seq_len)
                out, kv_cache = model(generated, mask, kv_cache=None)
            else:
                # every step after — pass only the last generated token
                # seq_len is always 1 here
                last_token = generated[:, -1:]
                N, seq_len = last_token.shape
                mask = model.create_causal_mask(N=N, seq_len=1)
                out, kv_cache = model(last_token, mask, kv_cache=kv_cache)

            # take the last token's logits and pick most likely next token
            next_token = out[:, -1, :].argmax(dim=-1).unsqueeze(1)

            # append new token to sequence
            generated = torch.cat([generated, next_token], dim=1)

    end_time.record()
    torch.cuda.synchronize()

    elapsed = start_time.elapsed_time(end_time)  # milliseconds

    decoded = tokenizer.decode(generated[0], skip_special_tokens=True)

    print(f"\n--- With KV Cache ---")
    print(f"Generated text : {decoded}")
    print(f"Time taken     : {elapsed:.2f} ms")
    print(f"Tokens generated: {max_new_tokens}")

    return decoded, elapsed

if __name__ == "__main__":
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    grouped = tokenized.map(group_texts, batched=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DecoderOnlyTransformer(vocab_size=tokenizer.vocab_size, embd_size=256, dropout_p=0.1, forward_expansion=4, max_len=128, heads=8, num_layers=5, device=device)

    # Train
    train_decoder_only_transformer(model=model, dataset=grouped["train"], tokenizer=tokenizer, device=device)

    # Evaluation
    prompt = "The cat sat on the"

    decoded_no_cache, time_no_cache = generate_without_cache(model=model, prompt=prompt, tokenizer=tokenizer, device=device, max_new_tokens=50)
    decoded_cache, time_cache = generate_with_cache(model=model, prompt=prompt, tokenizer=tokenizer, device=device, max_new_tokens=50)

    # Compare
    print(f"\n--- Comparison ---")
    print(f"Speedup         : {time_no_cache / time_cache:.2f}x faster with cache")
    print(f"Same output     : {decoded_no_cache == decoded_cache}")