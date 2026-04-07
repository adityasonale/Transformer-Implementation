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
            loss, logits = model(input_ids, mask=mask, labels=labels)

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