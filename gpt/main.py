from data import GptDataset, GptDataLoader
from sklearn.model_selection import train_test_split
from config import BaseConfig
from transformers import AutoTokenizer
import torch
import torch.nn as nn
from models import GPT
from trainer import Trainer
from visualize import Visualizer

def prepare_gpt_training_data(data, tokenizer, max_length=1024):
    sequences = []
    current_sequence = []

    # Group lines until end tag
    for line in data:
        current_sequence.append(line)
        if '<|endoftext|>' in line:
            
            # Join into one text
            full_text = ''.join(current_sequence)
            tokens = tokenizer.encode(full_text, truncation=True, max_length=max_length)

            if len(tokens) < max_length:
                tokens = tokens + [tokenizer.pad_token_id]*(max_length - len(tokens))

            sequences.append(tokens)
            current_sequence = []

    # Handle last sequence if no end tag
    if current_sequence:
        full_text = ''.join(current_sequence)
        tokens = tokenizer.encode(full_text, truncation=True, max_length=max_length)
        if len(tokens) < max_length:
            tokens = tokens + [tokenizer.pad_token_id] * (max_length - len(tokens))

        sequences.append(tokens)

    return sequences

if __name__ == "__main__":
    # Load dataset
    dataset_path = f"D:\Datasets\TinyStories-train.txt"
    config = BaseConfig()
    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    with open(dataset_path, encoding="utf-8") as f:
        data = f.readlines()

    data = data[:10000]

    print("Data preparation started...")
    data = prepare_gpt_training_data(data[:100], tokenizer, max_length=100)
    print("Data preparation complete...")

    max_len = max(len(item) for item in data)
    print(f"Maximum token length: {max_len}")

    # token = tokenizer.decode([199999])
    # print(token)
    visualizer = Visualizer(save_dir=config.checkpoint_dir/'visualizations')
    vocab_size = tokenizer.vocab_size + 2

    model = GPT(config=config, vocab_size=vocab_size, embd_size=config.embd_size, heads=config.num_heads, dropout_p=0.5, max_len=max_len, num_layers=config.num_layers, forward_expansion=1, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params = model.parameters(), lr=config.learning_rate, betas=config.betas, weight_decay=config.weight_decay)

    # Split dataset
    train_data, val_data = train_test_split(
        data,
        test_size=config.val_split,
        random_state=42
    )

    # Create Dataset class instances
    train_dataset = GptDataset(data=train_data)
    val_dataset = GptDataset(data=val_data)

    # Create dataloader
    dataloader = GptDataLoader(config=config, train_dataset=train_dataset, val_dataset=val_dataset)
    train_dataloader = dataloader.train_dataloader()
    val_dataloader = dataloader.val_dataloader()

    # initialising training instance
    trainer = Trainer(model=model, config=config, criterion=criterion, optimizer=optimizer)

    # Start Training
    trainer.fit(train_loader=train_dataloader, val_loader=val_dataloader)

    # Visualize training history
    visualizer.plot_training_history(
        train_losses=trainer.train_losses,
        val_losses=trainer.val_losses,
        train_metrics=trainer.train_metrics_history if hasattr(trainer, 'train_metrics_history') else None,
        val_metrics=trainer.val_metrics_history if hasattr(trainer, 'val_metrics_history') else None,
        save_name='training_history.png'
    )