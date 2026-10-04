import torch
import os
from pathlib import Path

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_text(path="data/input.txt"):
    path = Path(path)
    if not path.exists():
        # Create a small sample if no data is provided
        path.parent.mkdir(parents=True, exist_ok=True)
        sample = """Hello world. This is a sample training text for your AI from scratch.
The model will learn to predict the next character based on the previous ones.
You should replace this file with a much larger text file for better results.
Books, code, chat logs, Wikipedia dumps, or any plain text works well.
The more high-quality text you give it, the better your AI will become.
"""
        path.write_text(sample, encoding="utf-8")
        print(f"Created sample data at {path}. Please replace it with real training data!")
    return path.read_text(encoding="utf-8")

class CharTokenizer:
    """Simple character-level tokenizer."""
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, tokens):
        return "".join([self.itos[i] for i in tokens])

def get_batch(data, block_size, batch_size, device):
    """Generate a small batch of data."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, eval_iters, device):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(data, block_size, batch_size, device)
        _, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean()
