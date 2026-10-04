#!/usr/bin/env python3
"""
Train your AI from scratch.

Usage:
  python train.py                  # normal training (max_iters steps)
  python train.py --continuous     # keep training forever until Ctrl+C
  python train.py --resume         # resume from last checkpoint
"""

import argparse
import os
import time
from pathlib import Path

import torch
from tqdm import tqdm

from config import *
from model import GPTLanguageModel
from utils import CharTokenizer, get_batch, estimate_loss, get_device, load_text

def main():
    parser = argparse.ArgumentParser(description="Train AI from scratch")
    parser.add_argument("--continuous", action="store_true", help="Keep training forever until stopped")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--data", type=str, default="data/input.txt", help="Path to training text")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Load data
    text = load_text(args.data)
    tokenizer = CharTokenizer(text)
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Text length: {len(text):,} characters")

    # Encode entire dataset
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    # Train / val split
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # Create model
    model = GPTLanguageModel(
        vocab_size=tokenizer.vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size,
        dropout=dropout,
    )
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Checkpoint directory
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    latest_ckpt = ckpt_dir / "latest.pt"

    start_iter = 0
    if args.resume and latest_ckpt.exists():
        print("Resuming from checkpoint...")
        checkpoint = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_iter = checkpoint.get("iter", 0)
        print(f"Resumed from iteration {start_iter}")

    def save_checkpoint(iter_num):
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iter": iter_num,
            "config": {
                "n_embd": n_embd,
                "n_head": n_head,
                "n_layer": n_layer,
                "block_size": block_size,
                "vocab_size": tokenizer.vocab_size,
            },
            "tokenizer_stoi": tokenizer.stoi,
            "tokenizer_itos": tokenizer.itos,
        }, latest_ckpt)
        # also keep a numbered one sometimes
        if iter_num % 2000 == 0:
            torch.save({
                "model": model.state_dict(),
                "iter": iter_num,
            }, ckpt_dir / f"ckpt_{iter_num}.pt")

    print("\nStarting training...")
    if args.continuous:
        print(">>> CONTINUOUS TRAINING MODE ON <<<")
        print("Press Ctrl+C to stop. Model will be saved automatically.\n")

    iter_num = start_iter
    try:
        while True:
            # one training step
            xb, yb = get_batch(train_data, block_size, batch_size, device)
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if iter_num % eval_interval == 0 or iter_num == start_iter:
                train_loss = estimate_loss(model, train_data, block_size, batch_size, eval_iters, device)
                val_loss = estimate_loss(model, val_data, block_size, batch_size, eval_iters, device)
                print(f"step {iter_num:6d} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")

            if args.continuous:
                if iter_num % continuous_save_every == 0 and iter_num > start_iter:
                    save_checkpoint(iter_num)
                    print(f"  → checkpoint saved at step {iter_num}")
            else:
                if iter_num >= max_iters:
                    break

            iter_num += 1

    except KeyboardInterrupt:
        print("\n\nTraining stopped by user.")

    # Final save
    save_checkpoint(iter_num)
    print(f"Model saved to {latest_ckpt}")
    print("You can now run:  python generate.py")

if __name__ == "__main__":
    main()
