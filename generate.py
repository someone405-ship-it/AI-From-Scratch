#!/usr/bin/env python3
"""
Talk to your trained AI (or generate text from it).

Usage:
  python generate.py
  python generate.py --prompt "Once upon a time"
  python generate.py --max_tokens 500 --temperature 0.8
"""

import argparse
from pathlib import Path

import torch

from model import GPTLanguageModel
from utils import get_device

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="", help="Starting text")
    parser.add_argument("--max_tokens", type=int, default=300, help="How many characters to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Creativity (0.1 = boring, 1.5 = wild)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest.pt")
    args = parser.parse_args()

    device = get_device()
    ckpt_path = Path(args.checkpoint)

    if not ckpt_path.exists():
        print("No checkpoint found. Train the model first:")
        print("  python train.py")
        return

    print(f"Loading model from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)

    config = checkpoint["config"]
    model = GPTLanguageModel(
        vocab_size=config["vocab_size"],
        n_embd=config["n_embd"],
        n_head=config["n_head"],
        n_layer=config["n_layer"],
        block_size=config["block_size"],
        dropout=0.0,  # no dropout at inference
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.to(device)

    # Rebuild tokenizer from saved mappings
    stoi = checkpoint["tokenizer_stoi"]
    itos = checkpoint["tokenizer_itos"]

    def encode(s):
        return [stoi[c] for c in s if c in stoi]

    def decode(tokens):
        return "".join([itos[i] for i in tokens])

    print("\n=== Your AI is ready ===")
    print("Type a prompt and press Enter. Empty prompt = random generation.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if prompt.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        if not prompt and args.prompt:
            prompt = args.prompt

        # Encode prompt
        if prompt:
            context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
        else:
            # start with a random token or newline
            context = torch.zeros((1, 1), dtype=torch.long, device=device)

        print("AI: ", end="", flush=True)

        # Generate
        with torch.no_grad():
            generated = model.generate(context, max_new_tokens=args.max_tokens, temperature=args.temperature)
            full_text = decode(generated[0].tolist())

            # Only show the new part if we had a prompt
            if prompt:
                # find where the prompt ends (approximate)
                print(full_text[len(prompt):])
            else:
                print(full_text)

        print()

if __name__ == "__main__":
    main()
