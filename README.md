# AI From Scratch

A **real neural network language model** built entirely from scratch using pure Python + PyTorch.

No external AI APIs. No Groq. No Gemini. No OpenAI.
Just your own model that you train yourself.

## What this is

- A tiny but fully working GPT-style Transformer
- Written from the ground up (attention, MLP, residual connections, etc.)
- Character-level language model (easy to understand and train)
- Full training loop
- **Continuous Training Mode**: keeps training forever until you stop it
- Text generation / simple chat
- Completely yours — you own the weights

## Requirements

- Python 3.9+
- PyTorch (CPU or GPU)
- A text file to train on (books, code, your notes, chat logs...)

## Quick Start

```bash
# 1. Clone
git clone https://github.com/someone405-ship-it/AI-From-Scratch.git
cd AI-From-Scratch

# 2. Install
pip install -r requirements.txt

# 3. Put your training data in data/input.txt
#    (any text file works — the bigger the better)

# 4. Train
python train.py

# 5. Generate text / chat with your model
python generate.py
```

## Continuous Training Mode

```bash
python train.py --continuous
```

This will keep training forever (or until you press Ctrl+C).
It automatically saves checkpoints so you can stop and resume anytime.

You can also set how long each "thinking" / training burst lasts.

## Project Structure

```
AI-From-Scratch/
├── model.py          # The Transformer neural network (from scratch)
├── train.py          # Training loop + continuous mode
├── generate.py       # Text generation / chat interface
├── config.py         # Hyperparameters
├── utils.py          # Helpers (tokenizer, data loading)
├── requirements.txt
├── data/
│   └── input.txt     # Put your training text here
└── checkpoints/      # Model weights are saved here
```

## How it works (simple explanation)

1. We split text into characters
2. The model learns to predict the next character
3. After enough training, it can generate new text that looks like the training data
4. The more data + more training time you give it, the smarter it gets

This is the same core idea behind GPT, Claude, Grok, etc. — just much smaller so you can run it on a normal computer.

## Scaling up

- More data → better model
- Longer training → better model
- Bigger model (increase `n_embd`, `n_layer`, `n_head` in config.py) → better model (needs more VRAM)
- GPU makes training much faster

## License

MIT — do whatever you want with it.

---

Built for you by Grok. Own your AI.
