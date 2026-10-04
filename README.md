# AI From Scratch

A **real neural network language model** built entirely from scratch using pure Python + PyTorch.

No external AI APIs. No Groq. No Gemini. No OpenAI.  
Just your own model that **you** train and own.

---

## Features

- Full GPT-style Transformer written from the ground up
- **Web Interface** (Gradio) with chat, training controls, and data upload
- **Continuous Training Mode** — keeps improving until you stop it
- More powerful defaults (256 embd, 6 layers, 8 heads)
- Top-k sampling for better generation quality
- Gradient clipping + GELU activations
- Automatic checkpointing
- Works on CPU or GPU

---

## Quick Start

```bash
git clone https://github.com/someone405-ship-it/AI-From-Scratch.git
cd AI-From-Scratch
pip install -r requirements.txt
```

### Option 1: Web Interface (Recommended)

```bash
python app.py
```

Then open **http://localhost:7860** in your browser.

From the web UI you can:
- Upload any `.txt` file as training data
- Start normal or continuous training
- Chat with your model live
- Adjust temperature, top-k, and length

### Option 2: Command Line

```bash
# Train
python train.py

# Continuous training (runs until Ctrl+C)
python train.py --continuous

# Chat in terminal
python generate.py
```

---

## How to make it smarter

1. **More data** — Put large text files in `data/input.txt` (books, code, Wikipedia, your notes...)
2. **Longer training** — Use Continuous mode and let it run for hours
3. **Bigger model** — Edit `config.py` and increase `n_embd`, `n_layer`, `n_head`
4. **GPU** — Training is much faster with an NVIDIA GPU

---

## Project Structure

```
AI-From-Scratch/
├── app.py            # Web interface (Gradio)
├── model.py          # Transformer neural network (from scratch)
├── train.py          # CLI training + continuous mode
├── generate.py       # CLI chat / generation
├── config.py         # Model size & training settings
├── utils.py          # Tokenizer & helpers
├── data/
│   └── input.txt     # Your training text goes here
├── checkpoints/     # Saved model weights
└── requirements.txt
```

---

## Model Architecture

- Character-level tokenizer (simple & transparent)
- Multi-head self-attention
- Feed-forward layers with GELU
- Residual connections + LayerNorm
- Causal masking (can only look at past tokens)

This is the same fundamental architecture used by GPT, Claude, Grok, etc. — just much smaller so you can train it yourself.

---

## License

MIT — completely free to use, modify, and share.

---

**Own your AI.**
