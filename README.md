# AI From Scratch

Your own neural network — trained by you, running on your device, with live thinking, web search, and a beautiful mobile interface.

---

## Highlights

- **Live streaming thinking steps** — watch the AI plan, search, recall facts, and generate in real time
- **Four thinking modes**: Fast → Balanced → Strong → Research
- **Web search** (no API key required)
- **Conversation memory**
- **Learn from device files** + manual teaching
- **Continuous training** with auto checkpoints
- **X Bot** configuration section
- **GitHub** awareness
- **Export chat**
- **Mobile-first** excellent UI

---

## Quick Start

```bash
git clone https://github.com/someone405-ship-it/AI-From-Scratch.git
cd AI-From-Scratch
pip install -r requirements.txt
python app.py
```

Open **http://localhost:7860** (or your computer’s IP on your phone).

---

## Thinking Modes

| Mode | What it does |
|------|--------------|
| **Fast** | Quick local answer |
| **Balanced** | Memory + light reasoning |
| **Strong** | Live steps + knowledge + web search |
| **Research** | Deepest reasoning, multiple rounds, best search |

In Strong and Research modes you will see every step appear live:
- Planning…
- Recalling facts…
- Connecting to GitHub…
- Searching the web…
- Deep thinking round 1/3…
- Generating answer…

---

## Make the AI smarter

1. Go to **Learn** → upload large text files (books, notes, code, chat exports)
2. Go to **Train** → press **Continuous Mode** and let it run
3. Teach important facts manually
4. Use **Research** mode when you need current information

---

## Project structure

```
AI-From-Scratch/
├── app.py           # Full web app with streaming & all features
├── model.py         # Transformer built from scratch
├── train.py         # CLI training
├── generate.py      # CLI chat
├── config.py        # Model size & hyperparameters
├── utils.py
├── data/
├── checkpoints/
└── requirements.txt
```

---

## Requirements

- Python 3.9+
- PyTorch
- Gradio
- requests

Works on CPU and NVIDIA GPU.

---

**Own your intelligence.**
