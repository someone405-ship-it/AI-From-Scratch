# AI From Scratch

**Your own neural network** — trained by you, running on your device, with a beautiful mobile-first interface.

No external LLM APIs for the core brain. Pure PyTorch Transformer built from the ground up.

---

## Features

| Feature | Description |
|---------|-------------|
| **Mobile-first UI** | Excellent interface designed for phones |
| **Chat** | Talk to the model you trained |
| **Learn from device** | Upload files directly from your phone/computer |
| **Manual teaching** | Teach specific facts that the AI remembers |
| **Continuous training** | Keeps learning until you stop it |
| **Stronger model** | 384-dim, 8 layers, 8 heads, 512 context |
| **X Bot section** | Configure automatic posting 2–3 times per day |
| **GitHub section** | Works with your connected GitHub |
| **Top-k sampling** | Higher quality generation |

---

## Quick Start (Mobile or Desktop)

```bash
git clone https://github.com/someone405-ship-it/AI-From-Scratch.git
cd AI-From-Scratch
pip install -r requirements.txt
python app.py
```

Then open **http://YOUR-IP:7860** on your phone (or localhost:7860 on computer).

You can also install it as a Progressive Web App on mobile for an app-like experience.

---

## How to use

1. Go to **Learn** tab → upload text files from your device
2. Go to **Train** tab → press **Continuous Mode**
3. Let it train (the longer the better)
4. Go to **Chat** and talk to your AI
5. Optionally configure the **X Bot** and **GitHub** sections

---

## X Bot

In the X Bot tab you can:
- Set topics
- Choose how many posts per day (1–5)
- Enter your X API keys
- Generate sample posts with your trained model

Real automatic posting requires valid X API credentials (developer.x.com).

---

## Make it even stronger

- Feed it more data (books, code, personal notes, chat history)
- Run Continuous training for many hours
- Teach important facts in the Learn tab
- If you have a good GPU, increase numbers in `config.py`

---

## Project Structure

```
AI-From-Scratch/
├── app.py              # Full mobile-first web app
├── model.py            # Transformer (from scratch)
├── train.py            # CLI training
├── generate.py         # CLI chat
├── config.py           # Model size & settings
├── utils.py
├── data/
│   ├── input.txt
│   ├── manual_knowledge.json
│   └── x_bot_config.json
├── checkpoints/
└── requirements.txt
```

---

## Honest limits

This is a real neural network you own and train.  
It will not match frontier models in general knowledge or fluency across every language until you give it massive data and compute.  
What it *will* do is learn the style and information you provide and improve the more you train it.

---

**Own your AI.**
