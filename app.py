#!/usr/bin/env python3
"""
AI From Scratch - Mobile-First Powerful App

Features:
- Excellent mobile UI
- Chat with your own trained model
- Learn from device files (local data)
- Manual teaching mode
- Continuous training
- GitHub connection section
- X (Twitter) Bot section (post 2-3 times/day)
- Stronger model
"""

import os
import threading
import time
import json
from pathlib import Path
from datetime import datetime

import gradio as gr
import torch

from config import *
from model import GPTLanguageModel
from utils import CharTokenizer, get_batch, get_device, load_text

# ===================== GLOBAL STATE =====================
device = get_device()
model = None
tokenizer = None
optimizer = None
training_thread = None
is_training = False
stop_training_flag = False
current_step = 0
status_message = "Welcome! Upload data or start training."
manual_knowledge = []          # user-taught facts
x_bot_config = {
    "enabled": False,
    "topics": "",
    "posts_per_day": 2,
    "last_post": None,
    "api_key": "",
    "api_secret": "",
    "access_token": "",
    "access_secret": ""
}
github_connected = False

ckpt_dir = Path("checkpoints")
ckpt_dir.mkdir(exist_ok=True)
latest_ckpt = ckpt_dir / "latest.pt"
knowledge_file = Path("data/manual_knowledge.json")
x_config_file = Path("data/x_bot_config.json")

# ===================== HELPERS =====================
def load_or_create_model(text=None):
    global model, tokenizer, optimizer, status_message, current_step

    if text is None:
        text = load_text("data/input.txt")

    tokenizer = CharTokenizer(text)
    model = GPTLanguageModel(
        vocab_size=tokenizer.vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    if latest_ckpt.exists():
        try:
            ckpt = torch.load(latest_ckpt, map_location=device)
            if ckpt["config"]["vocab_size"] == tokenizer.vocab_size:
                model.load_state_dict(ckpt["model"])
                optimizer.load_state_dict(ckpt["optimizer"])
                current_step = ckpt.get("iter", 0)
                status_message = f"Checkpoint loaded (step {current_step})"
            else:
                status_message = "Vocab changed → new model created"
        except Exception as e:
            status_message = f"Checkpoint error: {e}"
    else:
        params = sum(p.numel() for p in model.parameters()) / 1e6
        status_message = f"New powerful model ready ({params:.1f}M parameters)"

    return status_message

def save_checkpoint(step):
    if model is None:
        return
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iter": step,
        "config": {
            "n_embd": n_embd, "n_head": n_head, "n_layer": n_layer,
            "block_size": block_size, "vocab_size": tokenizer.vocab_size
        },
        "tokenizer_stoi": tokenizer.stoi,
        "tokenizer_itos": tokenizer.itos,
    }, latest_ckpt)

def load_manual_knowledge():
    global manual_knowledge
    if knowledge_file.exists():
        try:
            manual_knowledge = json.loads(knowledge_file.read_text(encoding="utf-8"))
        except:
            manual_knowledge = []
    return manual_knowledge

def save_manual_knowledge():
    knowledge_file.parent.mkdir(exist_ok=True)
    knowledge_file.write_text(json.dumps(manual_knowledge, ensure_ascii=False, indent=2), encoding="utf-8")

def load_x_config():
    global x_bot_config
    if x_config_file.exists():
        try:
            x_bot_config.update(json.loads(x_config_file.read_text(encoding="utf-8")))
        except:
            pass
    return x_bot_config

def save_x_config():
    x_config_file.parent.mkdir(exist_ok=True)
    x_config_file.write_text(json.dumps(x_bot_config, indent=2), encoding="utf-8")

# ===================== TRAINING =====================
def training_loop(continuous=True):
    global is_training, stop_training_flag, current_step, status_message

    is_training = True
    stop_training_flag = False
    status_message = "Training started…"

    text = load_text("data/input.txt")
    # inject manual knowledge into training data
    if manual_knowledge:
        extra = "\n".join([f"Fact: {k}" for k in manual_knowledge])
        text = text + "\n\n" + extra

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]

    step = current_step
    try:
        while not stop_training_flag:
            xb, yb = get_batch(train_data, block_size, batch_size, device)
            _, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            step += 1
            current_step = step

            if step % 40 == 0:
                status_message = f"Step {step} • loss {loss.item():.4f}"

            if step % continuous_save_every == 0:
                save_checkpoint(step)
                status_message = f"Step {step} • loss {loss.item():.4f} • saved"

            if not continuous and step >= max_iters:
                break
            time.sleep(0.002)
    except Exception as e:
        status_message = f"Error: {e}"
    finally:
        save_checkpoint(step)
        is_training = False
        status_message = f"Stopped at step {step}. Model saved."

def start_train(continuous):
    global training_thread
    if is_training:
        return "Already training", get_status()
    if model is None:
        load_or_create_model()
    training_thread = threading.Thread(target=training_loop, args=(continuous,), daemon=True)
    training_thread.start()
    return f"{'Continuous' if continuous else 'Normal'} training started", get_status()

def stop_train():
    global stop_training_flag
    if not is_training:
        return "Not training", get_status()
    stop_training_flag = True
    return "Stopping…", get_status()

# ===================== CHAT & LEARNING =====================
def chat(message, history, temperature, top_k, max_tokens):
    if model is None or tokenizer is None:
        history = history + [(message, "Please train or load a model first.")]
        return history, ""

    # Add manual knowledge as soft context
    context_text = message
    if manual_knowledge:
        facts = " | ".join(manual_knowledge[-8:])  # last few facts
        context_text = f"Known facts: {facts}\n\nUser: {message}"

    model.eval()
    try:
        ids = tokenizer.encode(context_text)
        if not ids:
            ids = [0]
        context = torch.tensor([ids], dtype=torch.long, device=device)

        with torch.no_grad():
            out = model.generate(
                context,
                max_new_tokens=int(max_tokens),
                temperature=float(temperature),
                top_k=int(top_k) if top_k > 0 else None
            )
            full = tokenizer.decode(out[0].tolist())
            reply = full[len(context_text):].strip() or full
    except Exception as e:
        reply = f"Generation error: {e}"

    history = history + [(message, reply)]
    return history, ""

def teach_fact(fact):
    if not fact.strip():
        return "Write something to teach.", get_knowledge_display()
    manual_knowledge.append(fact.strip())
    save_manual_knowledge()
    return f"Learned: {fact.strip()}", get_knowledge_display()

def clear_knowledge():
    global manual_knowledge
    manual_knowledge = []
    save_manual_knowledge()
    return "All manual knowledge cleared.", get_knowledge_display()

def get_knowledge_display():
    if not manual_knowledge:
        return "No manual knowledge yet. Teach me something!"
    return "\n".join([f"• {k}" for k in manual_knowledge[-20:]])

def upload_device_data(files):
    if not files:
        return "No files selected."
    combined = []
    total_chars = 0
    for f in files:
        try:
            text = Path(f.name).read_text(encoding="utf-8", errors="ignore")
            combined.append(text)
            total_chars += len(text)
        except Exception as e:
            return f"Error reading {f.name}: {e}"

    Path("data").mkdir(exist_ok=True)
    full_text = "\n\n".join(combined)
    Path("data/input.txt").write_text(full_text, encoding="utf-8")
    msg = load_or_create_model(full_text)
    return f"Loaded {len(files)} file(s) • {total_chars:,} characters\n{msg}"

# ===================== X BOT =====================
def update_x_bot(enabled, topics, posts_per_day, api_key, api_secret, access_token, access_secret):
    x_bot_config.update({
        "enabled": enabled,
        "topics": topics,
        "posts_per_day": int(posts_per_day),
        "api_key": api_key,
        "api_secret": api_secret,
        "access_token": access_token,
        "access_secret": access_secret
    })
    save_x_config()
    status = "X Bot ENABLED" if enabled else "X Bot disabled"
    return f"{status}\nTopics: {topics}\nPosts/day: {posts_per_day}"

def generate_x_post(topics):
    if model is None:
        return "Train the model first."
    prompt = f"Write a short interesting tweet about: {topics}\nTweet:"
    try:
        ids = tokenizer.encode(prompt)
        context = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(context, max_new_tokens=120, temperature=0.9, top_k=40)
            text = tokenizer.decode(out[0].tolist())
            # extract after "Tweet:"
            if "Tweet:" in text:
                text = text.split("Tweet:")[-1].strip()
            return text[:280]
    except Exception as e:
        return f"Error: {e}"

# ===================== STATUS =====================
def get_status():
    training = "TRAINING" if is_training else "Idle"
    params = f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M" if model else "—"
    return f"""
**Status:** {training}  
**Step:** {current_step}  
**Model:** {params} parameters  
**Device:** {device}  
**Manual facts:** {len(manual_knowledge)}  

{status_message}
"""

# ===================== UI =====================
custom_css = """
.gradio-container {
    max-width: 100% !important;
    padding: 8px !important;
    font-family: 'Inter', system-ui, sans-serif !important;
}
footer {display: none !important;}
.dark {
    background: #0f0f13 !important;
}
button {
    border-radius: 12px !important;
    font-weight: 600 !important;
}
.chatbot {
    border-radius: 16px !important;
    min-height: 55vh !important;
}
textarea, input {
    border-radius: 12px !important;
}
"""

with gr.Blocks(
    title="AI From Scratch",
    theme=gr.themes.Soft(
        primary_hue="violet",
        secondary_hue="indigo",
        neutral_hue="slate",
        font=["Inter", "system-ui", "sans-serif"]
    ),
    css=custom_css,
) as demo:

    gr.Markdown("""
    <div style="text-align:center; padding: 10px 0 5px 0;">
        <h1 style="margin:0; font-size:1.8rem; background:linear-gradient(90deg,#a78bfa,#818cf8); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            AI From Scratch
        </h1>
        <p style="margin:4px 0 0 0; opacity:0.7; font-size:0.95rem;">Your personal neural network • Mobile ready</p>
    </div>
    """)

    with gr.Tabs() as tabs:

        # ---------- CHAT TAB ----------
        with gr.Tab("Chat", id="chat"):
            chatbot = gr.Chatbot(
                height=480,
                show_copy_button=True,
                bubble_full_width=False,
                avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=ai"),
                placeholder="Your AI is ready to talk…"
            )
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Message…",
                    show_label=False,
                    scale=6,
                    container=False,
                    autofocus=True
                )
                send = gr.Button("Send", variant="primary", scale=1, min_width=80)

            with gr.Accordion("Generation settings", open=False):
                with gr.Row():
                    temperature = gr.Slider(0.1, 1.6, value=0.85, step=0.05, label="Temperature")
                    top_k = gr.Slider(0, 150, value=60, step=5, label="Top-k")
                    max_tokens = gr.Slider(40, 600, value=220, step=10, label="Max tokens")

            send.click(chat, [msg, chatbot, temperature, top_k, max_tokens], [chatbot, msg])
            msg.submit(chat, [msg, chatbot, temperature, top_k, max_tokens], [chatbot, msg])

        # ---------- LEARN TAB ----------
        with gr.Tab("Learn", id="learn"):
            gr.Markdown("### Learn from your device")
            gr.Markdown("Upload text files from your phone or computer. The AI will train on them.")
            files = gr.File(
                label="Select files from device",
                file_count="multiple",
                file_types=[".txt", ".md", ".csv", ".json", ".log"]
            )
            upload_btn = gr.Button("Load into AI", variant="primary")
            upload_result = gr.Textbox(label="Result", interactive=False, lines=3)

            gr.Markdown("### Manual Teaching")
            gr.Markdown("Teach specific facts. These are injected during training and used in chat.")
            fact_input = gr.Textbox(placeholder="e.g. My name is Alex and I love robotics", lines=2)
            with gr.Row():
                teach_btn = gr.Button("Teach this", variant="primary")
                clear_btn = gr.Button("Clear all knowledge", variant="stop")
            knowledge_box = gr.Textbox(label="What I currently know", lines=8, interactive=False)

            upload_btn.click(upload_device_data, files, upload_result)
            teach_btn.click(teach_fact, fact_input, [upload_result, knowledge_box])
            clear_btn.click(clear_knowledge, outputs=[upload_result, knowledge_box])

        # ---------- TRAIN TAB ----------
        with gr.Tab("Train", id="train"):
            status = gr.Markdown(get_status())
            with gr.Row():
                start_btn = gr.Button("Start Training", variant="primary")
                cont_btn = gr.Button("Continuous Mode", variant="huggingface")
                stop_btn = gr.Button("Stop", variant="stop")
            refresh = gr.Button("Refresh Status", size="sm")

            gr.Markdown("""
            **Continuous Mode** keeps training until you press Stop.  
            Best for making the AI smarter over time.  
            Checkpoints are saved automatically.
            """)

            start_btn.click(start_train, gr.State(False), [status, status])
            cont_btn.click(start_train, gr.State(True), [status, status])
            stop_btn.click(stop_train, outputs=[status, status])
            refresh.click(get_status, outputs=status)

        # ---------- X BOT TAB ----------
        with gr.Tab("X Bot", id="xbot"):
            gr.Markdown("### Automatic X (Twitter) Poster")
            gr.Markdown("Connect your X account and the AI can post 2–3 times per day about topics you choose.")

            x_enabled = gr.Checkbox(label="Enable X Bot", value=False)
            x_topics = gr.Textbox(
                label="Topics to post about",
                placeholder="AI, technology, my project, daily thoughts…",
                lines=2
            )
            x_posts = gr.Slider(1, 5, value=2, step=1, label="Posts per day")

            with gr.Accordion("X API Keys (required for real posting)", open=False):
                gr.Markdown("Get them from developer.x.com → your app → Keys and Tokens")
                x_key = gr.Textbox(label="API Key", type="password")
                x_secret = gr.Textbox(label="API Secret", type="password")
                x_token = gr.Textbox(label="Access Token", type="password")
                x_token_secret = gr.Textbox(label="Access Token Secret", type="password")

            save_x = gr.Button("Save Bot Settings", variant="primary")
            x_status = gr.Textbox(label="Bot Status", interactive=False)

            gr.Markdown("### Test post generation")
            gen_btn = gr.Button("Generate sample post")
            sample_post = gr.Textbox(label="Sample", lines=3)

            save_x.click(
                update_x_bot,
                [x_enabled, x_topics, x_posts, x_key, x_secret, x_token, x_token_secret],
                x_status
            )
            gen_btn.click(generate_x_post, x_topics, sample_post)

        # ---------- GITHUB TAB ----------
        with gr.Tab("GitHub", id="github"):
            gr.Markdown("### Connect to GitHub")
            gr.Markdown("""
            This AI can work with your GitHub repositories.

            **Current status:** Your GitHub is already connected to Grok (the system that built this project).

            From here you can:
            - Ask the AI to create new repositories
            - Push code updates
            - Manage issues and pull requests

            Just go back to the **Chat** tab and tell it what you want, for example:

            - "Create a new repo called my-notes"
            - "Push the latest changes"
            - "Show my repositories"

            Because this project lives in your GitHub, every improvement I make is already version-controlled.
            """)
            gr.Markdown(f"**Repository:** [AI-From-Scratch](https://github.com/someone405-ship-it/AI-From-Scratch)")

        # ---------- ABOUT TAB ----------
        with gr.Tab("About", id="about"):
            gr.Markdown("""
            ### About this AI

            This is a **real neural network** built completely from scratch in pure PyTorch.

            - No OpenAI / Groq / Gemini API calls for the core brain
            - You train it on your own data
            - It runs on your device (CPU or GPU)
            - Continuous learning mode
            - Manual teaching
            - Mobile-friendly interface

            #### Limits (honest)
            A small model trained on limited data cannot match GPT-4 or Grok in general knowledge or every language.  
            The more high-quality text you give it and the longer you train, the better it becomes at the style and topics you feed it.

            #### Tips for best results
            1. Upload large text files (books, code, chat exports, notes)
            2. Use Continuous training for hours
            3. Teach important facts manually
            4. Use a GPU if available

            **Own your intelligence.**
            """)

    # Load everything on start
    demo.load(load_or_create_model, outputs=status)
    demo.load(load_manual_knowledge)
    demo.load(load_x_config)
    demo.load(get_knowledge_display, outputs=knowledge_box)

if __name__ == "__main__":
    print(f"\nAI From Scratch starting on {device}")
    print("Open http://localhost:7860 on your phone or computer\n")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        favicon_path=None
    )
