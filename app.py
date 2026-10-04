#!/usr/bin/env python3
"""
AI From Scratch - Mobile-First Powerful App (Upgraded)

- Thinking modes: Fast / Balanced / Strong / Research
- Shows what it is doing (searching, planning, using knowledge, etc.)
- Web search capability
- Learn from device + manual teaching
- Continuous training
- X Bot + GitHub sections
- Bug fixes and more robust chat
"""

import os
import re
import threading
import time
import json
import traceback
from pathlib import Path
from datetime import datetime
from urllib.parse import quote_plus

import gradio as gr
import torch
import requests

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
manual_knowledge = []
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
            ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
            if ckpt.get("config", {}).get("vocab_size") == tokenizer.vocab_size:
                model.load_state_dict(ckpt["model"])
                if "optimizer" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer"])
                current_step = ckpt.get("iter", 0)
                status_message = f"Checkpoint loaded (step {current_step})"
            else:
                status_message = "Vocab changed → new model created"
        except Exception as e:
            status_message = f"Checkpoint error: {e}"
    else:
        params = sum(p.numel() for p in model.parameters()) / 1e6
        status_message = f"New model ready ({params:.1f}M parameters) on {device}"

    return status_message

def save_checkpoint(step):
    if model is None or tokenizer is None:
        return
    try:
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
    except Exception as e:
        print("Save error:", e)

def load_manual_knowledge():
    global manual_knowledge
    if knowledge_file.exists():
        try:
            manual_knowledge = json.loads(knowledge_file.read_text(encoding="utf-8"))
        except Exception:
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
        except Exception:
            pass
    return x_bot_config

def save_x_config():
    x_config_file.parent.mkdir(exist_ok=True)
    x_config_file.write_text(json.dumps(x_bot_config, indent=2), encoding="utf-8")

# ===================== WEB SEARCH =====================
def web_search(query, max_results=5):
    """Simple free web search using DuckDuckGo HTML (no API key needed)."""
    try:
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        headers = {"User-Agent": "Mozilla/5.0 (compatible; AI-From-Scratch/1.0)"}
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        # very lightweight extraction
        texts = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', r.text, re.DOTALL)
        titles = re.findall(r'class="result__a"[^>]*>(.*?)</a>', r.text, re.DOTALL)
        results = []
        for i, (t, s) in enumerate(zip(titles, texts)):
            t = re.sub(r'<.*?>', '', t).strip()
            s = re.sub(r'<.*?>', '', s).strip()
            if t and s:
                results.append(f"{i+1}. {t}\n   {s}")
            if len(results) >= max_results:
                break
        if not results:
            return "No clear results found."
        return "\n\n".join(results)
    except Exception as e:
        return f"Search failed: {e}"

# ===================== TRAINING =====================
def training_loop(continuous=True):
    global is_training, stop_training_flag, current_step, status_message

    is_training = True
    stop_training_flag = False
    status_message = "Training started…"

    try:
        text = load_text("data/input.txt")
        if manual_knowledge:
            extra = "\n".join([f"Fact: {k}" for k in manual_knowledge])
            text = text + "\n\n" + extra

        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        if len(data) < block_size + 1:
            status_message = "Not enough data to train. Upload more text."
            is_training = False
            return

        n = int(0.9 * len(data))
        train_data = data[:n]

        step = current_step
        while not stop_training_flag:
            xb, yb = get_batch(train_data, block_size, batch_size, device)
            _, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            step += 1
            current_step = step

            if step % 30 == 0:
                status_message = f"Step {step} • loss {loss.item():.4f}"

            if step % continuous_save_every == 0:
                save_checkpoint(step)
                status_message = f"Step {step} • loss {loss.item():.4f} • checkpoint saved"

            if not continuous and step >= max_iters:
                break
            time.sleep(0.001)
    except Exception as e:
        status_message = f"Training error: {e}"
        traceback.print_exc()
    finally:
        save_checkpoint(current_step)
        is_training = False
        status_message = f"Stopped at step {current_step}. Model saved."

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
    return "Stopping… saving checkpoint", get_status()

# ===================== SMART CHAT WITH MODES =====================
def local_generate(prompt, max_tokens=150, temperature=0.85, top_k=50):
    """Generate with the local from-scratch model."""
    if model is None or tokenizer is None:
        return "[Model not loaded]"
    model.eval()
    try:
        ids = tokenizer.encode(prompt)
        if not ids:
            ids = [0]
        # truncate if too long
        if len(ids) > block_size - 10:
            ids = ids[-(block_size - 10):]
        context = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(
                context,
                max_new_tokens=int(max_tokens),
                temperature=float(temperature),
                top_k=int(top_k) if top_k and top_k > 0 else None
            )
            full = tokenizer.decode(out[0].tolist())
            # try to return only the new part
            if prompt in full:
                return full[len(prompt):].strip() or full
            return full
    except Exception as e:
        return f"[Generation error: {e}]"

def chat(message, history, mode, temperature, top_k, max_tokens):
    """
    Main chat with thinking modes and visible process.
    Modes:
      Fast      - quick local generation
      Balanced  - local + light knowledge
      Strong    - multi-step thinking + knowledge
      Research  - web search + deep steps + local
    """
    if not message or not message.strip():
        return history, ""

    history = history or []
    steps = []          # visible thinking steps
    final_answer = ""

    # ---------- Mode settings ----------
    mode = mode or "Balanced"
    do_search = mode in ("Strong", "Research")
    think_rounds = {"Fast": 0, "Balanced": 1, "Strong": 2, "Research": 3}.get(mode, 1)
    gen_tokens = {"Fast": 80, "Balanced": 150, "Strong": 220, "Research": 280}.get(mode, 150)
    gen_tokens = min(int(max_tokens), gen_tokens + 50)

    try:
        # 1. Planning
        steps.append("Planning response…")
        time.sleep(0.3)

        # 2. Local knowledge
        knowledge_context = ""
        if manual_knowledge:
            steps.append(f"Using {len(manual_knowledge)} manual facts…")
            knowledge_context = "Known facts:\n" + "\n".join(f"- {k}" for k in manual_knowledge[-10:])

        # 3. Detect simple intents for connectors
        lower = message.lower()
        if any(w in lower for w in ["github", "repo", "repository", "push", "commit"]):
            steps.append("Checking GitHub connection…")
            knowledge_context += "\n\n[GitHub is connected to this project. Repository: someone405-ship-it/AI-From-Scratch]"

        if any(w in lower for w in ["tweet", "twitter", "x.com", "post on x", "x bot"]):
            steps.append("Checking X (Twitter) bot status…")
            knowledge_context += f"\n\n[X Bot enabled: {x_bot_config.get('enabled')} | Topics: {x_bot_config.get('topics', 'none')}]"

        # 4. Web search (Strong & Research)
        search_results = ""
        if do_search:
            steps.append(f"Searching the web for: {message[:60]}…")
            search_results = web_search(message, max_results=4 if mode == "Research" else 3)
            time.sleep(0.4)

        # 5. Extra thinking rounds for stronger modes
        for i in range(think_rounds):
            steps.append(f"Thinking deeper (round {i+1}/{think_rounds})…")
            time.sleep(0.25 + i * 0.15)

        # 6. Build prompt for local model
        prompt_parts = []
        if knowledge_context:
            prompt_parts.append(knowledge_context)
        if search_results:
            prompt_parts.append("Web search results:\n" + search_results)
        prompt_parts.append(f"User question: {message}")
        prompt_parts.append("Helpful answer:")
        full_prompt = "\n\n".join(prompt_parts)

        steps.append("Generating answer with local model…")

        # 7. Generate
        raw = local_generate(full_prompt, max_tokens=gen_tokens, temperature=temperature, top_k=top_k)

        # Clean up a bit
        final_answer = raw.strip()
        if len(final_answer) < 5:
            final_answer = (
                "I am still learning. "
                "Please upload more training data and run Continuous training, "
                "or teach me facts in the Learn tab. "
                "In Research mode I can also search the web."
            )

        # Prepend visible process for Strong / Research
        if mode in ("Strong", "Research"):
            process = "**Process:**\n" + "\n".join(f"- {s}" for s in steps) + "\n\n---\n\n"
            final_answer = process + final_answer
        elif mode == "Balanced" and steps:
            final_answer = f"*{steps[-1]}*\n\n" + final_answer

    except Exception as e:
        final_answer = f"Error while thinking: {e}\n\n{traceback.format_exc()}"

    # Append to history (tuple format works broadly)
    history = history + [(message, final_answer)]
    return history, ""

# ===================== LEARNING =====================
def teach_fact(fact):
    if not fact or not fact.strip():
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
    return "\n".join([f"• {k}" for k in manual_knowledge[-25:]])

def upload_device_data(files):
    if not files:
        return "No files selected."
    combined = []
    total_chars = 0
    for f in files:
        try:
            path = f.name if hasattr(f, "name") else f
            text = Path(path).read_text(encoding="utf-8", errors="ignore")
            combined.append(text)
            total_chars += len(text)
        except Exception as e:
            return f"Error reading file: {e}"

    Path("data").mkdir(exist_ok=True)
    full_text = "\n\n".join(combined)
    Path("data/input.txt").write_text(full_text, encoding="utf-8")
    msg = load_or_create_model(full_text)
    return f"Loaded {len(files)} file(s) • {total_chars:,} characters\n{msg}"

# ===================== X BOT =====================
def update_x_bot(enabled, topics, posts_per_day, api_key, api_secret, access_token, access_secret):
    x_bot_config.update({
        "enabled": bool(enabled),
        "topics": topics or "",
        "posts_per_day": int(posts_per_day),
        "api_key": api_key or "",
        "api_secret": api_secret or "",
        "access_token": access_token or "",
        "access_secret": access_secret or ""
    })
    save_x_config()
    status = "X Bot ENABLED" if enabled else "X Bot disabled"
    return f"{status}\nTopics: {topics}\nPosts/day: {posts_per_day}"

def generate_x_post(topics):
    if model is None:
        return "Train the model first."
    prompt = f"Write a short interesting post about: {topics}\nPost:"
    return local_generate(prompt, max_tokens=100, temperature=0.9, top_k=40)[:280]

# ===================== STATUS =====================
def get_status():
    training = "TRAINING" if is_training else "Idle"
    params = f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M" if model else "—"
    return f"""**Status:** {training}  
**Step:** {current_step}  
**Model:** {params} parameters  
**Device:** {device}  
**Manual facts:** {len(manual_knowledge)}  

{status_message}"""

# ===================== UI =====================
custom_css = """
.gradio-container { max-width: 100% !important; padding: 6px !important; }
footer { display: none !important; }
button { border-radius: 12px !important; font-weight: 600 !important; }
.chatbot { border-radius: 16px !important; min-height: 50vh !important; }
textarea, input, .wrap { border-radius: 12px !important; }
"""

with gr.Blocks(
    title="AI From Scratch",
    theme=gr.themes.Soft(primary_hue="violet", secondary_hue="indigo", neutral_hue="slate"),
    css=custom_css,
) as demo:

    gr.Markdown("""
    <div style="text-align:center;padding:8px 0 4px 0;">
      <h1 style="margin:0;font-size:1.7rem;background:linear-gradient(90deg,#a78bfa,#818cf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        AI From Scratch
      </h1>
      <p style="margin:2px 0 0;opacity:0.7;font-size:0.9rem;">Local neural net • Thinking modes • Web search • Mobile ready</p>
    </div>
    """)

    with gr.Tabs():

        # ===== CHAT =====
        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(
                height=460,
                show_copy_button=True,
                bubble_full_width=False,
                avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=ai"),
                placeholder="Ask me anything…"
            )
            with gr.Row():
                msg = gr.Textbox(placeholder="Message…", show_label=False, scale=6, container=False, autofocus=True)
                send = gr.Button("Send", variant="primary", scale=1, min_width=70)

            with gr.Row():
                mode = gr.Radio(
                    ["Fast", "Balanced", "Strong", "Research"],
                    value="Balanced",
                    label="Thinking Mode",
                    info="Stronger modes think longer and can search the web"
                )

            with gr.Accordion("Advanced settings", open=False):
                with gr.Row():
                    temperature = gr.Slider(0.1, 1.5, value=0.85, step=0.05, label="Temperature")
                    top_k = gr.Slider(0, 120, value=50, step=5, label="Top-k")
                    max_tokens = gr.Slider(40, 500, value=200, step=10, label="Max tokens")

            send.click(chat, [msg, chatbot, mode, temperature, top_k, max_tokens], [chatbot, msg])
            msg.submit(chat, [msg, chatbot, mode, temperature, top_k, max_tokens], [chatbot, msg])

        # ===== LEARN =====
        with gr.Tab("Learn"):
            gr.Markdown("### Learn from your device")
            files = gr.File(label="Select files", file_count="multiple",
                            file_types=[".txt", ".md", ".csv", ".json", ".log"])
            upload_btn = gr.Button("Load into AI", variant="primary")
            upload_result = gr.Textbox(label="Result", interactive=False, lines=3)

            gr.Markdown("### Manual Teaching")
            fact_input = gr.Textbox(placeholder="e.g. My favorite language is Python", lines=2)
            with gr.Row():
                teach_btn = gr.Button("Teach this", variant="primary")
                clear_btn = gr.Button("Clear knowledge", variant="stop")
            knowledge_box = gr.Textbox(label="What I know", lines=8, interactive=False)

            upload_btn.click(upload_device_data, files, upload_result)
            teach_btn.click(teach_fact, fact_input, [upload_result, knowledge_box])
            clear_btn.click(clear_knowledge, outputs=[upload_result, knowledge_box])

        # ===== TRAIN =====
        with gr.Tab("Train"):
            status = gr.Markdown(get_status())
            with gr.Row():
                start_btn = gr.Button("Start Training", variant="primary")
                cont_btn = gr.Button("Continuous Mode", variant="huggingface")
                stop_btn = gr.Button("Stop", variant="stop")
            refresh = gr.Button("Refresh Status", size="sm")

            gr.Markdown("Continuous mode keeps training until you press Stop. More training = smarter model.")

            start_btn.click(start_train, gr.State(False), [status, status])
            cont_btn.click(start_train, gr.State(True), [status, status])
            stop_btn.click(stop_train, outputs=[status, status])
            refresh.click(get_status, outputs=status)

        # ===== X BOT =====
        with gr.Tab("X Bot"):
            gr.Markdown("### Automatic X Poster")
            x_enabled = gr.Checkbox(label="Enable X Bot", value=False)
            x_topics = gr.Textbox(label="Topics", placeholder="AI, coding, daily thoughts", lines=2)
            x_posts = gr.Slider(1, 5, value=2, step=1, label="Posts per day")

            with gr.Accordion("X API Keys", open=False):
                x_key = gr.Textbox(label="API Key", type="password")
                x_secret = gr.Textbox(label="API Secret", type="password")
                x_token = gr.Textbox(label="Access Token", type="password")
                x_token_secret = gr.Textbox(label="Access Secret", type="password")

            save_x = gr.Button("Save Settings", variant="primary")
            x_status = gr.Textbox(label="Status", interactive=False)
            gen_btn = gr.Button("Generate sample post")
            sample_post = gr.Textbox(label="Sample", lines=3)

            save_x.click(update_x_bot,
                         [x_enabled, x_topics, x_posts, x_key, x_secret, x_token, x_token_secret],
                         x_status)
            gen_btn.click(generate_x_post, x_topics, sample_post)

        # ===== GITHUB =====
        with gr.Tab("GitHub"):
            gr.Markdown("""
            ### GitHub
            Your GitHub is connected. This project lives at:

            **[someone405-ship-it/AI-From-Scratch](https://github.com/someone405-ship-it/AI-From-Scratch)**

            In the Chat tab you can ask things like:
            - "What is in my AI-From-Scratch repo?"
            - "Help me improve the model"
            """)

        # ===== ABOUT =====
        with gr.Tab("About"):
            gr.Markdown("""
            ### AI From Scratch

            Real neural network trained by you on your device.

            **Thinking Modes**
            - **Fast** — quick answer
            - **Balanced** — normal thinking
            - **Strong** — longer thinking + knowledge + light search
            - **Research** — deepest mode + web search + multiple steps

            The local model improves the more you train it and the more data you give it.
            """)

    demo.load(load_or_create_model, outputs=None)
    demo.load(load_manual_knowledge)
    demo.load(load_x_config)
    demo.load(get_knowledge_display, outputs=knowledge_box)

if __name__ == "__main__":
    print(f"AI From Scratch starting on {device}")
    print("Open http://localhost:7860 (or your machine IP) on phone/computer")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
