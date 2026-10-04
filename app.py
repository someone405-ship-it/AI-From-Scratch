#!/usr/bin/env python3
"""
AI From Scratch - Full Featured Mobile App

Live streaming thinking steps, better web search, conversation memory,
auto status refresh, export chat, strong modes, local training, X Bot, GitHub.
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
                    try:
                        optimizer.load_state_dict(ckpt["optimizer"])
                    except Exception:
                        pass
                current_step = ckpt.get("iter", 0)
                status_message = f"Checkpoint loaded (step {current_step})"
            else:
                status_message = "Vocab changed → new model created"
        except Exception as e:
            status_message = f"Checkpoint note: {e}"
    else:
        params = sum(p.numel() for p in model.parameters()) / 1e6
        status_message = f"New model ready ({params:.1f}M params) on {device}"

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

# ===================== BETTER WEB SEARCH =====================
def web_search(query, max_results=5):
    """Improved DuckDuckGo search with cleaner extraction and fallback."""
    results = []
    try:
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        r = requests.get(url, headers=headers, timeout=12)
        r.raise_for_status()
        html = r.text

        # Multiple patterns for robustness
        titles = re.findall(r'class="result__a"[^>]*>(.*?)</a>', html, re.DOTALL | re.IGNORECASE)
        snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</(?:a|td|div)>', html, re.DOTALL | re.IGNORECASE)

        if not titles:
            titles = re.findall(r'<a[^>]+class="[^"]*result[^"]*"[^>]*>(.*?)</a>', html, re.DOTALL | re.IGNORECASE)

        for i, title in enumerate(titles):
            title_clean = re.sub(r'<[^>]+>', '', title).strip()
            title_clean = re.sub(r'\s+', ' ', title_clean)
            snippet = ""
            if i < len(snippets):
                snippet = re.sub(r'<[^>]+>', '', snippets[i]).strip()
                snippet = re.sub(r'\s+', ' ', snippet)
            if title_clean and len(title_clean) > 3:
                entry = f"**{i+1}. {title_clean}**"
                if snippet:
                    entry += f"\n{snippet[:220]}"
                results.append(entry)
            if len(results) >= max_results:
                break

        if not results:
            # Fallback: try to pull any meaningful text blocks
            texts = re.findall(r'<a[^>]+href="//duckduckgo.com/l/[^"]+"[^>]*>(.*?)</a>', html, re.DOTALL)
            for t in texts[:max_results]:
                clean = re.sub(r'<[^>]+>', '', t).strip()
                if len(clean) > 15:
                    results.append(clean[:180])

        return "\n\n".join(results) if results else "No useful results found for this query."
    except requests.Timeout:
        return "Search timed out. Try again or use a shorter query."
    except Exception as e:
        return f"Search unavailable right now ({type(e).__name__}). Using local knowledge only."

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
        if len(data) < block_size + 2:
            status_message = "Not enough data. Upload more text first."
            is_training = False
            return

        n = max(1, int(0.9 * len(data)))
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

            if step % 25 == 0:
                status_message = f"Step {step} • loss {loss.item():.4f}"

            if step % continuous_save_every == 0:
                save_checkpoint(step)
                status_message = f"Step {step} • loss {loss.item():.4f} • saved"

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

# ===================== LOCAL GENERATION =====================
def local_generate(prompt, max_tokens=150, temperature=0.85, top_k=50):
    if model is None or tokenizer is None:
        return "[Model not loaded — train or upload data first]"
    model.eval()
    try:
        ids = tokenizer.encode(prompt)
        if not ids:
            ids = [0]
        if len(ids) > block_size - 15:
            ids = ids[-(block_size - 15):]
        context = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(
                context,
                max_new_tokens=int(max_tokens),
                temperature=max(0.1, float(temperature)),
                top_k=int(top_k) if top_k and top_k > 0 else None
            )
            full = tokenizer.decode(out[0].tolist())
            if prompt in full:
                reply = full[len(prompt):].strip()
                return reply if reply else full
            return full
    except Exception as e:
        return f"[Generation error: {e}]"

# ===================== STREAMING CHAT =====================
def chat_stream(message, history, mode, temperature, top_k, max_tokens):
    """
    Generator that streams thinking steps live, then the final answer.
    """
    if not message or not str(message).strip():
        yield history, ""
        return

    history = list(history or [])
    mode = mode or "Balanced"
    do_search = mode in ("Strong", "Research")
    think_rounds = {"Fast": 0, "Balanced": 1, "Strong": 2, "Research": 3}.get(mode, 1)
    base_tokens = {"Fast": 90, "Balanced": 160, "Strong": 240, "Research": 320}.get(mode, 160)
    gen_tokens = min(int(max_tokens or 200), base_tokens + 40)

    steps_so_far = []
    process_text = ""

    def update_process(new_step):
        nonlocal process_text, steps_so_far
        steps_so_far.append(new_step)
        process_text = "**Thinking process**\n" + "\n".join(f"- {s}" for s in steps_so_far)
        # Show intermediate state in chat
        temp_history = history + [(message, process_text + "\n\n_Working…_")]
        return temp_history

    try:
        # Step 1
        yield update_process("Planning the best way to answer…"), ""
        time.sleep(0.35)

        # Conversation memory (last 3 turns)
        memory = ""
        if history:
            recent = history[-3:]
            mem_parts = []
            for u, a in recent:
                mem_parts.append(f"User: {u[:120]}")
                # strip previous process blocks
                clean_a = re.sub(r'\*\*Thinking process\*\*.*?---', '', a, flags=re.DOTALL).strip()[:150]
                mem_parts.append(f"AI: {clean_a}")
            memory = "Recent conversation:\n" + "\n".join(mem_parts)

        # Local knowledge
        knowledge_context = ""
        if manual_knowledge:
            yield update_process(f"Recalling {len(manual_knowledge)} taught facts…"), ""
            time.sleep(0.25)
            knowledge_context = "Known facts:\n" + "\n".join(f"- {k}" for k in manual_knowledge[-12:])

        # Intent detection for connectors
        lower = message.lower()
        if any(w in lower for w in ["github", "repo", "repository", "push code", "commit", "pull request"]):
            yield update_process("Connecting to GitHub…"), ""
            time.sleep(0.3)
            knowledge_context += "\n\n[GitHub connected. Repo: someone405-ship-it/AI-From-Scratch]"

        if any(w in lower for w in ["tweet", "twitter", " post on x", "x bot", "x.com"]):
            yield update_process("Checking X (Twitter) bot configuration…"), ""
            time.sleep(0.25)
            knowledge_context += f"\n\n[X Bot: enabled={x_bot_config.get('enabled')}, topics={x_bot_config.get('topics', 'none')}]"

        # Web search
        search_results = ""
        if do_search:
            yield update_process(f"Searching the web for “{message[:50]}”…"), ""
            search_results = web_search(message, max_results=5 if mode == "Research" else 3)
            time.sleep(0.4)
            yield update_process("Search finished — analyzing results…"), ""
            time.sleep(0.3)

        # Extra thinking rounds
        for i in range(think_rounds):
            yield update_process(f"Deep thinking round {i+1}/{think_rounds}…"), ""
            time.sleep(0.35 + i * 0.2)

        # Build final prompt
        prompt_parts = []
        if memory:
            prompt_parts.append(memory)
        if knowledge_context:
            prompt_parts.append(knowledge_context)
        if search_results:
            prompt_parts.append("Web information:\n" + search_results)
        prompt_parts.append(f"Current user question: {message}")
        prompt_parts.append("Clear helpful answer:")
        full_prompt = "\n\n".join(prompt_parts)

        yield update_process("Generating answer with local neural network…"), ""
        time.sleep(0.2)

        raw = local_generate(full_prompt, max_tokens=gen_tokens, temperature=temperature, top_k=top_k)
        final_answer = raw.strip()

        if len(final_answer) < 8:
            final_answer = (
                "I am still learning from the data you give me.\n\n"
                "Tips to make me better:\n"
                "1. Upload more text files in the Learn tab\n"
                "2. Run Continuous training for a while\n"
                "3. Teach me important facts manually\n"
                "4. Use Research mode for questions that need current information"
            )

        # Final message with process (for Strong/Research) or clean (for Fast)
        if mode in ("Strong", "Research"):
            process_block = process_text + "\n\n---\n\n"
            final_message = process_block + final_answer
        elif mode == "Balanced":
            final_message = f"*{steps_so_far[-1] if steps_so_far else 'Done'}*\n\n" + final_answer
        else:
            final_message = final_answer

        new_history = history + [(message, final_message)]
        yield new_history, ""

    except Exception as e:
        err = f"Something went wrong while thinking:\n{e}"
        new_history = history + [(message, err)]
        yield new_history, ""

# ===================== LEARNING =====================
def teach_fact(fact):
    if not fact or not str(fact).strip():
        return "Write something to teach me.", get_knowledge_display()
    manual_knowledge.append(str(fact).strip())
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
    return "\n".join([f"• {k}" for k in manual_knowledge[-30:]])

def upload_device_data(files):
    if not files:
        return "No files selected."
    combined = []
    total_chars = 0
    count = 0
    for f in files:
        try:
            path = getattr(f, "name", f)
            text = Path(path).read_text(encoding="utf-8", errors="ignore")
            combined.append(text)
            total_chars += len(text)
            count += 1
        except Exception as e:
            return f"Error reading a file: {e}"

    Path("data").mkdir(exist_ok=True)
    full_text = "\n\n".join(combined)
    Path("data/input.txt").write_text(full_text, encoding="utf-8")
    msg = load_or_create_model(full_text)
    return f"Loaded {count} file(s) • {total_chars:,} characters\n{msg}"

# ===================== X BOT =====================
def update_x_bot(enabled, topics, posts_per_day, api_key, api_secret, access_token, access_secret):
    x_bot_config.update({
        "enabled": bool(enabled),
        "topics": topics or "",
        "posts_per_day": int(posts_per_day or 2),
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
    prompt = f"Write a short interesting social media post about: {topics}\nPost:"
    return local_generate(prompt, max_tokens=90, temperature=0.9, top_k=40)[:280]

# ===================== STATUS & EXPORT =====================
def get_status():
    training = "TRAINING" if is_training else "Idle"
    params = f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M" if model else "—"
    vocab = tokenizer.vocab_size if tokenizer else "—"
    return f"""**Status:** {training}  
**Step:** {current_step}  
**Model:** {params} parameters  
**Vocab:** {vocab}  
**Device:** {device}  
**Manual facts:** {len(manual_knowledge)}  

{status_message}"""

def export_chat(history):
    if not history:
        return "Nothing to export yet."
    lines = [f"AI From Scratch Chat Export — {datetime.now().strftime('%Y-%m-%d %H:%M')}", "=" * 50, ""]
    for user_msg, ai_msg in history:
        lines.append(f"YOU: {user_msg}")
        # clean process blocks for export
        clean = re.sub(r'\*\*Thinking process\*\*.*?---\s*', '', ai_msg, flags=re.DOTALL).strip()
        lines.append(f"AI: {clean}")
        lines.append("")
    return "\n".join(lines)

def clear_chat():
    return [], ""

# ===================== UI =====================
custom_css = """
.gradio-container { max-width: 100% !important; padding: 6px !important; }
footer { display: none !important; }
button { border-radius: 12px !important; font-weight: 600 !important; }
.chatbot { border-radius: 16px !important; min-height: 48vh !important; }
textarea, input { border-radius: 12px !important; }
"""

with gr.Blocks(
    title="AI From Scratch",
    theme=gr.themes.Soft(primary_hue="violet", secondary_hue="indigo", neutral_hue="slate"),
    css=custom_css,
) as demo:

    gr.Markdown("""
    <div style="text-align:center;padding:8px 0 2px 0;">
      <h1 style="margin:0;font-size:1.65rem;background:linear-gradient(90deg,#a78bfa,#818cf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        AI From Scratch
      </h1>
      <p style="margin:2px 0 0;opacity:0.75;font-size:0.88rem;">Live thinking • Web search • Local training • Mobile ready</p>
    </div>
    """)

    with gr.Tabs():

        # ===== CHAT =====
        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(
                height=440,
                show_copy_button=True,
                bubble_full_width=False,
                avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=ai"),
                placeholder="Ask me anything — watch me think live…"
            )
            with gr.Row():
                msg = gr.Textbox(placeholder="Message…", show_label=False, scale=6, container=False, autofocus=True)
                send = gr.Button("Send", variant="primary", scale=1, min_width=70)

            with gr.Row():
                mode = gr.Radio(
                    ["Fast", "Balanced", "Strong", "Research"],
                    value="Balanced",
                    label="Thinking Mode",
                    info="Research = live steps + web search + deepest reasoning"
                )

            with gr.Accordion("Settings & tools", open=False):
                with gr.Row():
                    temperature = gr.Slider(0.1, 1.5, value=0.85, step=0.05, label="Temperature")
                    top_k = gr.Slider(0, 120, value=50, step=5, label="Top-k")
                    max_tokens = gr.Slider(40, 500, value=220, step=10, label="Max tokens")
                with gr.Row():
                    clear_btn = gr.Button("Clear chat", size="sm")
                    export_btn = gr.Button("Export chat", size="sm")
                export_box = gr.Textbox(label="Exported conversation", lines=6, visible=False)

            # Streaming events
            send.click(
                chat_stream,
                [msg, chatbot, mode, temperature, top_k, max_tokens],
                [chatbot, msg]
            )
            msg.submit(
                chat_stream,
                [msg, chatbot, mode, temperature, top_k, max_tokens],
                [chatbot, msg]
            )
            clear_btn.click(clear_chat, outputs=[chatbot, msg])
            export_btn.click(export_chat, chatbot, export_box).then(
                lambda: gr.update(visible=True), None, export_box
            )

        # ===== LEARN =====
        with gr.Tab("Learn"):
            gr.Markdown("### Learn from your device")
            files = gr.File(label="Select files from phone or computer", file_count="multiple",
                            file_types=[".txt", ".md", ".csv", ".json", ".log"])
            upload_btn = gr.Button("Load into AI", variant="primary")
            upload_result = gr.Textbox(label="Result", interactive=False, lines=3)

            gr.Markdown("### Manual Teaching")
            fact_input = gr.Textbox(placeholder="Teach me a fact, e.g. My name is Sam and I build robots", lines=2)
            with gr.Row():
                teach_btn = gr.Button("Teach this", variant="primary")
                clear_k_btn = gr.Button("Clear all knowledge", variant="stop")
            knowledge_box = gr.Textbox(label="What I currently know", lines=8, interactive=False)

            upload_btn.click(upload_device_data, files, upload_result)
            teach_btn.click(teach_fact, fact_input, [upload_result, knowledge_box])
            clear_k_btn.click(clear_knowledge, outputs=[upload_result, knowledge_box])

        # ===== TRAIN =====
        with gr.Tab("Train"):
            status = gr.Markdown(get_status())
            with gr.Row():
                start_btn = gr.Button("Start Training", variant="primary")
                cont_btn = gr.Button("Continuous Mode", variant="huggingface")
                stop_btn = gr.Button("Stop", variant="stop")
            refresh = gr.Button("Refresh Status", size="sm")

            # Auto refresh every few seconds while training
            timer = gr.Timer(3)
            timer.tick(get_status, outputs=status)

            gr.Markdown("Continuous mode keeps improving the model until you press Stop. More data + more time = smarter AI.")

            start_btn.click(start_train, gr.State(False), [status, status])
            cont_btn.click(start_train, gr.State(True), [status, status])
            stop_btn.click(stop_train, outputs=[status, status])
            refresh.click(get_status, outputs=status)

        # ===== X BOT =====
        with gr.Tab("X Bot"):
            gr.Markdown("### Automatic X (Twitter) Poster")
            x_enabled = gr.Checkbox(label="Enable X Bot", value=False)
            x_topics = gr.Textbox(label="Topics to post about", placeholder="AI, coding, daily thoughts…", lines=2)
            x_posts = gr.Slider(1, 5, value=2, step=1, label="Posts per day")

            with gr.Accordion("X API Keys (needed for real posting)", open=False):
                x_key = gr.Textbox(label="API Key", type="password")
                x_secret = gr.Textbox(label="API Secret", type="password")
                x_token = gr.Textbox(label="Access Token", type="password")
                x_token_secret = gr.Textbox(label="Access Secret", type="password")

            save_x = gr.Button("Save Bot Settings", variant="primary")
            x_status = gr.Textbox(label="Bot Status", interactive=False)
            gen_btn = gr.Button("Generate sample post")
            sample_post = gr.Textbox(label="Sample post", lines=3)

            save_x.click(update_x_bot,
                         [x_enabled, x_topics, x_posts, x_key, x_secret, x_token, x_token_secret],
                         x_status)
            gen_btn.click(generate_x_post, x_topics, sample_post)

        # ===== GITHUB =====
        with gr.Tab("GitHub"):
            gr.Markdown("""
            ### GitHub Integration
            This project is already on your GitHub:

            **[someone405-ship-it/AI-From-Scratch](https://github.com/someone405-ship-it/AI-From-Scratch)**

            In Chat you can ask about the repository, request improvements, or discuss code.
            When the AI detects GitHub-related questions it shows “Connecting to GitHub…” in the thinking process.
            """)

        # ===== ABOUT =====
        with gr.Tab("About"):
            gr.Markdown("""
            ### AI From Scratch

            A real neural network you train yourself, with modern agent-style features.

            **Thinking Modes**
            - **Fast** — quick local answer
            - **Balanced** — normal reasoning + memory
            - **Strong** — live thinking steps + knowledge + search
            - **Research** — deepest mode, multiple rounds, best search

            **Key features**
            - Live streaming of every thinking step
            - Web search (no API key needed)
            - Conversation memory
            - Learn from device files + manual teaching
            - Continuous training
            - X Bot configuration
            - Export chat history
            - Mobile-first UI

            The more high-quality data you give it and the longer you train, the better it becomes.
            """)

    demo.load(load_or_create_model, outputs=None)
    demo.load(load_manual_knowledge)
    demo.load(load_x_config)
    demo.load(get_knowledge_display, outputs=knowledge_box)

if __name__ == "__main__":
    print(f"\nAI From Scratch starting on {device}")
    print("Open http://localhost:7860 (or your computer IP) on phone or desktop\n")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
