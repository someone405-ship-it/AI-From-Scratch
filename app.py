#!/usr/bin/env python3
"""
Web Interface for AI From Scratch
Beautiful Gradio UI with:
- Chat with your trained model
- Upload training data
- Start / Stop continuous training
- Live status and controls
"""

import os
import threading
import time
from pathlib import Path

import gradio as gr
import torch

from config import *
from model import GPTLanguageModel
from utils import CharTokenizer, get_batch, estimate_loss, get_device, load_text

# Global state
device = get_device()
model = None
tokenizer = None
optimizer = None
training_thread = None
is_training = False
stop_training_flag = False
current_step = 0
train_loss_history = []
status_message = "Ready. Train a model or load a checkpoint first."

ckpt_dir = Path("checkpoints")
ckpt_dir.mkdir(exist_ok=True)
latest_ckpt = ckpt_dir / "latest.pt"

def load_or_create_model(text=None):
    global model, tokenizer, optimizer, status_message

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
            checkpoint = torch.load(latest_ckpt, map_location=device)
            # Only load if vocab size matches
            if checkpoint["config"]["vocab_size"] == tokenizer.vocab_size:
                model.load_state_dict(checkpoint["model"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                global current_step
                current_step = checkpoint.get("iter", 0)
                status_message = f"Loaded checkpoint (step {current_step})"
            else:
                status_message = "Checkpoint vocab mismatch — starting fresh model"
        except Exception as e:
            status_message = f"Could not load checkpoint: {e}"
    else:
        status_message = f"New model created ({sum(p.numel() for p in model.parameters())/1e6:.2f}M params)"

    return status_message

def save_checkpoint(step):
    if model is None or tokenizer is None:
        return
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iter": step,
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

def training_loop(continuous=True):
    global is_training, stop_training_flag, current_step, train_loss_history, status_message

    is_training = True
    stop_training_flag = False
    status_message = "Training started..."

    text = load_text("data/input.txt")
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]

    step = current_step
    try:
        while not stop_training_flag:
            xb, yb = get_batch(train_data, block_size, batch_size, device)
            logits, loss = model(xb, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            step += 1
            current_step = step
            train_loss_history.append(loss.item())

            if step % 50 == 0:
                status_message = f"Step {step} | loss {loss.item():.4f}"

            if step % continuous_save_every == 0:
                save_checkpoint(step)
                status_message = f"Step {step} | loss {loss.item():.4f} | checkpoint saved"

            if not continuous and step >= max_iters:
                break

            time.sleep(0.001)  # tiny yield so UI stays responsive

    except Exception as e:
        status_message = f"Training error: {e}"
    finally:
        save_checkpoint(step)
        is_training = False
        status_message = f"Training stopped at step {step}. Model saved."

def start_training(continuous):
    global training_thread, is_training
    if is_training:
        return "Already training!", gr.update()

    if model is None:
        load_or_create_model()

    training_thread = threading.Thread(target=training_loop, args=(continuous,), daemon=True)
    training_thread.start()
    mode = "CONTINUOUS" if continuous else "normal"
    return f"Started {mode} training...", gr.update(interactive=False)

def stop_training():
    global stop_training_flag
    if not is_training:
        return "Not currently training."
    stop_training_flag = True
    return "Stopping training... (saving checkpoint)"

def chat(message, history, temperature, top_k, max_tokens):
    if model is None or tokenizer is None:
        return history + [[message, "Please train or load a model first."]], ""

    model.eval()
    try:
        context = torch.tensor([tokenizer.encode(message)], dtype=torch.long, device=device)
        if context.shape[1] == 0:
            context = torch.zeros((1, 1), dtype=torch.long, device=device)

        with torch.no_grad():
            generated = model.generate(
                context,
                max_new_tokens=int(max_tokens),
                temperature=float(temperature),
                top_k=int(top_k) if top_k > 0 else None
            )
            full = tokenizer.decode(generated[0].tolist())
            # return only the new part
            reply = full[len(message):] if message else full
    except Exception as e:
        reply = f"Error during generation: {e}"

    history = history + [[message, reply]]
    return history, ""

def upload_data(file):
    if file is None:
        return "No file uploaded."
    try:
        content = Path(file.name).read_text(encoding="utf-8", errors="ignore")
        Path("data").mkdir(exist_ok=True)
        Path("data/input.txt").write_text(content, encoding="utf-8")
        # rebuild model with new vocab
        msg = load_or_create_model(content)
        return f"Data uploaded ({len(content):,} characters).\n{msg}"
    except Exception as e:
        return f"Upload failed: {e}"

def get_status():
    global status_message
    training_status = "TRAINING" if is_training else "Idle"
    params = f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M" if model else "N/A"
    return f"""**Status:** {training_status}
**Step:** {current_step}
**Model size:** {params}
**Device:** {device}

{status_message}"""

# ========== Gradio UI ==========
with gr.Blocks(
    title="AI From Scratch",
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="blue"),
    css="""
    .gradio-container { max-width: 1100px !important; }
    footer { display: none !important; }
    """
) as demo:
    gr.Markdown("""
    # AI From Scratch
    **Your own neural network** — trained by you, running on your machine.
    No external APIs. Pure PyTorch Transformer built from the ground up.
    """)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=480, label="Chat with your AI", show_copy_button=True)
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Type a message and press Enter...",
                    show_label=False,
                    scale=5,
                    container=False
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

            with gr.Accordion("Generation Settings", open=False):
                temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Temperature (creativity)")
                top_k = gr.Slider(0, 200, value=50, step=5, label="Top-k (0 = disabled)")
                max_tokens = gr.Slider(50, 800, value=250, step=10, label="Max new tokens")

        with gr.Column(scale=2):
            status_box = gr.Markdown(get_status())
            refresh_btn = gr.Button("Refresh Status", size="sm")

            gr.Markdown("### Training Controls")
            with gr.Row():
                train_btn = gr.Button("Start Training", variant="primary")
                continuous_btn = gr.Button("Start Continuous", variant="huggingface")
            stop_btn = gr.Button("Stop Training", variant="stop")

            gr.Markdown("### Upload Training Data")
            file_upload = gr.File(label="Upload .txt file", file_types=[".txt"])
            upload_status = gr.Textbox(label="Upload result", interactive=False)

            gr.Markdown("""
            ### Tips
            - Upload a large text file (books, code, notes...)
            - Click **Start Continuous** and let it train for a while
            - Then chat with your model
            - Bigger data + longer training = smarter AI
            """)

    # Events
    send_btn.click(chat, inputs=[msg, chatbot, temperature, top_k, max_tokens], outputs=[chatbot, msg])
    msg.submit(chat, inputs=[msg, chatbot, temperature, top_k, max_tokens], outputs=[chatbot, msg])

    train_btn.click(start_training, inputs=[gr.State(False)], outputs=[status_box, train_btn])
    continuous_btn.click(start_training, inputs=[gr.State(True)], outputs=[status_box, continuous_btn])
    stop_btn.click(stop_training, outputs=status_box)
    refresh_btn.click(get_status, outputs=status_box)

    file_upload.upload(upload_data, inputs=file_upload, outputs=upload_status)

    # Load model on startup
    demo.load(load_or_create_model, outputs=status_box)

if __name__ == "__main__":
    print(f"Starting web interface on device: {device}")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
