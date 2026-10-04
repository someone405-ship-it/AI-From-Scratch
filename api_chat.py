"""
Clean API helper for the mobile app.
This file is imported by app.py to provide a simple non-streaming chat endpoint.
"""

def create_api_chat_fn(chat_stream_fn):
    """Wraps the streaming chat into a simple final-answer function for mobile."""

    def api_chat(message, mode="Balanced", temperature=0.85, top_k=50, max_tokens=220):
        if not message or not str(message).strip():
            return "Please send a message."

        history = []
        final_history = history
        try:
            # Run the generator until the last yield
            gen = chat_stream_fn(message, history, mode, temperature, top_k, max_tokens)
            for result in gen:
                final_history = result[0]

            if final_history and len(final_history) > 0:
                last = final_history[-1]
                # last is (user_msg, ai_msg)
                ai_reply = last[1] if isinstance(last, (list, tuple)) and len(last) > 1 else str(last)
                return ai_reply
            return "No response generated."
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

    return api_chat
