# Configuration for the from-scratch AI
# Made more powerful by default

# Model size (increased for better quality)
n_embd = 256        # embedding dimension (was 128)
n_head = 8          # number of attention heads (was 4)
n_layer = 6         # number of transformer layers (was 4)
block_size = 256    # context length (was 128)

# Training
batch_size = 64
learning_rate = 3e-4
max_iters = 10000
eval_interval = 250
eval_iters = 50
dropout = 0.1
grad_clip = 1.0     # gradient clipping for stability

# Continuous training mode
continuous_save_every = 500

# Generation defaults
default_temperature = 0.8
default_top_k = 50
