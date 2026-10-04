# Powerful configuration for AI From Scratch

# Larger model for better quality (still runnable on most devices)
n_embd = 384
n_head = 8
n_layer = 8
block_size = 512

# Training
batch_size = 32          # lower for mobile / low VRAM
learning_rate = 3e-4
max_iters = 15000
eval_interval = 200
eval_iters = 40
dropout = 0.1
grad_clip = 1.0

# Continuous mode
continuous_save_every = 300

# Generation
default_temperature = 0.85
default_top_k = 60
