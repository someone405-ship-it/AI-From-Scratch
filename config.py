# Configuration for the from-scratch AI

# Model size (start small, increase if you have a good GPU)
n_embd = 128        # embedding dimension
n_head = 4          # number of attention heads
n_layer = 4         # number of transformer layers
block_size = 128    # context length (how many characters the model sees at once)

# Training
batch_size = 32
learning_rate = 3e-4
max_iters = 5000            # for normal training
eval_interval = 200
eval_iters = 50
dropout = 0.1

# Continuous training mode
continuous_save_every = 500  # save checkpoint every N steps in continuous mode

# Device
# Will automatically use CUDA if available
