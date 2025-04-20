
# Configuration for a medium-sized G2PT model

# Number of transformer layers
n_layer = 18

# Number of attention heads
n_head = 14

# Embedding dimension (must be divisible by n_head)
n_embd = 896

# Dropout rate (set during training command or in default config)
dropout = 0.1

# Use bias in Linear and LayerNorm layers (False is often recommended)
bias = False

# Model name identifier
model_name = 'medium'