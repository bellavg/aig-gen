# --- Primary Configuration Constants ---
dataset = 'aig'

# AIG constraint constants
MAX_NODE_COUNT = 64
MIN_PI_COUNT = 2
MAX_PI_COUNT = 8
MIN_PO_COUNT = 1
MAX_PO_COUNT = 8
MIN_AND_COUNT = 1 # Assuming at least one AND gate needed


NUM_ADJ_CHANNELS = 3

NODE_TYPE_KEYS = ["NODE_CONST0", "NODE_PI", "NODE_AND", "NODE_PO"]
EDGE_TYPE_KEYS = ["EDGE_REG", "EDGE_INV"]


NUM_NODE_FEATURES = len(NODE_TYPE_KEYS) # Should be 4
NUM_EXPLICIT_EDGE_TYPES = len(EDGE_TYPE_KEYS) # Should be 2

# Define one-hot encodings (Keep these hardcoded as they define the feature representation)
NODE_TYPE_ENCODING_NX = {
    # Map "NODE_CONST0" to the first encoding vector (index 0), etc.
    "NODE_CONST0": [1.0, 0.0, 0.0, 0.0], # Index 0 feature
    "NODE_PI":     [0.0, 1.0, 0.0, 0.0], # Index 1 feature
    "NODE_AND":    [0.0, 0.0, 1.0, 0.0], # Index 2 feature
    "NODE_PO":     [0.0, 0.0, 0.0, 1.0]  # Index 3 feature
}

# IMPORTANT: Ensure keys/order match EDGE_TYPE_VOCAB derivation and NUM_EDGE_FEATURES
# The order here should ideally match the order in EDGE_TYPE_KEYS
EDGE_TYPE_ENCODING_NX = {
    "EDGE_REG": [1.0, 0.0],  # Index 0 feature
    "EDGE_INV": [0.0, 1.0]   # Index 1 feature
}


# Define one-hot encodings (Keep these hardcoded as they define the feature representation)
NODE_TYPE_ENCODING_PYG = {
    # Map "NODE_CONST0" to the first encoding vector (index 0), etc.
    "NODE_CONST0": 0, # Index 0 feature
    "NODE_PI":     1, # Index 1 feature
    "NODE_AND":    2, # Index 2 feature
    "NODE_PO":     3  # Index 3 feature
}

# Define one-hot encodings (Keep these hardcoded as they define the feature representation)
NUM2NODETYPE = {
    # Map "NODE_CONST0" to the first encoding vector (index 0), etc.
    0: "NODE_CONST0", # Index 0 feature
    1: "NODE_PI", # Index 1 feature
    2: "NODE_AND", # Index 2 feature
    3: "NODE_PO"  # Index 3 feature
}
# IMPORTANT: Ensure keys/order match EDGE_TYPE_VOCAB derivation and NUM_EDGE_FEATURES
# The order here should ideally match the order in EDGE_TYPE_KEYS
EDGE_LABEL_ENCODING_PYG = {
    "EDGE_REG": 0,  # Index 0 feature
    "EDGE_INV": 1,   # Index 1 feature
    "NONE" : 2
}
VIRTUAL_EDGE_INDEX = EDGE_LABEL_ENCODING_PYG['NONE']
# IMPORTANT: Ensure keys/order match EDGE_TYPE_VOCAB derivation and NUM_EDGE_FEATURES
# The order here should ideally match the order in EDGE_TYPE_KEYS
NUM2EDGETYPE = {
    0: "EDGE_REG",  # Index 0 feature
    1: "EDGE_INV",   # Index 1 feature
}

def check_validity():
    pass # return True or False

