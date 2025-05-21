# In src/dataset/__init__.py
# Import your new loader
from .aig_dataset import get_aig_dataset # Assuming you named your file aig_dataset.py
from .general import DAGDataset
from .layer_dag import *
from .tpu_tile import get_tpu_tile

def load_dataset(name: str, **kwargs): # Add **kwargs to pass path and num_categories
    if name == 'tpu_tile':
        return get_tpu_tile()
    elif name == 'aig':
        return get_aig_dataset(path_to_pt_file, num_node_categories, conditional_flag)
    else:
        raise ValueError(f'Unknown dataset: {name}')