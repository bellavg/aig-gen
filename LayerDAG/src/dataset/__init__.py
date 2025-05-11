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
        # You'll need to pass the path to your .pt file and NUM_NODE_FEATURES
        # This could come from a config file or be hardcoded if static
        path_to_pt_file = kwargs.get('path')
        num_node_categories = kwargs.get('num_node_categories')
        conditional_flag = kwargs.get('conditional', False) # Default to False
        if path_to_pt_file is None or num_node_categories is None:
            raise ValueError("For AIG dataset, 'path' and 'num_node_categories' must be provided.")
        return get_aig_dataset(path_to_pt_file, num_node_categories, conditional_flag)
    else:
        raise ValueError(f'Unknown dataset: {name}')