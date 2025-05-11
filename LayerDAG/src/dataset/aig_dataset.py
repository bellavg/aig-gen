# In e.g., src/dataset/aig_dataset.py
import torch
from .general import DAGDataset #
import os # For path joining

# Assuming NUM_NODE_FEATURES is defined, e.g., imported from your aig_config
# from aig_config import NUM_NODE_FEATURES # Or pass it as an argument

def to_aig_dag_dataset(pyg_data_list, num_node_categories, conditional=False):
    # num_node_categories will be your NUM_NODE_FEATURES
    dataset = DAGDataset(num_categories=num_node_categories, label=conditional)

    for pyg_data in pyg_data_list:
        src = pyg_data.edge_index[0, :]
        dst = pyg_data.edge_index[1, :]

        # Convert one-hot node features to categorical integer labels
        if pyg_data.x.ndim == 2: # Should be (num_nodes, NUM_NODE_FEATURES)
            x_n = torch.argmax(pyg_data.x, dim=1)
        else:
            # Handle cases if x is already categorical or needs other processing
            # This branch might indicate an issue if one-hot was expected
            x_n = pyg_data.x

        y_val = None
        if conditional:
            # Decide what your graph-level label 'y' should be.
            # For example, using 'num_gates' if available and desired.
            # This needs to be a single scalar per graph.
            # If using num_gates:
            # y_val = pyg_data.num_gates.item() if hasattr(pyg_data, 'num_gates') else 0
            # For now, let's assume a placeholder if true unconditional isn't fully supported
            # by the sample script without modification.
            # Or, for truly unconditional, you'd ensure the model parts don't expect 'y'.
            pass # Define y_val if conditional is True

        dataset.add_data(src, dst, x_n, y_val)
    return dataset

def get_aig_dataset(path_to_your_pt_file, num_node_categories_from_config, conditional_flag):
    print(f"Loading AIG dataset from {path_to_your_pt_file}...")
    # Ensure you load with weights_only=False if it contains complex objects like PyG Data
    loaded_data_list = torch.load(path_to_your_pt_file, map_location=torch.device('cpu'), weights_only=False) # Add weights_only=False

    # Assuming loaded_data_list is a list of your PyG Data objects.
    # You might need to split this into train/val/test sets.
    # For simplicity, let's assume the whole file is for training for now,
    # and you'll create a smaller validation set manually or by splitting.

    # Example: Splitting (e.g., 80% train, 20% val)
    num_graphs = len(loaded_data_list)
    train_split_idx = int(num_graphs * 0.8)

    train_pyg_list = loaded_data_list[:train_split_idx]
    val_pyg_list = loaded_data_list[train_split_idx:]
    # test_pyg_list could be another split if you have a separate test file or split further

    train_set = to_aig_dag_dataset(train_pyg_list, num_node_categories_from_config, conditional_flag)
    val_set = to_aig_dag_dataset(val_pyg_list, num_node_categories_from_config, conditional_flag)
    test_set = None # Or create one if needed

    # It's important that train_set.num_categories is correctly set.
    # The DAGDataset constructor handles num_categories, and num_categories + 1 is used internally.
    # The dummy_category is num_categories (original).
    # So, if NUM_NODE_FEATURES is, say, 5, then num_categories should be 5.
    # The model will see num_categories + 1 (0-4 as features, 5 as dummy).

    return train_set, val_set, test_set