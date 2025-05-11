# In e.g., src/dataset/aig_dataset.py
import torch
from .general import DAGDataset  # Assuming general.py is in the same directory
import os  # For path joining


# Assuming NUM_NODE_FEATURES is defined, e.g., imported from your aig_config
# from aig_config import NUM_NODE_FEATURES # Or pass it as an argument

def to_aig_dag_dataset(pyg_data_list, num_node_categories, conditional=False):
    # num_node_categories is the number of actual distinct node types (e.g., 4 for your AIGs)
    # DAGDataset constructor will add 1 for the dummy category internally.
    dataset = DAGDataset(num_categories=num_node_categories, label=conditional)

    for pyg_data in pyg_data_list:
        if not hasattr(pyg_data, 'edge_index') or not hasattr(pyg_data, 'x'):
            print(f"Warning: Skipping PyG data object missing 'edge_index' or 'x'. Object: {pyg_data}")
            continue

        src = pyg_data.edge_index[0, :]
        dst = pyg_data.edge_index[1, :]

        # Convert one-hot node features to categorical integer labels
        if pyg_data.x.ndim == 2:  # Should be (num_nodes, NUM_NODE_FEATURES)
            x_n = torch.argmax(pyg_data.x, dim=1)
        elif pyg_data.x.ndim == 1:  # If x_n is already categorical
            x_n = pyg_data.x
        else:
            print(f"Warning: Skipping PyG data object with unexpected x.ndim: {pyg_data.x.ndim}. Object: {pyg_data}")
            continue  # Skip this graph if node features are not in expected format

        y_val = None
        if conditional:
            # This part is currently not used since your aig.yaml has conditional: false
            # If you were to use conditional generation:
            # if hasattr(pyg_data, 'num_gates') and pyg_data.num_gates is not None:
            #     y_val = pyg_data.num_gates.item()
            # else:
            #     y_val = 0 # Default or placeholder if attribute is missing
            pass

        dataset.add_data(src, dst, x_n, y_val)
    return dataset


def get_aig_dataset(path_to_your_pt_file, num_node_categories_from_config, conditional_flag):
    """
    Loads the AIG dataset from a .pt file containing a list of PyG Data objects.

    Args:
        path_to_your_pt_file (str): Path to the .pt file.
        num_node_categories_from_config (int): The number of distinct node types
                                               (e.g., 4 for your AIGs: CONST0, PI, AND, PO).
        conditional_flag (bool): Whether the dataset should be loaded for conditional generation.

    Returns:
        tuple: (train_set, val_set, test_set)
    """
    print(f"Loading AIG dataset from {path_to_your_pt_file}...")

    # For PyTorch 1.12.0, the 'weights_only' argument is not supported.
    # Remove it.
    loaded_data_list = torch.load(path_to_your_pt_file, map_location=torch.device('cpu'))

    if not isinstance(loaded_data_list, list):
        print(
            f"Warning: Loaded data from {path_to_your_pt_file} is not a list. Attempting to use it directly if it's a single graph, otherwise this will fail.")
        # Handle case where it might be a single graph or other structure if necessary,
        # but the processing script saved a list.
        # For now, assume it should be a list. If it's a single item, wrap it.
        # if not isinstance(loaded_data_list, list) and hasattr(loaded_data_list, 'edge_index'): # Heuristic for single PyG Data
        #    loaded_data_list = [loaded_data_list]
        # else:
        raise TypeError(
            f"Expected a list of PyG Data objects from {path_to_your_pt_file}, but got {type(loaded_data_list)}")

    # Example: Splitting (e.g., 80% train, 20% val)
    num_graphs = len(loaded_data_list)
    if num_graphs == 0:
        raise ValueError(f"No graphs found in {path_to_your_pt_file}.")

    train_split_idx = int(num_graphs * 0.8)

    if train_split_idx == 0 and num_graphs > 0:  # Ensure val_set is not empty if there's only 1 graph total
        train_split_idx = 1
    if train_split_idx == num_graphs and num_graphs > 1:  # Ensure val_set has at least one graph if possible
        train_split_idx = num_graphs - 1

    train_pyg_list = loaded_data_list[:train_split_idx]
    val_pyg_list = loaded_data_list[train_split_idx:]

    if not train_pyg_list:
        raise ValueError("Training list is empty after split. Check dataset size and split logic.")
    if not val_pyg_list:
        print(
            "Warning: Validation list is empty after split. Consider a larger dataset or different split for proper validation.")
        # Create a dummy val_set if absolutely necessary for code flow, but this is not ideal for training
        # For now, let it proceed, but training might behave unexpectedly without validation data.
        # Alternatively, could duplicate one training sample for validation if code requires non-empty val_set.
        # val_pyg_list = [train_pyg_list[0]] # Example: use one training sample for val if val is empty.

    print(
        f"Total graphs loaded: {num_graphs}. Training graphs: {len(train_pyg_list)}, Validation graphs: {len(val_pyg_list)}")

    train_set = to_aig_dag_dataset(train_pyg_list, num_node_categories_from_config, conditional_flag)
    # Only create val_set if val_pyg_list is not empty
    val_set = to_aig_dag_dataset(val_pyg_list, num_node_categories_from_config,
                                 conditional_flag) if val_pyg_list else None

    test_set = None  # Or create one if needed for later evaluation

    return train_set, val_set, test_set
