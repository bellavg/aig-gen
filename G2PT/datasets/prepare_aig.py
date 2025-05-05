# G2PT/datasets/prepare_aig.py
# Stage 2: Convert AIG PyG Data objects into final .bin format.
# MODIFIED: Bypasses AIGPygDataModule initialization to avoid issues with missing 'test' split.
#           Loads AIGPygDataset directly for train and val splits.
# Saves node/edge vocab IDs, edge indices, AND num_inputs/num_outputs using memmap.
# Uses configurations from G2PT/configs/aig.py.
# Processes ONLY train and validation splits.

import os
import numpy as np
import json
import torch
import warnings
import sys
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from types import SimpleNamespace # Import SimpleNamespace

# --- Configuration ---
# Import AIG specific configurations
try:
    # Try direct import first
    import G2PT.configs.aig as aig_cfg
    print("Successfully imported G2PT.configs.aig")
except ImportError:
    print("Could not import G2PT.configs.aig directly. Adjusting sys.path.")
    script_dir = os.path.dirname(os.path.realpath(__file__))
    g2pt_root = os.path.dirname(script_dir)
    if g2pt_root not in sys.path: sys.path.insert(0, g2pt_root)
    try:
        import G2PT.configs.aig as aig_cfg
        print(f"Successfully imported G2PT.configs.aig after path adjustment.")
    except ImportError as e:
        print(f"Error importing G2PT.configs.aig even after path adjustment: {e}")
        sys.exit(1)
except Exception as e:
     print(f"An unexpected error occurred during config import: {e}")
     sys.exit(1)

# --- Import the AIGPygDataset class directly ---
try:
    # Ensure the path adjustment above allows this import
    from G2PT.datasets.aig_dataset import AIGPygDataset
    print("Successfully imported AIGPygDataset.")
except ImportError as e:
     print(f"Error importing AIGPygDataset: {e}")
     print("Ensure G2PT/datasets/aig_dataset.py exists and G2PT is in the Python path.")
     sys.exit(1)
# --- Removed AIGPygDataModule import ---


# --- Determine Input Data Directory (where processed/ files are) ---
try:
    script_dir = os.path.dirname(os.path.realpath(__file__)) # G2PT/datasets
    g2pt_root = os.path.dirname(script_dir) # G2PT/
    # This is the directory containing raw/ and processed/ subdirectories
    PYG_DATA_ROOT_DIR = os.path.abspath(os.path.join(g2pt_root, os.path.normpath(aig_cfg.data_dir)))
    print(f"Determined PyG Data Root Directory (expecting processed/ files inside): {PYG_DATA_ROOT_DIR}")
    # Check for processed dir, as AIGPygDataset loads from there
    if not os.path.isdir(os.path.join(PYG_DATA_ROOT_DIR, 'processed')):
         print(f"Warning: 'processed' subdirectory not found inside {PYG_DATA_ROOT_DIR}. AIGPygDataset initialization might fail.")
except AttributeError:
     print("Error: Missing 'data_dir' attribute in imported G2PT.configs.aig.")
     sys.exit(1)
except Exception as e:
     print(f"Error determining PyG data root directory: {e}")
     sys.exit(1)


# --- Define Final Output Directory for .bin files ---
FINAL_OUTPUT_DIR = "/Users/bellavg/aig-gen/G2PT/datasets/aig" # Your specified final location

# --- Main Processing Logic ---
if __name__ == '__main__':
    print(f"\n--- Stage 2: AIG PyG Processed Data to final .bin Conversion ---")
    print(f"--- Processing ONLY train and validation splits ---")
    print(f"Using Configuration from G2PT.configs.aig (for vocab/padding)")
    print(f"Input PyG Data Root Dir: {PYG_DATA_ROOT_DIR}") # Directory containing processed/
    print(f"Final Output Directory for .bin files: {FINAL_OUTPUT_DIR}")

    # --- REMOVED DataModule Initialization ---

    # --- Directly Instantiate AIGPygDataset for train and val ---
    print("\nInitializing datasets directly...")
    dataset_split = {}
    try:
        # Instantiate dataset for 'train' split
        # This will load processed/aig_processed_train.pt if it exists,
        # or trigger AIGPygDataset.process() if it doesn't (which reads raw/train.pt)
        train_dataset = AIGPygDataset(split='train', root=PYG_DATA_ROOT_DIR)
        dataset_split['train'] = train_dataset # Use 'train' key for internal processing loop

        # Instantiate dataset for 'val' split
        # This will load processed/aig_processed_val.pt if it exists,
        # or trigger AIGPygDataset.process() if it doesn't (which reads raw/val.pt)
        val_dataset = AIGPygDataset(split='val', root=PYG_DATA_ROOT_DIR)
        # Use 'eval' key to match the original example script's output naming convention
        dataset_split['val'] = val_dataset

        print(f"Directly initialized datasets: Train={len(dataset_split.get('train',[]))}, Eval={len(dataset_split.get('eval',[]))}")

    except FileNotFoundError as e:
         # This error would now come from AIGPygDataset if raw or processed files are missing
         print(f"\n*** File Not Found Error during AIGPygDataset initialization: {e} ***")
         print(f"Please ensure the necessary raw (.pt) or processed (aig_processed_*.pt) files exist in the correct subdirectories within '{PYG_DATA_ROOT_DIR}'.")
         sys.exit(1)
    except Exception as e:
        print(f"Failed to initialize AIGPygDataset directly: {e}")
        sys.exit(1)
    # --- Datasets loaded directly ---


    data_meta = {} # Dictionary to store final shapes for metadata file

    # Ensure final output directory exists
    try:
        os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)
        print(f"Ensured final output directory exists: {FINAL_OUTPUT_DIR}")
    except OSError as e:
         print(f"Error creating final output directory {FINAL_OUTPUT_DIR}: {e}")
         sys.exit(1)

    print("\nProcessing PyG datasets into final .bin format...")
    # Iterate through the 'train' and 'eval' splits loaded directly
    # Note: split_name_output is now the key in dataset_split ('train' or 'eval')
    for split_name_output, pyg_dataset in dataset_split.items():

        # Check if the dataset for the split is empty
        if not pyg_dataset or len(pyg_dataset) == 0:
            print(f"Skipping empty split: {split_name_output}")
            # Define empty shapes for metadata if split is empty
            data_meta[f'{split_name_output}_shape'] = {
                'xs': [0, 0], 'edge_indices': [0, 0, 0], 'edge_attrs': [0, 0],
                'num_inputs': [0], 'num_outputs': [0]
            }
            continue

        # Define and create the output directory for the current split's .bin files
        split_output_dir = os.path.join(FINAL_OUTPUT_DIR, split_name_output)
        try:
            os.makedirs(split_output_dir, exist_ok=True)
        except OSError as e:
             print(f"Error creating output directory {split_output_dir} for split {split_name_output}: {e}")
             print("Skipping saving for this split.")
             continue # Skip to the next split

        print(f"\nProcessing split: {split_name_output} ({len(pyg_dataset)} graphs)")

        # Lists to store data for the entire split before padding/saving
        xs_vocab_ids_list = []          # List of tensors [num_nodes]
        edge_indices_list = []        # List of tensors [num_edges, 2]
        edge_attrs_vocab_ids_list = []# List of tensors [num_edges]
        num_inputs_list = []          # List of ints
        num_outputs_list = []         # List of ints
        num_processed_graphs = 0      # Counter for valid graphs in this split

        # Iterate through PyG Data objects in the current split's dataset
        for data in tqdm(pyg_dataset, desc=f"  Converting {split_name_output} PyG data"):
            # --- Basic Checks & Attribute Fetching ---
            if data.x is None or data.x.numel() == 0:
                warnings.warn("Skipping graph: Missing or empty node features (data.x)")
                continue
            try:
                 current_num_inputs = int(data.num_inputs)
                 current_num_outputs = int(data.num_outputs)
            except (AttributeError, TypeError, ValueError) as e:
                 warnings.warn(f"Skipping graph: Missing or invalid 'num_inputs'/'num_outputs' attribute in Data object: {e}")
                 continue

            # --- Process Nodes: Convert one-hot features to vocab IDs ---
            valid_nodes = True
            try:
                node_feature_indices = data.x.argmax(dim=-1)
                node_ids = torch.tensor(
                    [aig_cfg.NODE_FEATURE_INDEX_TO_VOCAB.get(idx.item(), -1) for idx in node_feature_indices],
                    dtype=torch.long)
                if torch.any(node_ids == -1):
                    invalid_indices = node_feature_indices[node_ids == -1]
                    warnings.warn(f"Skipping graph: Found invalid node feature index during mapping: {invalid_indices.tolist()}. Check data.x and config.")
                    valid_nodes = False
            except Exception as e:
                warnings.warn(f"Skipping graph: Error processing node features: {e}")
                valid_nodes = False
            if not valid_nodes: continue

            # --- Process Edges: Convert one-hot features to vocab IDs and get indices ---
            valid_edges = True
            current_edge_ids = torch.tensor([], dtype=torch.long)
            current_edge_indices = torch.tensor([], dtype=torch.long).reshape(0,2)
            if data.edge_index is not None and data.edge_index.numel() > 0:
                if data.edge_attr is None or data.edge_attr.numel() == 0:
                     warnings.warn("Skipping graph: edge_index present but edge_attr missing or empty.")
                     valid_edges = False
                elif data.edge_attr.shape[0] != data.edge_index.shape[1]:
                     warnings.warn(f"Skipping graph: Mismatch between edge_index ({data.edge_index.shape[1]}) and edge_attr ({data.edge_attr.shape[0]}) counts.")
                     valid_edges = False
                else:
                    try:
                        edge_feature_indices = data.edge_attr.argmax(dim=-1)
                        edge_ids = torch.tensor(
                            [aig_cfg.EDGE_FEATURE_INDEX_TO_VOCAB.get(idx.item(), -1) for idx in edge_feature_indices],
                            dtype=torch.long)
                        if torch.any(edge_ids == -1):
                            invalid_indices = edge_feature_indices[edge_ids == -1]
                            warnings.warn(f"Skipping graph: Found invalid edge feature index during mapping: {invalid_indices.tolist()}. Check data.edge_attr and config.")
                            valid_edges = False
                        else:
                             current_edge_ids = edge_ids
                             current_edge_indices = data.edge_index.t() # Transpose to [num_edges, 2]
                    except Exception as e:
                        warnings.warn(f"Skipping graph: Error processing edge features/indices: {e}")
                        valid_edges = False
            if not valid_edges: continue

            # --- Append processed data to lists ---
            xs_vocab_ids_list.append(node_ids)
            edge_attrs_vocab_ids_list.append(current_edge_ids)
            edge_indices_list.append(current_edge_indices)
            num_inputs_list.append(current_num_inputs)
            num_outputs_list.append(current_num_outputs)
            num_processed_graphs += 1
        # --- End Graph Loop ---

        # --- Padding and Conversion to NumPy ---
        if num_processed_graphs == 0:
             print(f"Skipping saving for {split_name_output} as no valid graphs were processed.")
             data_meta[f'{split_name_output}_shape'] = {
                 'xs': [0, 0], 'edge_indices': [0, 0, 0], 'edge_attrs': [0, 0],
                 'num_inputs': [0], 'num_outputs': [0] }
             continue

        print(f"  Padding {split_name_output} data ({num_processed_graphs} graphs)...")
        try:
            padding_val_float = float(aig_cfg.PAD_VALUE)
            xs_padded = pad_sequence(xs_vocab_ids_list, batch_first=True, padding_value=padding_val_float)
            xs_np = xs_padded.numpy().astype(np.int16)
            edge_attrs_padded = pad_sequence(edge_attrs_vocab_ids_list, batch_first=True, padding_value=padding_val_float)
            edge_attrs_np = edge_attrs_padded.numpy().astype(np.int16)
            edge_indices_padded = pad_sequence(edge_indices_list, batch_first=True, padding_value=padding_val_float)
            edge_indices_np = edge_indices_padded.numpy().astype(np.int16).transpose(0, 2, 1) # Transpose to [B, 2, max_E]
            num_inputs_np = np.array(num_inputs_list, dtype=np.int16)
            num_outputs_np = np.array(num_outputs_list, dtype=np.int16)
        except Exception as e:
            print(f"Error during padding/conversion for split {split_name_output}: {e}")
            data_meta[f'{split_name_output}_shape'] = {
                'xs': [0, 0], 'edge_indices': [0, 0, 0], 'edge_attrs': [0, 0],
                'num_inputs': [0], 'num_outputs': [0] }
            continue

        print(f"    Final shapes - xs: {xs_np.shape}, edge_indices: {edge_indices_np.shape}, edge_attrs: {edge_attrs_np.shape}")
        print(f"    Final shapes - num_inputs: {num_inputs_np.shape}, num_outputs: {num_outputs_np.shape}")

        # --- Saving with Memmap ---
        print(f"  Saving final {split_name_output} data using memmap to {split_output_dir}...")
        xs_path = os.path.join(split_output_dir, 'xs.bin')
        edge_indices_path = os.path.join(split_output_dir, 'edge_indices.bin')
        edge_attrs_path = os.path.join(split_output_dir, 'edge_attrs.bin')
        num_inputs_path = os.path.join(split_output_dir, 'num_inputs.bin')
        num_outputs_path = os.path.join(split_output_dir, 'num_outputs.bin')
        try:
            xs_memmap = np.memmap(xs_path, dtype=np.int16, mode='w+', shape=xs_np.shape)
            xs_memmap[:] = xs_np; xs_memmap.flush(); del xs_memmap
            edge_indices_memmap = np.memmap(edge_indices_path, dtype=np.int16, mode='w+', shape=edge_indices_np.shape)
            edge_indices_memmap[:] = edge_indices_np; edge_indices_memmap.flush(); del edge_indices_memmap
            edge_attrs_memmap = np.memmap(edge_attrs_path, dtype=np.int16, mode='w+', shape=edge_attrs_np.shape)
            edge_attrs_memmap[:] = edge_attrs_np; edge_attrs_memmap.flush(); del edge_attrs_memmap
            num_inputs_memmap = np.memmap(num_inputs_path, dtype=np.int16, mode='w+', shape=num_inputs_np.shape)
            num_inputs_memmap[:] = num_inputs_np; num_inputs_memmap.flush(); del num_inputs_memmap
            num_outputs_memmap = np.memmap(num_outputs_path, dtype=np.int16, mode='w+', shape=num_outputs_np.shape)
            num_outputs_memmap[:] = num_outputs_np; num_outputs_memmap.flush(); del num_outputs_memmap
            print(f"    Saved final data bins to {split_output_dir}")
            data_meta[f'{split_name_output}_shape'] = {
                'xs': list(xs_np.shape), 'edge_indices': list(edge_indices_np.shape),
                'edge_attrs': list(edge_attrs_np.shape), 'num_inputs': list(num_inputs_np.shape),
                'num_outputs': list(num_outputs_np.shape) }
        except Exception as e:
            print(f"Error saving memmap files for split {split_name_output}: {e}")
            data_meta[f'{split_name_output}_shape'] = {
                'xs': [0, 0], 'edge_indices': [0, 0, 0], 'edge_attrs': [0, 0],
                'num_inputs': [0], 'num_outputs': [0] }

    # --- Save Metadata ---
    meta_path = os.path.join(FINAL_OUTPUT_DIR, 'data_meta.json')
    print(f"\nSaving final metadata to: {meta_path}")
    try:
        with open(meta_path, 'w') as f:
            json.dump(data_meta, f, indent=2)
    except Exception as e:
        print(f"Error saving metadata file {meta_path}: {e}")

    print("\n--- Stage 2: Final dataset preparation (.bin files incl. counts) finished. ---")
    print(f"Final data location: {FINAL_OUTPUT_DIR}")
