# G2PT/datasets/prepare_aig_final.py
# Stage 2: Convert AIG PyG Data objects into final .bin format with vocab IDs.
# Based on prepare_tree.py

from types import SimpleNamespace
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import os
import numpy as np
import json
import torch
import warnings

# Import your new PyG DataModule
from aig_dataset import *


# --- Configuration ---
CFG = SimpleNamespace(
    # --- Dataset loading config ---
    dataset=SimpleNamespace(
        name='aig',                   # Keep name 'aig' for consistency downstream
        datadir='./aig/',         # <--- POINT THIS to your PyG dataset dir (contains raw/processed)
        filter=False,                 # Filtering should be done before this stage
    ),
    # --- Dataloader config (used by DataModule init) ---
    train=SimpleNamespace(
        batch_size=32,                # Not critical for this script, but needed by DataModule
        num_workers=0,                # Use 0 for sequential processing in this script
    ),
    # --- General processing config ---
    general=SimpleNamespace(
        name='aig',
        padding_value=-100,          # Padding value for .bin files
        # --- Target Vocab IDs (MUST MATCH vocab.json) ---
        node_const0_id=97,
        node_pi_id=98,
        node_and_id=99,
        node_po_id=100,
        edge_inv_id=101,
        edge_reg_id=102,
    ),
    # --- Mapping from PyG feature index (from argmax) back to Vocab ID ---
    # Assumes one-hot encoding used in prepare_aig_pyg.py stage:
    # Index 0 -> CONST0, Index 1 -> PI, etc.
    node_feature_index_to_id = {
        0: 97, # CONST0 Feature Index 0 maps to Vocab ID 97
        1: 98, # PI Feature Index 1 maps to Vocab ID 98
        2: 99, # AND Feature Index 2 maps to Vocab ID 99
        3: 100 # PO Feature Index 3 maps to Vocab ID 100
    },
    # Index 0 -> INV, Index 1 -> REG
    edge_feature_index_to_id = {
        0: 101, # INV Feature Index 0 maps to Vocab ID 101
        1: 102  # REG Feature Index 1 maps to Vocab ID 102
    },
    # --- Final output directory for .bin files ---
    final_output_dir='./aig/' # <--- SET THIS: Where G2PT expects the final dataset (e.g., ./aig/)
                                # Relative to the script location (datasets/)
)

# --- Main Processing Logic ---
if __name__ == '__main__':
    print(f"--- Stage 2: AIG PyG .pt to final .bin Conversion ---")
    print(f"Loading PyG data using AIGPygDataModule from: {CFG.dataset.datadir}")

    # Instantiate your AIG PyG DataModule
    try:
        datamodule = AIGPygDataModule(CFG)
    except Exception as e:
        print(f"Failed to initialize AIGPygDataModule: {e}")
        print("Check if the paths are correct and if the processed PyG files exist.")
        exit(1)
    print("DataModule initialized.")

    # Access the underlying PyG datasets for each split
    # The DataModule ensures they are loaded/processed
    dataset_split = {}
    try:
         dataset_split['train'] = datamodule.train_dataset
         dataset_split['eval'] = datamodule.val_dataset # Use 'eval' key for consistency with G2PT
         dataset_split['test'] = datamodule.test_dataset
    except AttributeError:
         print("Error accessing datasets from DataModule. Check DataModule implementation.")
         exit(1)

    data_meta = {} # To store final shapes for data_meta.json

    # Ensure final output directory exists
    os.makedirs(CFG.final_output_dir, exist_ok=True)
    print(f"Output directory for .bin files: {CFG.final_output_dir}")

    print("Processing PyG datasets into final .bin format...")
    for split_name_internal, pyg_dataset in dataset_split.items():
        # Map internal split names ('val') to G2PT's expected dir names ('eval')
        split_name_output = 'eval' if split_name_internal == 'val' else split_name_internal

        if not pyg_dataset or len(pyg_dataset) == 0:
            print(f"Skipping empty split: {split_name_internal}")
            # Create metadata entry even for empty splits if needed downstream
            data_meta[f'{split_name_output}_shape'] = {'xs': [0, 0], 'edge_indices': [0, 0, 0], 'edge_attrs': [0, 0]}
            continue

        split_output_dir = os.path.join(CFG.final_output_dir, split_name_output)
        os.makedirs(split_output_dir, exist_ok=True)
        print(f"\nProcessing split: {split_name_internal} -> {split_name_output} ({len(pyg_dataset)} graphs)")

        xs_vocab_ids = []           # List to store tensors of node vocab IDs
        edge_indices_list = []      # List to store edge_index tensors [num_edges, 2]
        edge_attrs_vocab_ids = []   # List to store tensors of edge vocab IDs

        # Iterate through PyG Data objects from the loaded dataset
        for data in tqdm(pyg_dataset, desc=f"  Converting {split_name_internal} PyG data"):
            # 1. Convert node features (e.g., one-hot) back to integer vocab IDs
            if data.x is None or data.x.numel() == 0:
                warnings.warn("Skipping graph with missing or empty node features (data.x)")
                continue
            try:
                # Get index of '1' in one-hot (or class index directly if not one-hot)
                node_feature_indices = data.x.argmax(dim=-1)
                # Map feature index to vocabulary ID using the config mapping
                node_ids = torch.tensor(
                    [CFG.node_feature_index_to_id.get(idx.item(), -1) # Default to -1 for unknown
                     for idx in node_feature_indices],
                    dtype=torch.long
                )
                # Check if any node mapping failed
                if torch.any(node_ids == -1):
                    warnings.warn(f"Skipping graph: Found unknown node feature index during mapping.")
                    continue # Skip this graph
                xs_vocab_ids.append(node_ids) # Append tensor of vocab IDs
            except Exception as e:
                warnings.warn(f"Skipping graph: Error processing node features: {e}")
                continue

            # 2. Convert edge features back to integer vocab IDs & prepare edge_index
            if data.edge_index is not None and data.edge_index.numel() > 0:
                if data.edge_attr is None or data.edge_attr.numel() == 0:
                     warnings.warn("Skipping graph: edge_index present but edge_attr missing or empty.")
                     xs_vocab_ids.pop() # Remove corresponding node data
                     continue
                try:
                    edge_feature_indices = data.edge_attr.argmax(dim=-1)
                    edge_ids = torch.tensor(
                        [CFG.edge_feature_index_to_id.get(idx.item(), -1) # Default -1
                         for idx in edge_feature_indices],
                        dtype=torch.long
                    )
                    if torch.any(edge_ids == -1):
                        warnings.warn(f"Skipping graph: Found unknown edge feature index during mapping.")
                        xs_vocab_ids.pop() # Remove corresponding node data
                        continue
                    edge_attrs_vocab_ids.append(edge_ids) # Append tensor of vocab IDs
                    # Append edge_index (transpose required for padding later)
                    edge_indices_list.append(data.edge_index.t()) # Shape [num_edges, 2]

                except Exception as e:
                    warnings.warn(f"Skipping graph: Error processing edge features/indices: {e}")
                    xs_vocab_ids.pop() # Remove corresponding node data
                    continue
            else: # No edges in this graph
                 edge_attrs_vocab_ids.append(torch.tensor([], dtype=torch.long))
                 edge_indices_list.append(torch.tensor([], dtype=torch.long).reshape(0,2)) # Shape [0, 2]


        # --- Padding (convert lists of tensors to padded numpy arrays) ---
        if not xs_vocab_ids:
             print(f"Skipping saving for {split_name_output} as no valid graphs were processed.")
             data_meta[f'{split_name_output}_shape'] = {'xs': [0, 0], 'edge_indices': [0, 0, 0], 'edge_attrs': [0, 0]}
             continue # Skip to next split

        print(f"  Padding {split_name_output} data ({len(xs_vocab_ids)} graphs)...")
        try:
            # Pad node vocab IDs
            xs_padded = pad_sequence(xs_vocab_ids, batch_first=True, padding_value=float(CFG.general.padding_value))
            xs_np = xs_padded.numpy().astype(np.int16)

            # Pad edge vocab IDs
            edge_attrs_padded = pad_sequence(edge_attrs_vocab_ids, batch_first=True, padding_value=float(CFG.general.padding_value))
            edge_attrs_np = edge_attrs_padded.numpy().astype(np.int16)

            # Pad edge_index list (elements are shape [num_edges, 2])
            edge_indices_padded = pad_sequence(edge_indices_list, batch_first=True, padding_value=float(CFG.general.padding_value)) # Pads to [N, max_edges, 2]
            # Transpose final array to match expected [N, 2, max_edges] format
            edge_indices_np = edge_indices_padded.numpy().astype(np.int16).transpose(0, 2, 1)

        except Exception as e:
            print(f"Error during padding for split {split_name_output}: {e}")
            # Avoid saving potentially corrupted data for this split
            data_meta[f'{split_name_output}_shape'] = {'xs': [0, 0], 'edge_indices': [0, 0, 0], 'edge_attrs': [0, 0]}
            continue # Skip to next split

        print(f"    Final shapes - xs: {xs_np.shape}, edge_indices: {edge_indices_np.shape}, edge_attrs: {edge_attrs_np.shape}")

        # --- Saving with Memmap ---
        print(f"  Saving final {split_name_output} data using memmap to {split_output_dir}...")
        xs_path = os.path.join(split_output_dir, 'xs.bin')
        edge_indices_path = os.path.join(split_output_dir, 'edge_indices.bin') # Plural key
        edge_attrs_path = os.path.join(split_output_dir, 'edge_attrs.bin')     # Plural key

        try:
            # Save node data
            xs_memmap = np.memmap(xs_path, dtype=np.int16, mode='w+', shape=xs_np.shape)
            xs_memmap[:] = xs_np
            xs_memmap.flush(); del xs_memmap # Flush and close

            # Save edge index data
            edge_indices_memmap = np.memmap(edge_indices_path, dtype=np.int16, mode='w+', shape=edge_indices_np.shape)
            edge_indices_memmap[:] = edge_indices_np
            edge_indices_memmap.flush(); del edge_indices_memmap

            # Save edge attribute data
            edge_attrs_memmap = np.memmap(edge_attrs_path, dtype=np.int16, mode='w+', shape=edge_attrs_np.shape)
            edge_attrs_memmap[:] = edge_attrs_np
            edge_attrs_memmap.flush(); del edge_attrs_memmap

            print(f"    Saved final data bins to {split_output_dir}")

            # Store shapes in metadata (use output split name)
            data_meta[f'{split_name_output}_shape'] = {
                'xs': list(xs_np.shape),
                'edge_indices': list(edge_indices_np.shape), # Use plural key
                'edge_attrs': list(edge_attrs_np.shape)      # Use plural key
            }
        except Exception as e:
            print(f"Error saving memmap files for split {split_name_output}: {e}")
            # Set empty shapes in metadata if saving failed
            data_meta[f'{split_name_output}_shape'] = {'xs': [0, 0], 'edge_indices': [0, 0, 0], 'edge_attrs': [0, 0]}


    # --- Save Metadata ---
    meta_path = os.path.join(CFG.final_output_dir, 'data_meta.json')
    print(f"\nSaving final metadata to: {meta_path}")
    try:
        with open(meta_path, 'w') as f:
            json.dump(data_meta, f, indent=2)
    except Exception as e:
        print(f"Error saving metadata: {e}")

    print("\n--- Stage 2: Final dataset preparation (.bin files) finished. ---")
    print(f"Final data location: {CFG.final_output_dir}")