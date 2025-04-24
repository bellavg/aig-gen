# G2PT/datasets/prepare_aig_final_with_counts.py # <-- New Name Suggested
# Stage 2: Convert AIG PyG Data objects into final .bin format.
# Saves node/edge vocab IDs, edge indices, AND num_inputs/num_outputs.

from types import SimpleNamespace
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import os
import numpy as np
import json
import torch
import warnings

# Import your new PyG DataModule
from aig_dataset import * # Assumes this loads PyG Data with num_inputs/num_outputs

# TODO: Vocabulary loading for count->token mapping will happen in datasets_utils.py

# --- Configuration ---
CFG = SimpleNamespace(
    # --- Dataset loading config ---
    dataset=SimpleNamespace(
        name='aig',
        datadir='./real_aigs/', # <--- POINT THIS to your PyG dataset dir (contains raw/processed)
        filter=False,
        # Tokenizer path not strictly needed here, but maybe for consistency
        tokenizer_path='../tokenizers/aig'
    ),
    # --- Dataloader config (used by DataModule init) ---
    train=SimpleNamespace(
        batch_size=32,
        num_workers=0,
    ),
    # --- General processing config ---
    general=SimpleNamespace(
        name='aig',
        padding_value=-100,
    ),
    # --- Mapping from PyG feature index back to Vocab ID ---
    # Should match your latest vocabulary
    node_feature_index_to_id = {
        0: 71, # CONST0
        1: 72, # PI
        2: 73, # AND
        3: 74  # PO
    },
    edge_feature_index_to_id = {
        0: 75, # INV
        1: 76  # REG
    },
    # --- Final output directory for .bin files ---
    final_output_dir='./aig_final_data/' # <--- SUGGESTED new distinct output dir
                                    # Relative to the script location (datasets/)
)

# --- Main Processing Logic ---
if __name__ == '__main__':
    print(f"--- Stage 2: AIG PyG .pt to final .bin Conversion (incl. Counts) ---")
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
    dataset_split = {}
    try:
         dataset_split['train'] = datamodule.train_dataset
         dataset_split['eval'] = datamodule.val_dataset
         dataset_split['test'] = datamodule.test_dataset
    except AttributeError:
         print("Error accessing datasets from DataModule. Check DataModule implementation.")
         exit(1)

    data_meta = {} # To store final shapes

    # Ensure final output directory exists
    os.makedirs(CFG.final_output_dir, exist_ok=True)
    print(f"Output directory for final .bin files: {CFG.final_output_dir}")

    print("Processing PyG datasets into final .bin format...")
    for split_name_internal, pyg_dataset in dataset_split.items():
        split_name_output = 'eval' if split_name_internal == 'val' else split_name_internal

        if not pyg_dataset or len(pyg_dataset) == 0:
            print(f"Skipping empty split: {split_name_internal}")
            # Define empty shapes for metadata
            data_meta[f'{split_name_output}_shape'] = {
                'xs': [0, 0], 'edge_indices': [0, 0, 0], 'edge_attrs': [0, 0],
                'num_inputs': [0], 'num_outputs': [0] # Add shapes for new arrays
            }
            continue

        split_output_dir = os.path.join(CFG.final_output_dir, split_name_output)
        os.makedirs(split_output_dir, exist_ok=True)
        print(f"\nProcessing split: {split_name_internal} -> {split_name_output} ({len(pyg_dataset)} graphs)")

        # Lists to store data for the entire split before padding/saving
        xs_vocab_ids_list = []
        edge_indices_list = []
        edge_attrs_vocab_ids_list = []
        num_inputs_list = [] # <<<--- ADDED LIST
        num_outputs_list = [] # <<<--- ADDED LIST
        num_processed_graphs = 0

        # Iterate through PyG Data objects
        for data in tqdm(pyg_dataset, desc=f"  Converting {split_name_internal} PyG data"):
            # --- Basic Checks & Attribute Fetching ---
            if data.x is None or data.x.numel() == 0:
                warnings.warn("Skipping graph: Missing or empty node features (data.x)")
                continue

            # <<<--- FETCH num_inputs and num_outputs --- >>>
            try:
                 current_num_inputs = data.num_inputs
                 current_num_outputs = data.num_outputs
                 if current_num_inputs is None or current_num_outputs is None:
                      raise AttributeError("num_inputs or num_outputs is None")
                 # Convert to int if they are tensors or other types
                 current_num_inputs = int(current_num_inputs)
                 current_num_outputs = int(current_num_outputs)
            except AttributeError as e:
                 warnings.warn(f"Skipping graph: Missing 'num_inputs' or 'num_outputs' attribute: {e}")
                 continue
            except (TypeError, ValueError) as e:
                 warnings.warn(f"Skipping graph: Invalid type or value for 'num_inputs'/'num_outputs': {e}")
                 continue
            # <<<--------------------------------------- >>>

            # --- Process Nodes ---
            valid_nodes = True
            try:
                node_feature_indices = data.x.argmax(dim=-1)
                node_ids = torch.tensor(
                    [CFG.node_feature_index_to_id.get(idx.item(), -1)
                     for idx in node_feature_indices],
                    dtype=torch.long
                )
                if torch.any(node_ids == -1):
                    warnings.warn(f"Skipping graph: Found unknown node feature index during mapping.")
                    valid_nodes = False
            except Exception as e:
                warnings.warn(f"Skipping graph: Error processing node features: {e}")
                valid_nodes = False
            if not valid_nodes: continue

            # --- Process Edges ---
            valid_edges = True
            current_edge_ids = torch.tensor([], dtype=torch.long)
            current_edge_indices = torch.tensor([], dtype=torch.long).reshape(0,2)
            if data.edge_index is not None and data.edge_index.numel() > 0:
                if data.edge_attr is None or data.edge_attr.numel() == 0:
                     warnings.warn("Skipping graph: edge_index present but edge_attr missing or empty.")
                     valid_edges = False
                else:
                    try:
                        edge_feature_indices = data.edge_attr.argmax(dim=-1)
                        edge_ids = torch.tensor(
                            [CFG.edge_feature_index_to_id.get(idx.item(), -1)
                             for idx in edge_feature_indices],
                            dtype=torch.long
                        )
                        if torch.any(edge_ids == -1):
                            warnings.warn(f"Skipping graph: Found unknown edge feature index during mapping.")
                            valid_edges = False
                        else:
                             current_edge_ids = edge_ids
                             current_edge_indices = data.edge_index.t() # Shape [num_edges, 2]
                    except Exception as e:
                        warnings.warn(f"Skipping graph: Error processing edge features/indices: {e}")
                        valid_edges = False
            if not valid_edges: continue

            # --- If all checks passed, append data to lists ---
            xs_vocab_ids_list.append(node_ids)
            edge_attrs_vocab_ids_list.append(current_edge_ids)
            edge_indices_list.append(current_edge_indices)
            num_inputs_list.append(current_num_inputs)   # <<<--- APPEND COUNT
            num_outputs_list.append(current_num_outputs) # <<<--- APPEND COUNT
            num_processed_graphs += 1
        # --- End Graph Loop ---

        # --- Padding ---
        if num_processed_graphs == 0:
             print(f"Skipping saving for {split_name_output} as no valid graphs were processed.")
             data_meta[f'{split_name_output}_shape'] = {
                 'xs': [0, 0], 'edge_indices': [0, 0, 0], 'edge_attrs': [0, 0],
                 'num_inputs': [0], 'num_outputs': [0] # Add shapes for new arrays
                 }
             continue

        print(f"  Padding {split_name_output} data ({num_processed_graphs} graphs)...")
        try:
            xs_padded = pad_sequence(xs_vocab_ids_list, batch_first=True, padding_value=float(CFG.general.padding_value))
            xs_np = xs_padded.numpy().astype(np.int16)

            edge_attrs_padded = pad_sequence(edge_attrs_vocab_ids_list, batch_first=True, padding_value=float(CFG.general.padding_value))
            edge_attrs_np = edge_attrs_padded.numpy().astype(np.int16)

            edge_indices_padded = pad_sequence(edge_indices_list, batch_first=True, padding_value=float(CFG.general.padding_value))
            edge_indices_np = edge_indices_padded.numpy().astype(np.int16).transpose(0, 2, 1)

            # Convert count lists to numpy arrays (no padding needed)
            num_inputs_np = np.array(num_inputs_list, dtype=np.int16) # <<<--- CONVERT COUNT
            num_outputs_np = np.array(num_outputs_list, dtype=np.int16)# <<<--- CONVERT COUNT

        except Exception as e:
            print(f"Error during padding/conversion for split {split_name_output}: {e}")
            data_meta[f'{split_name_output}_shape'] = {
                'xs': [0, 0], 'edge_indices': [0, 0, 0], 'edge_attrs': [0, 0],
                'num_inputs': [0], 'num_outputs': [0]
                }
            continue

        print(f"    Final shapes - xs: {xs_np.shape}, edge_indices: {edge_indices_np.shape}, edge_attrs: {edge_attrs_np.shape}")
        print(f"    Final shapes - num_inputs: {num_inputs_np.shape}, num_outputs: {num_outputs_np.shape}") # <<<--- PRINT SHAPES


        # --- Saving with Memmap ---
        print(f"  Saving final {split_name_output} data using memmap to {split_output_dir}...")
        xs_path = os.path.join(split_output_dir, 'xs.bin')
        edge_indices_path = os.path.join(split_output_dir, 'edge_indices.bin')
        edge_attrs_path = os.path.join(split_output_dir, 'edge_attrs.bin')
        num_inputs_path = os.path.join(split_output_dir, 'num_inputs.bin') # <<<--- NEW PATH
        num_outputs_path = os.path.join(split_output_dir, 'num_outputs.bin')# <<<--- NEW PATH

        try:
            # Save node data
            xs_memmap = np.memmap(xs_path, dtype=np.int16, mode='w+', shape=xs_np.shape)
            xs_memmap[:] = xs_np; xs_memmap.flush(); del xs_memmap

            # Save edge index data
            edge_indices_memmap = np.memmap(edge_indices_path, dtype=np.int16, mode='w+', shape=edge_indices_np.shape)
            edge_indices_memmap[:] = edge_indices_np; edge_indices_memmap.flush(); del edge_indices_memmap

            # Save edge attribute data
            edge_attrs_memmap = np.memmap(edge_attrs_path, dtype=np.int16, mode='w+', shape=edge_attrs_np.shape)
            edge_attrs_memmap[:] = edge_attrs_np; edge_attrs_memmap.flush(); del edge_attrs_memmap

            # <<<--- SAVE COUNTS --- >>>
            num_inputs_memmap = np.memmap(num_inputs_path, dtype=np.int16, mode='w+', shape=num_inputs_np.shape)
            num_inputs_memmap[:] = num_inputs_np; num_inputs_memmap.flush(); del num_inputs_memmap

            num_outputs_memmap = np.memmap(num_outputs_path, dtype=np.int16, mode='w+', shape=num_outputs_np.shape)
            num_outputs_memmap[:] = num_outputs_np; num_outputs_memmap.flush(); del num_outputs_memmap
            # <<<-------------------- >>>

            print(f"    Saved final data bins to {split_output_dir}")

            # Store shapes in metadata
            data_meta[f'{split_name_output}_shape'] = {
                'xs': list(xs_np.shape),
                'edge_indices': list(edge_indices_np.shape),
                'edge_attrs': list(edge_attrs_np.shape),
                'num_inputs': list(num_inputs_np.shape),   # <<<--- ADD SHAPE
                'num_outputs': list(num_outputs_np.shape) # <<<--- ADD SHAPE
            }
        except Exception as e:
            print(f"Error saving memmap files for split {split_name_output}: {e}")
            data_meta[f'{split_name_output}_shape'] = {
                'xs': [0, 0], 'edge_indices': [0, 0, 0], 'edge_attrs': [0, 0],
                'num_inputs': [0], 'num_outputs': [0]
            }

    # --- Save Metadata ---
    meta_path = os.path.join(CFG.final_output_dir, 'data_meta.json')
    print(f"\nSaving final metadata to: {meta_path}")
    try:
        with open(meta_path, 'w') as f:
            json.dump(data_meta, f, indent=2)
    except Exception as e:
        print(f"Error saving metadata: {e}")

    print("\n--- Stage 2: Final dataset preparation (.bin files incl. counts) finished. ---")
    print(f"Final data location: {CFG.final_output_dir}")
    print("Next step: Modify datasets_utils.py (NumpyBinDataset and seq generation) to load counts and prepend context tokens.")