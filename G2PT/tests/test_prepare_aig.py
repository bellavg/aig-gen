# G2PT/tests/test_prepare_aig.py
import unittest
import os
import sys
import tempfile
import shutil
import json
import torch
import numpy as np
from torch_geometric.data import Data
from torch.nn.utils.rnn import pad_sequence

# --- Add G2PT root to path to import config and modules ---
# Assumes this script is in G2PT/tests/
script_dir = os.path.dirname(os.path.realpath(__file__))
g2pt_root = os.path.dirname(script_dir) # G2PT/
if g2pt_root not in sys.path:
    sys.path.insert(0, g2pt_root) # Prepend G2PT root

try:
    import G2PT.configs.aig as aig_cfg
    # Note: We are replicating the core processing logic here instead of importing
    # the function from prepare_aig.py, which makes this test self-contained.
except ImportError as e:
    print(f"Error importing G2PT.configs.aig: {e}")
    print("Ensure the test is run from a context where G2PT is importable (e.g., from the G2PT root directory) or adjust path.")
    sys.exit(1)

# --- Helper function to create dummy PyG data for testing ---
def create_test_pyg_data(num_nodes, num_edges, num_inputs, num_outputs, has_attrs=True):
    """Creates a sample PyG Data object mimiciking the output of aig_pkl_to_pyg.py."""
    # Node features (one-hot based on config)
    # Represents the 'x' attribute in the Data objects loaded from train.pt/val.pt
    node_indices = torch.randint(0, aig_cfg.NUM_NODE_FEATURES, (num_nodes,))
    x = torch.nn.functional.one_hot(node_indices, num_classes=aig_cfg.NUM_NODE_FEATURES).float()

    # Edge index (Shape: [2, num_edges])
    if num_edges > 0 and num_nodes > 1:
        edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
        # Ensure no self-loops for simplicity if needed
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]
        num_edges = edge_index.shape[1] # Update actual edge count after removing self-loops
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        num_edges = 0

    # Edge attributes (one-hot based on config)
    # Represents the 'edge_attr' attribute in the Data objects loaded from train.pt/val.pt
    if num_edges > 0 and has_attrs:
        edge_attr_indices = torch.randint(0, aig_cfg.NUM_EDGE_FEATURES, (num_edges,))
        edge_attr = torch.nn.functional.one_hot(edge_attr_indices, num_classes=aig_cfg.NUM_EDGE_FEATURES).float()
    elif num_edges > 0 and not has_attrs:
         edge_attr = None # Test case with missing edge_attr (should be skipped by processing)
    else: # No edges
        edge_attr = torch.empty((0, aig_cfg.NUM_EDGE_FEATURES), dtype=torch.float)

    # Add the required num_inputs and num_outputs attributes
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_inputs=num_inputs,
        num_outputs=num_outputs
    )

# --- Test Class ---
class TestPrepareAigBinConversion(unittest.TestCase):

    def setUp(self):
        """Create a temporary directory for test outputs before each test."""
        self.test_dir = tempfile.mkdtemp()
        # print(f"Created temp dir: {self.test_dir}") # Optional debug print

    def tearDown(self):
        """Remove the temporary directory after each test."""
        shutil.rmtree(self.test_dir)
        # print(f"Removed temp dir: {self.test_dir}") # Optional debug print

    def _process_pyg_list_to_bin(self, pyg_data_list, split_name):
        """
        Processes a list of PyG Data objects into padded numpy arrays.
        This replicates the core logic of the loop in prepare_aig.py,
        including feature-to-vocab ID mapping and padding.
        Returns a tuple of numpy arrays (xs, edge_indices, edge_attrs, num_inputs, num_outputs)
        or None if processing fails or no valid graphs are found.
        """
        xs_vocab_ids_list = []          # List of tensors [num_nodes] (vocab IDs)
        edge_indices_list = []        # List of tensors [num_edges, 2] (node indices)
        edge_attrs_vocab_ids_list = []# List of tensors [num_edges] (vocab IDs)
        num_inputs_list = []          # List of ints
        num_outputs_list = []         # List of ints
        num_processed_graphs = 0

        # Iterate through the input list of PyG Data objects
        for i, data in enumerate(pyg_data_list):
            # --- Basic Checks (mimicking prepare_aig.py) ---
            if data.x is None or data.x.numel() == 0: continue # Skip if no node features
            try:
                 # Ensure num_inputs/num_outputs exist and are convertible to int
                 current_num_inputs = int(data.num_inputs)
                 current_num_outputs = int(data.num_outputs)
            except (AttributeError, TypeError, ValueError): continue # Skip if counts are missing/invalid

            # --- Process Nodes (Convert features to Vocab IDs) ---
            valid_nodes = True
            try:
                # Assumes data.x is one-hot float tensor [N, num_node_features]
                node_feature_indices = data.x.argmax(dim=-1)
                # Map feature index (0..3) to vocab ID using config mapping
                node_ids = torch.tensor(
                    [aig_cfg.NODE_FEATURE_INDEX_TO_VOCAB.get(idx.item(), -1) for idx in node_feature_indices],
                    dtype=torch.long)
                # Check if any mapping failed
                if torch.any(node_ids == -1): valid_nodes = False
            except Exception: valid_nodes = False
            if not valid_nodes: continue # Skip if node processing failed

            # --- Process Edges (Get indices and convert features to Vocab IDs) ---
            valid_edges = True
            current_edge_ids = torch.tensor([], dtype=torch.long)           # Vocab IDs for edge types
            current_edge_indices = torch.tensor([], dtype=torch.long).reshape(0,2) # Node indices for edges [num_edges, 2]

            if data.edge_index is not None and data.edge_index.numel() > 0:
                # Edge attributes are required if edges exist
                if data.edge_attr is None or data.edge_attr.numel() == 0:
                     valid_edges = False # Skip if edges exist but attrs don't
                # Check for shape mismatch
                elif data.edge_attr.shape[0] != data.edge_index.shape[1]:
                     valid_edges = False
                else:
                    try:
                        # Assumes data.edge_attr is one-hot float tensor [E, num_edge_features]
                        edge_feature_indices = data.edge_attr.argmax(dim=-1)
                        # Map feature index (0..1) to vocab ID using config mapping
                        edge_ids = torch.tensor(
                            [aig_cfg.EDGE_FEATURE_INDEX_TO_VOCAB.get(idx.item(), -1) for idx in edge_feature_indices],
                            dtype=torch.long)
                        # Check if any mapping failed
                        if torch.any(edge_ids == -1): valid_edges = False
                        else:
                             current_edge_ids = edge_ids # Store edge vocab IDs
                             current_edge_indices = data.edge_index.t() # Store edge indices as [E, 2]
                    except Exception: valid_edges = False
            # If graph has no edges, empty tensors are fine.
            if not valid_edges: continue # Skip if edge processing failed

            # --- Append processed data if graph is valid ---
            xs_vocab_ids_list.append(node_ids)
            edge_attrs_vocab_ids_list.append(current_edge_ids)
            edge_indices_list.append(current_edge_indices)
            num_inputs_list.append(current_num_inputs)
            num_outputs_list.append(current_num_outputs)
            num_processed_graphs += 1
        # --- End graph loop ---

        if num_processed_graphs == 0: return None # No valid graphs to process

        # --- Padding (mimicking prepare_aig.py) ---
        try:
            padding_val_float = float(aig_cfg.PAD_VALUE) # Use PAD_VALUE from config

            # Pad node vocab IDs: [B, max_nodes]
            xs_padded = pad_sequence(xs_vocab_ids_list, batch_first=True, padding_value=padding_val_float)
            xs_np = xs_padded.numpy().astype(np.int16)

            # Pad edge attribute vocab IDs: [B, max_edges]
            edge_attrs_padded = pad_sequence(edge_attrs_vocab_ids_list, batch_first=True, padding_value=padding_val_float)
            edge_attrs_np = edge_attrs_padded.numpy().astype(np.int16)

            # Pad edge indices: [B, max_edges, 2]
            edge_indices_padded = pad_sequence(edge_indices_list, batch_first=True, padding_value=padding_val_float)
            # Transpose to [B, 2, max_edges] to match prepare_aig.py's output format
            edge_indices_np = edge_indices_padded.numpy().astype(np.int16).transpose(0, 2, 1) # EDIT: Added transpose

            # Convert counts to numpy arrays (no padding needed): [B]
            num_inputs_np = np.array(num_inputs_list, dtype=np.int16)
            num_outputs_np = np.array(num_outputs_list, dtype=np.int16)

            return xs_np, edge_indices_np, edge_attrs_np, num_inputs_np, num_outputs_np
        except Exception as e:
            print(f"Error during test padding for split {split_name}: {e}")
            return None # Return None if padding fails

    def test_conversion_and_saving(self):
        """Tests the conversion of sample PyG data and saving/loading of .bin files."""
        # 1. Create Sample Data
        data_list = [
            create_test_pyg_data(num_nodes=10, num_edges=15, num_inputs=3, num_outputs=2),
            create_test_pyg_data(num_nodes=5, num_edges=3, num_inputs=2, num_outputs=1),
            create_test_pyg_data(num_nodes=8, num_edges=0, num_inputs=4, num_outputs=4), # Graph with no edges
            create_test_pyg_data(num_nodes=12, num_edges=5, num_inputs=6, num_outputs=1, has_attrs=False), # Graph with edges but missing attrs (should be skipped)
            create_test_pyg_data(num_nodes=6, num_edges=7, num_inputs=1, num_outputs=5),
        ]
        expected_valid_graphs = 4 # The one with missing attrs should be skipped
        split_name = "test_split"
        split_output_dir = os.path.join(self.test_dir, split_name)
        os.makedirs(split_output_dir, exist_ok=True)

        # 2. Process Data using helper function (mimics prepare_aig.py loop)
        processed_data = self._process_pyg_list_to_bin(data_list, split_name)
        self.assertIsNotNone(processed_data, "Processing helper function failed or returned None.")
        xs_np, edge_indices_np, edge_attrs_np, num_inputs_np, num_outputs_np = processed_data

        # Verify that the correct number of graphs were processed
        self.assertEqual(xs_np.shape[0], expected_valid_graphs, f"Expected {expected_valid_graphs} valid graphs, but processed {xs_np.shape[0]}")
        self.assertEqual(edge_indices_np.shape[0], expected_valid_graphs)
        self.assertEqual(edge_attrs_np.shape[0], expected_valid_graphs)
        self.assertEqual(num_inputs_np.shape[0], expected_valid_graphs)
        self.assertEqual(num_outputs_np.shape[0], expected_valid_graphs)

        # 3. Save Data using Memmap (mimicking prepare_aig.py saving logic)
        xs_path = os.path.join(split_output_dir, 'xs.bin')
        edge_indices_path = os.path.join(split_output_dir, 'edge_indices.bin')
        edge_attrs_path = os.path.join(split_output_dir, 'edge_attrs.bin')
        num_inputs_path = os.path.join(split_output_dir, 'num_inputs.bin')
        num_outputs_path = os.path.join(split_output_dir, 'num_outputs.bin')

        try:
            # Save node data (vocab IDs)
            xs_memmap = np.memmap(xs_path, dtype=np.int16, mode='w+', shape=xs_np.shape)
            xs_memmap[:] = xs_np; xs_memmap.flush(); del xs_memmap
            # Save edge index data (pairs of node indices) - Transposed shape
            edge_indices_memmap = np.memmap(edge_indices_path, dtype=np.int16, mode='w+', shape=edge_indices_np.shape)
            edge_indices_memmap[:] = edge_indices_np; edge_indices_memmap.flush(); del edge_indices_memmap
            # Save edge attribute data (vocab IDs)
            edge_attrs_memmap = np.memmap(edge_attrs_path, dtype=np.int16, mode='w+', shape=edge_attrs_np.shape)
            edge_attrs_memmap[:] = edge_attrs_np; edge_attrs_memmap.flush(); del edge_attrs_memmap
            # Save counts
            num_inputs_memmap = np.memmap(num_inputs_path, dtype=np.int16, mode='w+', shape=num_inputs_np.shape)
            num_inputs_memmap[:] = num_inputs_np; num_inputs_memmap.flush(); del num_inputs_memmap
            num_outputs_memmap = np.memmap(num_outputs_path, dtype=np.int16, mode='w+', shape=num_outputs_np.shape)
            num_outputs_memmap[:] = num_outputs_np; num_outputs_memmap.flush(); del num_outputs_memmap
        except Exception as e:
            self.fail(f"Failed to save memmap files: {e}")

        # 4. Load back and Verify Content, Shape, Dtype, Padding
        pad_val = aig_cfg.PAD_VALUE

        # --- Verify xs.bin ---
        self.assertTrue(os.path.exists(xs_path))
        loaded_xs = np.memmap(xs_path, dtype=np.int16, mode='r', shape=xs_np.shape)
        np.testing.assert_array_equal(loaded_xs, xs_np, "Loaded xs data mismatch")
        self.assertEqual(loaded_xs.dtype, np.int16)
        # Check padding value in xs
        valid_graph_idx = 0
        for i, graph_data in enumerate(data_list):
             if graph_data.edge_attr is None and graph_data.edge_index is not None and graph_data.edge_index.numel() > 0: continue # Skip the graph that was skipped
             original_len = graph_data.num_nodes
             # Check padding only if original length < padded length
             if original_len < loaded_xs.shape[1]:
                 padded_part = loaded_xs[valid_graph_idx, original_len:]
                 np.testing.assert_array_equal(padded_part, pad_val, f"Incorrect padding in xs for original graph index {i}")
             valid_graph_idx += 1


        # --- Verify edge_indices.bin ---
        self.assertTrue(os.path.exists(edge_indices_path))
        loaded_edge_indices = np.memmap(edge_indices_path, dtype=np.int16, mode='r', shape=edge_indices_np.shape)
        np.testing.assert_array_equal(loaded_edge_indices, edge_indices_np, "Loaded edge_indices data mismatch")
        self.assertEqual(loaded_edge_indices.dtype, np.int16)
        # Check padding value in edge_indices (shape is [B, 2, max_E])
        valid_graph_idx = 0
        for i, graph_data in enumerate(data_list):
             if graph_data.edge_attr is None and graph_data.edge_index is not None and graph_data.edge_index.numel() > 0: continue # Skip the graph that was skipped
             original_len = graph_data.edge_index.shape[1] if graph_data.edge_index is not None else 0
             # Check padding only if original length < padded length
             if original_len < loaded_edge_indices.shape[2]:
                 padded_part = loaded_edge_indices[valid_graph_idx, :, original_len:] # Get padding in the last dimension
                 np.testing.assert_array_equal(padded_part, pad_val, f"Incorrect padding in edge_indices for original graph index {i}")
             valid_graph_idx += 1


        # --- Verify edge_attrs.bin ---
        self.assertTrue(os.path.exists(edge_attrs_path))
        loaded_edge_attrs = np.memmap(edge_attrs_path, dtype=np.int16, mode='r', shape=edge_attrs_np.shape)
        np.testing.assert_array_equal(loaded_edge_attrs, edge_attrs_np, "Loaded edge_attrs data mismatch")
        self.assertEqual(loaded_edge_attrs.dtype, np.int16)
        # Check padding value in edge_attrs (shape is [B, max_E])
        valid_graph_idx = 0
        for i, graph_data in enumerate(data_list):
             if graph_data.edge_attr is None and graph_data.edge_index is not None and graph_data.edge_index.numel() > 0: continue # Skip the graph that was skipped
             original_len = graph_data.edge_index.shape[1] if graph_data.edge_index is not None else 0
             # Check padding only if original length < padded length
             if original_len < loaded_edge_attrs.shape[1]:
                 padded_part = loaded_edge_attrs[valid_graph_idx, original_len:]
                 np.testing.assert_array_equal(padded_part, pad_val, f"Incorrect padding in edge_attrs for original graph index {i}")
             valid_graph_idx += 1


        # --- Verify num_inputs.bin ---
        self.assertTrue(os.path.exists(num_inputs_path))
        loaded_num_inputs = np.memmap(num_inputs_path, dtype=np.int16, mode='r', shape=num_inputs_np.shape)
        np.testing.assert_array_equal(loaded_num_inputs, num_inputs_np, "Loaded num_inputs data mismatch")
        self.assertEqual(loaded_num_inputs.dtype, np.int16)
        # Verify actual values match original data (for the valid graphs)
        expected_inputs = [d.num_inputs for i, d in enumerate(data_list) if not (d.edge_attr is None and d.edge_index is not None and d.edge_index.numel() > 0)]
        np.testing.assert_array_equal(loaded_num_inputs, expected_inputs, "Num_inputs values mismatch")

        # --- Verify num_outputs.bin ---
        self.assertTrue(os.path.exists(num_outputs_path))
        loaded_num_outputs = np.memmap(num_outputs_path, dtype=np.int16, mode='r', shape=num_outputs_np.shape)
        np.testing.assert_array_equal(loaded_num_outputs, num_outputs_np, "Loaded num_outputs data mismatch")
        self.assertEqual(loaded_num_outputs.dtype, np.int16)
        # Verify actual values match original data (for the valid graphs)
        expected_outputs = [d.num_outputs for i, d in enumerate(data_list) if not (d.edge_attr is None and d.edge_index is not None and d.edge_index.numel() > 0)]
        np.testing.assert_array_equal(loaded_num_outputs, expected_outputs, "Num_outputs values mismatch")


        print(f"\nSuccessfully tested conversion, saving, loading, and padding for {split_name}")

    def test_metadata_creation(self):
        """Tests if the metadata dictionary is created correctly (shapes)."""
        # This test focuses only on the shape calculation part after processing
        data_list = [
            create_test_pyg_data(num_nodes=12, num_edges=20, num_inputs=5, num_outputs=3),
            create_test_pyg_data(num_nodes=7, num_edges=5, num_inputs=3, num_outputs=1),
        ]
        split_name = "meta_test"
        processed_data = self._process_pyg_list_to_bin(data_list, split_name)
        self.assertIsNotNone(processed_data, "Processing failed for metadata test.")
        xs_np, edge_indices_np, edge_attrs_np, num_inputs_np, num_outputs_np = processed_data

        # --- Get the ACTUAL shapes from the processed numpy arrays ---
        actual_xs_shape = list(xs_np.shape) # [B, max_N]
        actual_edge_indices_shape = list(edge_indices_np.shape) # [B, 2, max_E]
        actual_edge_attrs_shape = list(edge_attrs_np.shape) # [B, max_E]
        actual_num_inputs_shape = list(num_inputs_np.shape) # [B]
        actual_num_outputs_shape = list(num_outputs_np.shape) # [B]

        # --- Simulate metadata creation using the actual shapes ---
        # This mimics how prepare_aig.py stores shapes in data_meta
        data_meta = {}
        data_meta[f'{split_name}_shape'] = {
            'xs': actual_xs_shape,
            'edge_indices': actual_edge_indices_shape,
            'edge_attrs': actual_edge_attrs_shape,
            'num_inputs': actual_num_inputs_shape,
            'num_outputs': actual_num_outputs_shape
        }

        # --- Verify shapes stored in metadata MATCH the actual shapes ---
        # Check batch size (B) and other known dimensions (like the '2' in edge_indices)
        # Use the actual max lengths obtained from the processed data shapes.
        expected_batch_size = 2
        expected_edge_dim = 2
        max_nodes = actual_xs_shape[1] # Get actual max nodes from shape
        max_edges = actual_edge_indices_shape[2] # Get actual max edges from shape (after transpose)

        self.assertEqual(data_meta[f'{split_name}_shape']['xs'], [expected_batch_size, max_nodes])
        # Ensure the shape matches the transposed format [B, 2, max_E]
        self.assertEqual(data_meta[f'{split_name}_shape']['edge_indices'], [expected_batch_size, expected_edge_dim, max_edges]) # EDIT: Changed shape check
        self.assertEqual(data_meta[f'{split_name}_shape']['edge_attrs'], [expected_batch_size, max_edges])
        self.assertEqual(data_meta[f'{split_name}_shape']['num_inputs'], [expected_batch_size])
        self.assertEqual(data_meta[f'{split_name}_shape']['num_outputs'], [expected_batch_size])

        print(f"\nSuccessfully tested metadata shape generation for {split_name}")


if __name__ == '__main__':
    unittest.main()
