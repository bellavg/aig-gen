# test_evaluate_aigs.py
import unittest
import networkx as nx
import sys
import os
import torch
import numpy as np
import json
import logging
from collections import Counter

#PASSED

# --- Configure Test Environment ---

# Get the directory of the current test file
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
# Assume the G2PT project root is one level up from the 'tests' directory
PROJECT_ROOT = os.path.dirname(TEST_DIR)
# Add project root to sys.path to allow importing G2PT modules
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
print(f"Added '{PROJECT_ROOT}' to sys.path") # Confirm path addition

# Define paths for test data relative to the project root
TEST_DATA_DIR = "/Users/bellavg/aig-gen/G2PT/datasets/aig"
TEST_META_FILE = os.path.join(TEST_DATA_DIR, "data_meta.json") # Meta file for bin test data
TEST_SPLIT_NAME = 'val' # Define the expected split name

# Check if test data exists, needed for skipping tests
pyg_data_exists = os.path.isdir(os.path.join(TEST_DATA_DIR, "processed")) and \
                  os.path.exists(os.path.join(TEST_DATA_DIR, "processed", f"aig_processed_{TEST_SPLIT_NAME}.pt")) # Use processed filename convention
bin_data_exists = os.path.isdir(os.path.join(TEST_DATA_DIR, TEST_SPLIT_NAME)) and \
                  os.path.exists(os.path.join(TEST_DATA_DIR, TEST_SPLIT_NAME, "xs.bin")) and \
                  os.path.exists(TEST_META_FILE)


# --- Attempt to Import Required Modules ---
from ..evaluate_aigs import (
    calculate_structural_aig_metrics,
    count_pi_po_paths
)
from ..configs.aig import (NODE_TYPE_KEYS, EDGE_TYPE_KEYS,
                           MIN_AND_COUNT, MIN_PO_COUNT, NODE_TYPE_VOCAB)

# Assuming validate_input_data.py is in G2PT root
from .graph_utils import pyg_data_to_nx, bin_data_to_nx
print("Imported from validate_input_data successfully.")

# Assuming aig_dataset.py is in G2PT/datasets/
from ..datasets.aig_dataset import AIGPygDataset
print("Imported AIGPygDataset successfully.")

from ..datasets_utils import NumpyBinDataset

# === REPLACE with ===
NODE_PI = "NODE_PI"
NODE_AND = "NODE_AND"
NODE_PO = "NODE_PO"
NODE_CONST0 = "NODE_CONST0" # Define it consistently as a string
# Define edge types if needed for clarity, though your helper uses the string directly
EDGE_REG = "EDGE_REG"
EDGE_INV = "EDGE_INV"


# --- Unit Test Classes ---


class TestEvaluateAIGs(unittest.TestCase):
    """Contains unit tests for the core evaluation logic using manually created graphs."""

    def setUp(self):
        """Executed before each test method in this class."""
        pass # No specific setup needed

    # --- Helper Methods (Use constants if evaluate_aigs loaded) ---
    def _add_node(self, G, node_id, node_type):
        # Use actual constants if loaded, otherwise use placeholder strings
        node_type_to_use = node_type
        G.add_node(node_id, type=node_type_to_use)

    def _add_edge(self, G, u, v, edge_type="EDGE_REG"):
        G.add_edge(u, v, type=edge_type)

    # === Structural Validity Tests ===
    # (Keep all the original tests here - they will use dummy types if needed)
    def test_empty_graph(self):
        G = nx.DiGraph(); metrics = calculate_structural_aig_metrics(G)
        self.assertFalse(metrics['is_structurally_valid'])
        self.assertIn("Empty or Invalid Graph Object", metrics['constraints_failed'])

    def test_minimal_valid_aig(self):
        G = nx.DiGraph(); self._add_node(G, 0, NODE_PI); self._add_node(G, 1, NODE_PI)
        self._add_node(G, 2, NODE_AND); self._add_node(G, 3, NODE_PO)
        self._add_edge(G, 0, 2); self._add_edge(G, 1, 2); self._add_edge(G, 2, 3)
        metrics = calculate_structural_aig_metrics(G)
        self.assertTrue(metrics['is_structurally_valid'], msg=f"Constraints failed: {metrics['constraints_failed']}")
        self.assertTrue(metrics['is_dag']); self.assertEqual(metrics['num_nodes'], 4)

    def test_valid_aig_with_const0(self):
        G = nx.DiGraph(); self._add_node(G, 0, NODE_CONST0); self._add_node(G, 1, NODE_PI)
        self._add_node(G, 2, NODE_AND); self._add_node(G, 3, NODE_PO)
        self._add_edge(G, 0, 2); self._add_edge(G, 1, 2); self._add_edge(G, 2, 3)
        metrics = calculate_structural_aig_metrics(G)
        self.assertTrue(metrics['is_structurally_valid'], msg=f"Constraints failed: {metrics['constraints_failed']}")

    def test_invalid_not_dag(self):
        G = nx.DiGraph(); self._add_node(G, 0, NODE_PI); self._add_node(G, 1, NODE_AND)
        self._add_node(G, 2, NODE_PI); self._add_node(G, 3, NODE_PO)
        self._add_edge(G, 0, 1); self._add_edge(G, 2, 1); self._add_edge(G, 1, 3); self._add_edge(G, 3, 0) # Cycle
        metrics = calculate_structural_aig_metrics(G)
        self.assertFalse(metrics['is_dag']); self.assertFalse(metrics['is_structurally_valid'])
        self.assertIn("Not a DAG", metrics['constraints_failed'])

    def test_invalid_pi_indegree(self):
        G = nx.DiGraph(); self._add_node(G, 0, NODE_PI); self._add_node(G, 1, NODE_PI)
        self._add_node(G, 2, NODE_AND); self._add_node(G, 3, NODE_PO)
        self._add_edge(G, 1, 0); self._add_edge(G, 0, 2); self._add_edge(G, 1, 2); self._add_edge(G, 2, 3)
        metrics = calculate_structural_aig_metrics(G)
        self.assertFalse(metrics['is_structurally_valid'])
        self.assertEqual(metrics['pi_indegree_violations'], 1)

    def test_invalid_and_indegree(self):
        G = nx.DiGraph(); self._add_node(G, 0, NODE_PI); self._add_node(G, 1, NODE_AND)
        self._add_node(G, 2, NODE_PO); self._add_edge(G, 0, 1); self._add_edge(G, 1, 2)
        metrics = calculate_structural_aig_metrics(G)
        self.assertFalse(metrics['is_structurally_valid'])
        self.assertEqual(metrics['and_indegree_violations'], 1)

    def test_invalid_po_outdegree(self):
        G = nx.DiGraph(); self._add_node(G, 0, NODE_PI); self._add_node(G, 1, NODE_PI)
        self._add_node(G, 2, NODE_AND); self._add_node(G, 3, NODE_PO); self._add_node(G, 4, NODE_AND)
        self._add_edge(G, 0, 2); self._add_edge(G, 1, 2); self._add_edge(G, 2, 3); self._add_edge(G, 3, 4)
        metrics = calculate_structural_aig_metrics(G)
        self.assertFalse(metrics['is_structurally_valid'])
        self.assertEqual(metrics['po_outdegree_violations'], 1)

    def test_invalid_po_indegree_zero(self):
        G = nx.DiGraph(); self._add_node(G, 0, NODE_PI); self._add_node(G, 1, NODE_PI)
        self._add_node(G, 2, NODE_AND); self._add_node(G, 3, NODE_PO)
        self._add_edge(G, 0, 2); self._add_edge(G, 1, 2)
        metrics = calculate_structural_aig_metrics(G)
        self.assertFalse(metrics['is_structurally_valid'])
        self.assertEqual(metrics['po_indegree_violations'], 1)

    def test_invalid_const0_indegree(self):
        G = nx.DiGraph(); self._add_node(G, 0, NODE_CONST0); self._add_node(G, 1, NODE_PI)
        self._add_node(G, 2, NODE_AND); self._add_node(G, 3, NODE_PO)
        self._add_edge(G, 1, 0); self._add_edge(G, 0, 2); self._add_edge(G, 1, 2); self._add_edge(G, 2, 3)
        metrics = calculate_structural_aig_metrics(G)
        self.assertFalse(metrics['is_structurally_valid'])
        self.assertEqual(metrics['const0_indegree_violations'], 1)

    def test_invalid_unknown_node_type(self):
        G = nx.DiGraph(); self._add_node(G, 0, NODE_PI); self._add_node(G, 1, "NODE_UNKNOWN")
        self._add_node(G, 2, NODE_PO); self._add_edge(G, 0, 1); self._add_edge(G, 1, 2)
        metrics = calculate_structural_aig_metrics(G)
        self.assertFalse(metrics['is_structurally_valid'])
        self.assertEqual(metrics['num_unknown_nodes'], 1)

    def test_invalid_unknown_edge_type(self):
        G = nx.DiGraph(); self._add_node(G, 0, NODE_PI); self._add_node(G, 1, NODE_PI)
        self._add_node(G, 2, NODE_AND); self._add_node(G, 3, NODE_PO)
        self._add_edge(G, 0, 2); self._add_edge(G, 1, 2, edge_type="EDGE_UNKNOWN"); self._add_edge(G, 2, 3)
        metrics = calculate_structural_aig_metrics(G)
        self.assertFalse(metrics['is_structurally_valid'])
        self.assertEqual(metrics['num_unknown_edges'], 1)

    def test_invalid_missing_pi_const0(self):
        G = nx.DiGraph(); self._add_node(G, 0, NODE_AND); self._add_node(G, 1, NODE_PO)
        self._add_edge(G, 0, 1)
        metrics = calculate_structural_aig_metrics(G)
        self.assertFalse(metrics['is_structurally_valid'])
        self.assertIn("No Primary Inputs or Const0 found", metrics['constraints_failed'])

    def test_invalid_missing_and(self):
        if MIN_AND_COUNT > 0:
            G = nx.DiGraph(); self._add_node(G, 0, NODE_PI); self._add_node(G, 1, NODE_PO); self._add_edge(G, 0, 1)
            metrics = calculate_structural_aig_metrics(G)
            self.assertFalse(metrics['is_structurally_valid'])
            self.assertIn(f"Insufficient AND gates (0 < {MIN_AND_COUNT})", "".join(metrics['constraints_failed']))
        else: self.skipTest("Skipping test because MIN_AND_COUNT_CONFIG is 0.")

    def test_invalid_missing_po(self):
        if MIN_PO_COUNT > 0:
            G = nx.DiGraph(); self._add_node(G, 0, NODE_PI); self._add_node(G, 1, NODE_PI); self._add_node(G, 2, NODE_AND)
            self._add_edge(G, 0, 2); self._add_edge(G, 1, 2)
            metrics = calculate_structural_aig_metrics(G)
            self.assertFalse(metrics['is_structurally_valid'])
            self.assertIn(f"Insufficient POs (0 < {MIN_PO_COUNT})", "".join(metrics['constraints_failed']))
        else: self.skipTest("Skipping test because MIN_PO_COUNT_CONFIG is 0.")

    def test_isolated_node_does_not_invalidate(self):
        G = nx.DiGraph(); self._add_node(G, 0, NODE_PI); self._add_node(G, 1, NODE_PI); self._add_node(G, 2, NODE_AND)
        self._add_node(G, 3, NODE_PO); self._add_edge(G, 0, 2); self._add_edge(G, 1, 2); self._add_edge(G, 2, 3)
        self._add_node(G, 4, NODE_PI) # Isolated PI
        metrics = calculate_structural_aig_metrics(G)
        self.assertTrue(metrics['is_structurally_valid'], msg=f"Constraints failed: {metrics['constraints_failed']}")
        self.assertEqual(metrics['isolated_nodes'], 1)

    def test_isolated_const0_is_ok(self):
        G = nx.DiGraph(); self._add_node(G, 1, NODE_PI); self._add_node(G, 2, NODE_PI); self._add_node(G, 3, NODE_AND)
        self._add_node(G, 4, NODE_PO); self._add_edge(G, 1, 3); self._add_edge(G, 2, 3); self._add_edge(G, 3, 4)
        self._add_node(G, 0, NODE_CONST0) # Isolated CONST0
        metrics = calculate_structural_aig_metrics(G)
        self.assertTrue(metrics['is_structurally_valid'], msg=f"Constraints failed: {metrics['constraints_failed']}")
        self.assertEqual(metrics['isolated_nodes'], 0)

    # === Path Connectivity Tests ===
    def test_path_connectivity_full(self):
        G = nx.DiGraph(); self._add_node(G, 0, NODE_PI); self._add_node(G, 1, NODE_PI); self._add_node(G, 2, NODE_AND)
        self._add_node(G, 3, NODE_PO); self._add_edge(G, 0, 2); self._add_edge(G, 1, 2); self._add_edge(G, 2, 3)
        path_metrics = count_pi_po_paths(G); self.assertIsNone(path_metrics.get('error'))
        self.assertEqual(path_metrics['num_pis_reaching_po'], 2)
        self.assertEqual(path_metrics['num_pos_reachable_from_pi'], 1)
        self.assertAlmostEqual(path_metrics['fraction_pis_connected'], 1.0)
        self.assertAlmostEqual(path_metrics['fraction_pos_connected'], 1.0)

    def test_path_connectivity_partial_pi(self):
        G = nx.DiGraph(); self._add_node(G, 0, NODE_PI); self._add_node(G, 1, NODE_PI); self._add_node(G, 2, NODE_AND)
        self._add_node(G, 3, NODE_PO); self._add_node(G, 4, NODE_CONST0)
        self._add_edge(G, 0, 2); self._add_edge(G, 4, 2); self._add_edge(G, 2, 3) # PI 1 is disconnected source
        path_metrics = count_pi_po_paths(G); self.assertIsNone(path_metrics.get('error'))
        self.assertEqual(path_metrics['num_pis_reaching_po'], 2) # PI 0 and CONST0 4 reach PO 3
        self.assertEqual(path_metrics['num_pos_reachable_from_pi'], 1) # PO 3 is reachable
        self.assertAlmostEqual(path_metrics['fraction_pis_connected'], 2.0 / 3.0) # 2 sources connected / 3 total sources

    def test_path_connectivity_partial_po(self):
        G = nx.DiGraph(); self._add_node(G, 0, NODE_PI); self._add_node(G, 1, NODE_PI); self._add_node(G, 2, NODE_AND)
        self._add_node(G, 3, NODE_PO); self._add_node(G, 4, NODE_PO) # PO 4 is unconnected
        self._add_edge(G, 0, 2); self._add_edge(G, 1, 2); self._add_edge(G, 2, 3)
        path_metrics = count_pi_po_paths(G); self.assertIsNone(path_metrics.get('error'))
        self.assertEqual(path_metrics['num_pis_reaching_po'], 2)
        self.assertEqual(path_metrics['num_pos_reachable_from_pi'], 1) # Only PO 3 reachable
        self.assertAlmostEqual(path_metrics['fraction_pos_connected'], 0.5) # 1 PO reachable / 2 total POs

    def test_path_connectivity_none(self):
        G = nx.DiGraph(); self._add_node(G, 0, NODE_PI); self._add_node(G, 1, NODE_PI); self._add_node(G, 2, NODE_AND)
        self._add_node(G, 3, NODE_PO) # All unconnected
        path_metrics = count_pi_po_paths(G); self.assertIsNone(path_metrics.get('error'))
        self.assertEqual(path_metrics['num_pis_reaching_po'], 0)
        self.assertEqual(path_metrics['num_pos_reachable_from_pi'], 0)
        self.assertAlmostEqual(path_metrics['fraction_pis_connected'], 0.0)
        self.assertAlmostEqual(path_metrics['fraction_pos_connected'], 0.0)

    def test_path_connectivity_with_const0(self):
        G = nx.DiGraph(); self._add_node(G, 0, NODE_CONST0); self._add_node(G, 1, NODE_PI)
        self._add_node(G, 2, NODE_AND); self._add_node(G, 3, NODE_PO); self._add_node(G, 4, NODE_PI)
        self._add_edge(G, 0, 2); self._add_edge(G, 4, 2); self._add_edge(G, 2, 3) # PI 1 disconnected
        path_metrics = count_pi_po_paths(G); self.assertIsNone(path_metrics.get('error'))
        self.assertEqual(path_metrics['num_pis_reaching_po'], 2) # CONST0 0 and PI 4 reach PO 3
        self.assertEqual(path_metrics['num_pos_reachable_from_pi'], 1)
        self.assertAlmostEqual(path_metrics['fraction_pis_connected'], 2.0 / 3.0) # 2 sources connected / 3 total sources


# --- Data Validation Tests ---


class TestDataValidation(unittest.TestCase):
    """Contains tests to validate actual dataset files (.pt, .bin) against evaluation logic."""

    @classmethod
    def setUpClass(cls):
        """Set up for all tests in this class."""
        cls.pyg_data_available = pyg_data_exists
        cls.bin_data_available = bin_data_exists
        if not cls.pyg_data_available:
            print(f"WARNING: PyG test data not found in {TEST_DATA_DIR}/processed/aig_processed_{TEST_SPLIT_NAME}.pt. PyG data tests will be skipped.")
        if not cls.bin_data_available:
            print(f"WARNING: Bin test data not found in {TEST_DATA_DIR}/{TEST_SPLIT_NAME} or {TEST_META_FILE}. Bin data tests will be skipped.")

    @unittest.skipUnless(pyg_data_exists, f"PyG test data not found at {TEST_DATA_DIR}")
    def test_pyg_dataset_graphs_are_valid(self):
        """Verify that graphs loaded from the PyG test dataset are structurally valid."""
        try:
            # Load the test split from the specified directory
            pyg_dataset = AIGPygDataset(root=TEST_DATA_DIR, split=TEST_SPLIT_NAME)
        except Exception as e:
            self.fail(f"Failed to load PyG test dataset from {TEST_DATA_DIR}: {e}")

        self.assertGreater(len(pyg_dataset), 0, "PyG test dataset is empty.")

        invalid_graphs = []
        print(f"Validating {len(pyg_dataset)} graphs from PyG test data...")
        for i in range(len(pyg_dataset)):
            try:
                data = pyg_dataset[i]
                nx_graph = pyg_data_to_nx(data) # Use the conversion function
                if nx_graph is None:
                    invalid_graphs.append({'index': i, 'reason': 'Failed NX conversion', 'details': 'None'})
                    continue

                metrics = calculate_structural_aig_metrics(nx_graph)
                if not metrics.get('is_structurally_valid', False):
                    invalid_graphs.append({
                        'index': i,
                        'reason': 'Failed structural validation',
                        'details': metrics.get('constraints_failed', [])
                    })
            except Exception as e:
                 invalid_graphs.append({'index': i, 'reason': f'Error during processing: {e}', 'details': None})

        # Assert that the list of invalid graphs is empty
        self.assertEqual(len(invalid_graphs), 0,
                         f"Found {len(invalid_graphs)} invalid graphs in PyG test data: {invalid_graphs}")
        print("PyG data validation successful.")

    @unittest.skipUnless(bin_data_exists, f"Bin test data or meta file not found at {TEST_DATA_DIR}")
    def test_bin_dataset_graphs_are_valid(self):
        """Verify that graphs loaded and unpadded from the Bin test dataset are structurally valid."""
        try:
            with open(TEST_META_FILE, 'r') as f:
                data_meta = json.load(f)
            split_shape_key = f"{TEST_SPLIT_NAME}_shape"
            if split_shape_key not in data_meta:
                 raise KeyError(f"Shape information for split '{TEST_SPLIT_NAME}' not found in {TEST_META_FILE}")
            bin_shape = data_meta[split_shape_key]

            # --- Load the bin dataset using NumpyBinDataset ---
            # This requires the correct path to the *directory* containing the .bin files
            bin_split_path = os.path.join(TEST_DATA_DIR, TEST_SPLIT_NAME)
            if not os.path.isdir(bin_split_path):
                 raise FileNotFoundError(f"Bin data directory not found: {bin_split_path}")

            # Note: NumpyBinDataset needs process_fn, but we won't call __getitem__
            # Pass dummy values for unused args.
            # Ensure aig_cfg is available for PAD_VALUE access during unpadding
            if 'aig_cfg' not in globals():
                 import G2PT.configs.aig as aig_cfg # Import locally if needed
                 print("Imported aig_cfg locally for bin test.")

            bin_dataset = NumpyBinDataset(
                path=bin_split_path,
                num_data=bin_shape['xs'][0],
                num_node_class=4, num_edge_class=2, # Placeholders
                shape=bin_shape,
                process_fn=None, # Not used in this test
                num_augmentations=1
            )
        except Exception as e:
            self.fail(f"Failed to load Bin test dataset or meta file: {e}")

        self.assertGreater(len(bin_dataset.xs), 0, "Bin test dataset is empty.")

        invalid_graphs = []
        num_bin_graphs = len(bin_dataset.xs)
        print(f"Validating {num_bin_graphs} graphs from Bin test data...")

        for i in range(num_bin_graphs):
            try:
                # --- Manual Unpadding Logic (adapted from validate_input_data.py / datasets_utils.py) ---
                # Access memmap arrays directly
                raw_x = np.array(bin_dataset.xs[i]).astype(np.int64)
                raw_edge_index = np.array(bin_dataset.edge_indices[i]).astype(np.int64) # Shape [2, max_E]
                raw_edge_attr = np.array(bin_dataset.edge_attrs[i]).astype(np.int64)   # Shape [max_E]

                # 1. Unpad nodes based on node features (x)
                # Use PAD_VALUE from config
                node_padding_mask = raw_x != aig_cfg.PAD_VALUE
                x_ids = torch.from_numpy(raw_x[node_padding_mask]) # Node Vocab IDs
                num_valid_nodes = len(x_ids)

                if num_valid_nodes == 0: continue # Skip if graph became empty

                # 2. Create mapping from old node indices to new node indices
                old_indices = np.arange(len(raw_x))
                new_indices_map = -np.ones_like(old_indices, dtype=np.int64)
                new_indices_map[node_padding_mask] = np.arange(num_valid_nodes)

                # 3. Unpad edges based on edge attributes
                if raw_edge_attr.ndim > 1: raw_edge_attr = raw_edge_attr.flatten() # Ensure 1D
                edge_padding_mask = raw_edge_attr != aig_cfg.PAD_VALUE # Edge Vocab IDs

                # Check for shape consistency before filtering edge index
                if edge_padding_mask.shape[0] != raw_edge_index.shape[1]:
                    edge_index_filtered_by_attr = torch.tensor([[], []], dtype=torch.long)
                    edge_attr_ids_filtered_by_attr = torch.tensor([], dtype=torch.long)
                    print(f"Warning: Shape mismatch edge_attr vs edge_index for graph {i}. Assuming no valid edges.")
                else:
                    edge_attr_ids_filtered_by_attr = torch.from_numpy(raw_edge_attr[edge_padding_mask])
                    edge_index_filtered_by_attr = torch.from_numpy(raw_edge_index[:, edge_padding_mask])


                # 4. Remap edge indices and filter edges pointing to/from padded nodes
                if edge_index_filtered_by_attr.numel() > 0:
                    src_nodes_old = edge_index_filtered_by_attr[0, :].numpy()
                    dst_nodes_old = edge_index_filtered_by_attr[1, :].numpy()
                    src_nodes_new = new_indices_map[src_nodes_old]
                    dst_nodes_new = new_indices_map[dst_nodes_old]
                    valid_edge_mask = (src_nodes_new != -1) & (dst_nodes_new != -1)
                    edge_index_final = torch.tensor([src_nodes_new[valid_edge_mask], dst_nodes_new[valid_edge_mask]], dtype=torch.long)
                    edge_attr_final = edge_attr_ids_filtered_by_attr[valid_edge_mask] # Edge Vocab IDs
                else:
                    edge_index_final = torch.tensor([[], []], dtype=torch.long)
                    edge_attr_final = torch.tensor([], dtype=torch.long)
                # --- End Unpadding ---

                # Convert the unpadded data (with vocab IDs) to NetworkX
                # bin_data_to_nx expects vocab IDs
                nx_graph = bin_data_to_nx(x_ids, edge_index_final, edge_attr_final)
                if nx_graph is None:
                    invalid_graphs.append({'index': i, 'reason': 'Failed NX conversion', 'details': 'None'})
                    continue

                # Validate the resulting graph
                metrics = calculate_structural_aig_metrics(nx_graph)
                if not metrics.get('is_structurally_valid', False):
                     invalid_graphs.append({
                         'index': i,
                         'reason': 'Failed structural validation',
                         'details': metrics.get('constraints_failed', [])
                     })
            except Exception as e:
                 invalid_graphs.append({'index': i, 'reason': f'Error during processing: {e}', 'details': None})


        # Assert that the list of invalid graphs is empty
        self.assertEqual(len(invalid_graphs), 0,
                         f"Found {len(invalid_graphs)} invalid graphs in Bin test data: {invalid_graphs}")
        print("Bin data validation successful.")


if __name__ == '__main__':
    # Ensure logging from evaluate_aigs doesn't interfere too much during tests
    logging.getLogger("evaluate_g2pt_aigs").setLevel(logging.WARNING)
    # Run tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

