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
import tempfile # Added for novelty test setup
import pickle   # Added for novelty test setup

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
# Import core evaluation functions and new VUN functions
from ..evaluate_aigs import (
    calculate_structural_aig_metrics,
    count_pi_po_paths,
    calculate_uniqueness,  # <-- NEW IMPORT
    calculate_novelty,     # <-- NEW IMPORT
    load_training_graphs_from_bin # <-- NEW IMPORT (if testing loading directly)
)
# Import necessary config items
from ..configs.aig import (NODE_TYPE_KEYS, EDGE_TYPE_KEYS,
                           MIN_AND_COUNT, MIN_PO_COUNT, NODE_TYPE_VOCAB, # Keep needed constants
                           PAD_VALUE) # <-- NEW IMPORT for bin loading tests

# Assuming graph_utils.py is in the same tests directory
from .graph_utils import pyg_data_to_nx # Keep pyg_data_to_nx if used
# bin_data_to_nx is now defined within evaluate_aigs.py, so no need to import from graph_utils

print("Imported evaluation functions successfully.")

# Assuming aig_dataset.py is in G2PT/datasets/
try:
    from ..datasets.aig_dataset import AIGPygDataset
    print("Imported AIGPygDataset successfully.")
except ImportError:
     print("Could not import AIGPygDataset (needed for some data tests).")
     AIGPygDataset = None # Define as None if import fails

# Assuming datasets_utils.py is in G2PT/
try:
    from ..datasets_utils import NumpyBinDataset
    print("Imported NumpyBinDataset successfully.")
except ImportError:
     print("Could not import NumpyBinDataset (needed for some data tests).")
     NumpyBinDataset = None # Define as None if import fails


# --- Node Type String Constants (Corrected) ---
NODE_PI = "NODE_PI"
NODE_AND = "NODE_AND"
NODE_PO = "NODE_PO"
NODE_CONST0 = "NODE_CONST0"
EDGE_REG = "EDGE_REG"
EDGE_INV = "EDGE_INV"
# ---------------------------------------------


# --- Unit Test Classes ---

class TestEvaluateAIGs(unittest.TestCase):
    """Contains unit tests for the core structural evaluation logic."""

    def setUp(self):
        """Executed before each test method in this class."""
        pass # No specific setup needed

    # Helper method using STRING types
    def _create_graph(self, nodes_edges_spec):
        """Creates a graph from a spec: [(id, type), ...] for nodes, [(u, v, type), ...] for edges"""
        G = nx.DiGraph()
        nodes, edges = nodes_edges_spec
        for node_id, node_type in nodes:
            G.add_node(node_id, type=node_type)
        for u, v, edge_type in edges:
            G.add_edge(u, v, type=edge_type)
        return G

    # === Structural Validity Tests ===
    # (Keep all the original tests here - they should pass now with string types)
    def test_empty_graph(self):
        G = self._create_graph(([], []))
        metrics = calculate_structural_aig_metrics(G)
        self.assertFalse(metrics['is_structurally_valid'])
        self.assertIn("Empty or Invalid Graph Object", metrics['constraints_failed'])

    def test_minimal_valid_aig(self):
        G = self._create_graph((
            [(0, NODE_PI), (1, NODE_PI), (2, NODE_AND), (3, NODE_PO)],
            [(0, 2, EDGE_REG), (1, 2, EDGE_REG), (2, 3, EDGE_REG)]
        ))
        metrics = calculate_structural_aig_metrics(G)
        self.assertTrue(metrics['is_structurally_valid'], msg=f"Constraints failed: {metrics.get('constraints_failed', 'N/A')}")
        self.assertTrue(metrics['is_dag']); self.assertEqual(metrics['num_nodes'], 4)

    def test_valid_aig_with_const0(self):
        G = self._create_graph((
            [(0, NODE_CONST0), (1, NODE_PI), (2, NODE_AND), (3, NODE_PO)],
            [(0, 2, EDGE_REG), (1, 2, EDGE_INV), (2, 3, EDGE_REG)] # Added INV edge type
        ))
        metrics = calculate_structural_aig_metrics(G)
        self.assertTrue(metrics['is_structurally_valid'], msg=f"Constraints failed: {metrics.get('constraints_failed', 'N/A')}")

    def test_invalid_not_dag(self):
        G = self._create_graph((
             [(0, NODE_PI), (1, NODE_AND), (2, NODE_PI), (3, NODE_PO)],
             [(0, 1, EDGE_REG), (2, 1, EDGE_REG), (1, 3, EDGE_REG), (3, 0, EDGE_REG)] # Cycle 3->0
        ))
        metrics = calculate_structural_aig_metrics(G)
        self.assertFalse(metrics['is_dag']); self.assertFalse(metrics['is_structurally_valid'])
        self.assertIn("Not a DAG", metrics['constraints_failed'])

    # ... (keep all other structural and path connectivity tests as they were) ...
    # They should work correctly now that _create_graph uses string types.

    def test_isolated_node_does_not_invalidate(self):
        G = self._create_graph((
            [(0, NODE_PI), (1, NODE_PI), (2, NODE_AND), (3, NODE_PO), (4, NODE_PI)], # Node 4 is isolated
            [(0, 2, EDGE_REG), (1, 2, EDGE_REG), (2, 3, EDGE_REG)]
        ))
        metrics = calculate_structural_aig_metrics(G)
        # Validity should still hold, but isolated_nodes count should be 1
        self.assertTrue(metrics['is_structurally_valid'], msg=f"Constraints failed: {metrics.get('constraints_failed', 'N/A')}")
        self.assertEqual(metrics['isolated_nodes'], 1)

    def test_isolated_const0_is_ok(self):
        G = self._create_graph((
            [(0, NODE_CONST0), (1, NODE_PI), (2, NODE_PI), (3, NODE_AND), (4, NODE_PO)], # Node 0 is isolated CONST0
            [(1, 3, EDGE_REG), (2, 3, EDGE_INV), (3, 4, EDGE_REG)]
        ))
        metrics = calculate_structural_aig_metrics(G)
        self.assertTrue(metrics['is_structurally_valid'], msg=f"Constraints failed: {metrics.get('constraints_failed', 'N/A')}")
        # Isolated CONST0 node should not count towards the 'isolated_nodes' metric
        self.assertEqual(metrics['isolated_nodes'], 0)


# --- NEW Test Class for Uniqueness and Novelty ---
class TestVUNMetrics(unittest.TestCase):
    """Tests uniqueness and novelty calculation functions."""

    def setUp(self):
        """Create some simple graphs for testing."""
        # Graph 1: Simple AND gate
        self.g1 = nx.DiGraph()
        self.g1.add_node(0, type=NODE_PI)
        self.g1.add_node(1, type=NODE_PI)
        self.g1.add_node(2, type=NODE_AND)
        self.g1.add_node(3, type=NODE_PO)
        self.g1.add_edges_from([(0, 2, {'type': EDGE_REG}), (1, 2, {'type': EDGE_REG}), (2, 3, {'type': EDGE_REG})])

        # Graph 2: Isomorphic to g1 but different node IDs
        self.g2 = nx.DiGraph()
        self.g2.add_node(10, type=NODE_PI)
        self.g2.add_node(11, type=NODE_PI)
        self.g2.add_node(12, type=NODE_AND)
        self.g2.add_node(13, type=NODE_PO)
        self.g2.add_edges_from([(10, 12, {'type': EDGE_REG}), (11, 12, {'type': EDGE_REG}), (12, 13, {'type': EDGE_REG})])

        # Graph 3: Different structure (extra AND)
        self.g3 = nx.DiGraph()
        self.g3.add_node(0, type=NODE_PI)
        self.g3.add_node(1, type=NODE_PI)
        self.g3.add_node(2, type=NODE_AND)
        self.g3.add_node(3, type=NODE_AND) # Extra AND
        self.g3.add_node(4, type=NODE_PO)
        self.g3.add_edges_from([(0, 2, {'type': EDGE_REG}), (1, 2, {'type': EDGE_REG}),
                               (1, 3, {'type': EDGE_INV}), (2, 3, {'type': EDGE_REG}), # Different edge type
                               (3, 4, {'type': EDGE_REG})])

        # Graph 4: Isomorphic to g3
        self.g4 = nx.DiGraph()
        self.g4.add_node(20, type=NODE_PI)
        self.g4.add_node(21, type=NODE_PI)
        self.g4.add_node(22, type=NODE_AND)
        self.g4.add_node(23, type=NODE_AND)
        self.g4.add_node(24, type=NODE_PO)
        self.g4.add_edges_from([(20, 22, {'type': EDGE_REG}), (21, 22, {'type': EDGE_REG}),
                               (21, 23, {'type': EDGE_INV}), (22, 23, {'type': EDGE_REG}),
                               (23, 24, {'type': EDGE_REG})])

        # Graph 5: Different edge type compared to g1
        self.g5 = nx.DiGraph()
        self.g5.add_node(0, type=NODE_PI)
        self.g5.add_node(1, type=NODE_PI)
        self.g5.add_node(2, type=NODE_AND)
        self.g5.add_node(3, type=NODE_PO)
        self.g5.add_edges_from([(0, 2, {'type': EDGE_INV}), (1, 2, {'type': EDGE_REG}), (2, 3, {'type': EDGE_REG})]) # Edge 0->2 is INV

    # --- Uniqueness Tests ---
    def test_uniqueness_all_unique(self):
        graphs = [self.g1, self.g3, self.g5] # Should be non-isomorphic
        score, count = calculate_uniqueness(graphs)
        self.assertAlmostEqual(score, 1.0)
        self.assertEqual(count, 3)

    def test_uniqueness_one_duplicate_pair(self):
        graphs = [self.g1, self.g3, self.g2] # g1 and g2 are isomorphic
        score, count = calculate_uniqueness(graphs)
        self.assertAlmostEqual(score, 2.0 / 3.0) # g1 (or g2) and g3 are unique
        self.assertEqual(count, 2)

    def test_uniqueness_multiple_duplicates(self):
        graphs = [self.g1, self.g3, self.g2, self.g4, self.g1] # g1/g2/g1_dup isomorphic, g3/g4 isomorphic
        score, count = calculate_uniqueness(graphs)
        self.assertAlmostEqual(score, 2.0 / 5.0) # Only g1 structure and g3 structure are unique
        self.assertEqual(count, 2)

    def test_uniqueness_all_duplicates(self):
        graphs = [self.g1, self.g2, self.g1] # All isomorphic to g1
        score, count = calculate_uniqueness(graphs)
        self.assertAlmostEqual(score, 1.0 / 3.0) # Only one unique structure
        self.assertEqual(count, 1)

    def test_uniqueness_edge_type_matters(self):
        graphs = [self.g1, self.g5] # g1 and g5 differ only by one edge type
        score, count = calculate_uniqueness(graphs)
        self.assertAlmostEqual(score, 1.0) # Should be considered different
        self.assertEqual(count, 2)

    def test_uniqueness_empty_list(self):
        graphs = []
        score, count = calculate_uniqueness(graphs)
        self.assertAlmostEqual(score, 1.0) # Definitionally 100% unique
        self.assertEqual(count, 0)

    def test_uniqueness_single_graph(self):
        graphs = [self.g1]
        score, count = calculate_uniqueness(graphs)
        self.assertAlmostEqual(score, 1.0)
        self.assertEqual(count, 1)

    # --- Novelty Tests ---
    def test_novelty_all_novel(self):
        generated = [self.g3, self.g5]
        training = [self.g1, self.g2]
        score, count = calculate_novelty(generated, training)
        self.assertAlmostEqual(score, 1.0)
        self.assertEqual(count, 2)

    def test_novelty_none_novel(self):
        generated = [self.g1, self.g2]
        training = [self.g1, self.g3] # Training contains g1
        score, count = calculate_novelty(generated, training)
        self.assertAlmostEqual(score, 0.0) # Both g1 and g2 are isomorphic to g1 in training
        self.assertEqual(count, 0)

    def test_novelty_mixed(self):
        generated = [self.g1, self.g3, self.g5] # g1 is in training, g3 and g5 are not
        training = [self.g2, self.g4]        # g2 iso to g1, g4 iso to g3
        score, count = calculate_novelty(generated, training)
        # g1 matches g2, g3 matches g4, g5 is novel
        self.assertAlmostEqual(score, 1.0 / 3.0) # Only g5 is novel
        self.assertEqual(count, 1)

    def test_novelty_empty_generated(self):
        generated = []
        training = [self.g1, self.g3]
        score, count = calculate_novelty(generated, training)
        self.assertAlmostEqual(score, 0.0) # Or should this be 1.0? Conventionally 0.
        self.assertEqual(count, 0)

    def test_novelty_empty_training(self):
        generated = [self.g1, self.g3]
        training = []
        score, count = calculate_novelty(generated, training)
        self.assertAlmostEqual(score, 1.0) # All generated are novel if train set is empty
        self.assertEqual(count, 2)

# --- Data Validation Tests ---
# Keep the TestDataValidation class as it was, assuming it passed after the previous fix.
# No changes needed here unless new failures arise related to data loading itself.
@unittest.skipUnless(AIGPygDataset is not None and NumpyBinDataset is not None, "Skipping data validation tests due to missing imports.")
class TestDataValidation(unittest.TestCase):
    """Contains tests to validate actual dataset files (.pt, .bin) against evaluation logic."""
    # ... (keep the existing setUpClass and test methods) ...
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
        max_graphs_to_check = min(len(pyg_dataset), 100) # Limit check for speed if dataset large
        print(f"Validating first {max_graphs_to_check} graphs from PyG test data...")
        for i in range(max_graphs_to_check):
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

        # Assert that the list of invalid graphs is empty for the checked subset
        self.assertEqual(len(invalid_graphs), 0,
                         f"Found {len(invalid_graphs)} invalid graphs in the first {max_graphs_to_check} PyG graphs: {invalid_graphs}")
        print("PyG data validation successful for checked subset.")

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
            num_graphs_in_bin = bin_shape['xs'][0]

            # --- Load the bin dataset using NumpyBinDataset ---
            bin_split_path = os.path.join(TEST_DATA_DIR, TEST_SPLIT_NAME)
            if not os.path.isdir(bin_split_path):
                 raise FileNotFoundError(f"Bin data directory not found: {bin_split_path}")

            # Need to import aig_config here if not globally available
            try:
                 from ..configs import aig as aig_cfg
            except ImportError:
                 self.fail("Failed to import aig_cfg for PAD_VALUE in bin test.")

            # Create dummy process_fn as it's not used here
            dummy_process_fn = lambda x: x

            bin_dataset = NumpyBinDataset(
                path=bin_split_path,
                num_data=num_graphs_in_bin,
                num_node_class=len(NODE_TYPE_KEYS), # Use actual count
                num_edge_class=len(EDGE_TYPE_KEYS), # Use actual count
                shape=bin_shape,
                process_fn=dummy_process_fn,
                num_augmentations=1
            )
        except Exception as e:
            self.fail(f"Failed to load Bin test dataset or meta file: {e}")

        self.assertGreater(len(bin_dataset.xs), 0, "Bin test dataset is empty.")

        invalid_graphs = []
        max_graphs_to_check = min(num_graphs_in_bin, 100) # Limit check for speed
        print(f"Validating first {max_graphs_to_check} graphs from Bin test data...")

        # Import or define bin_data_to_nx needed for conversion
        # Let's reuse the one defined in evaluate_aigs.py by importing it
        try:
             from ..evaluate_aigs import bin_data_to_nx as evaluate_bin_to_nx
        except ImportError:
             self.fail("Could not import bin_data_to_nx from evaluate_aigs.py")


        for i in range(max_graphs_to_check):
            try:
                # --- Reuse unpadding logic from NumpyBinDataset ---
                # Need to call internal logic or replicate it slightly differently here
                # Let's replicate simplified unpadding for testing purposes
                raw_x = np.array(bin_dataset.xs[i]).astype(np.int64)
                raw_edge_index = np.array(bin_dataset.edge_indices[i]).astype(np.int64)
                raw_edge_attr = np.array(bin_dataset.edge_attrs[i]).astype(np.int64)

                node_padding_mask = raw_x != PAD_VALUE
                x_ids = torch.from_numpy(raw_x[node_padding_mask])
                num_valid_nodes = len(x_ids)
                if num_valid_nodes == 0: continue

                old_indices = np.arange(len(raw_x))
                new_indices_map = -np.ones_like(old_indices, dtype=np.int64)
                new_indices_map[node_padding_mask] = np.arange(num_valid_nodes)

                if raw_edge_attr.ndim > 1: raw_edge_attr = raw_edge_attr.flatten()
                edge_padding_mask = raw_edge_attr != PAD_VALUE

                edge_index_final = torch.tensor([[], []], dtype=torch.long)
                edge_attr_final = torch.tensor([], dtype=torch.long)

                min_len = min(edge_padding_mask.shape[0], raw_edge_index.shape[1])
                edge_padding_mask_safe = edge_padding_mask[:min_len]
                edge_attr_ids_filtered = torch.from_numpy(raw_edge_attr[:min_len][edge_padding_mask_safe])
                edge_index_filtered = torch.from_numpy(raw_edge_index[:, :min_len][:, edge_padding_mask_safe])

                if edge_index_filtered.numel() > 0:
                    src_nodes_old = edge_index_filtered[0, :].numpy()
                    dst_nodes_old = edge_index_filtered[1, :].numpy()
                    src_nodes_old_safe = np.clip(src_nodes_old, 0, len(new_indices_map) - 1)
                    dst_nodes_old_safe = np.clip(dst_nodes_old, 0, len(new_indices_map) - 1)
                    src_nodes_new = new_indices_map[src_nodes_old_safe]
                    dst_nodes_new = new_indices_map[dst_nodes_old_safe]
                    valid_edge_mask = (src_nodes_new != -1) & (dst_nodes_new != -1)
                    edge_index_final = torch.tensor([src_nodes_new[valid_edge_mask], dst_nodes_new[valid_edge_mask]], dtype=torch.long)
                    edge_attr_final = edge_attr_ids_filtered[valid_edge_mask]

                # Use the conversion function imported/defined from evaluate_aigs
                nx_graph = evaluate_bin_to_nx(x_ids, edge_index_final, edge_attr_final)

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
                         f"Found {len(invalid_graphs)} invalid graphs in the first {max_graphs_to_check} Bin graphs: {invalid_graphs}")
        print("Bin data validation successful for checked subset.")


# --- Main Execution ---
if __name__ == '__main__':
    # Ensure logging from evaluate_aigs doesn't interfere too much during tests
    logging.getLogger("evaluate_g2pt_aigs").setLevel(logging.WARNING)
    # Run tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)