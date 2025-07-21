import unittest
import torch
import numpy as np
import os
import pickle
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from torch_geometric.data import Data

# We assume the data preparation script is saved as `data_prep.py`
try:
    from data_prep import (
        top_sort,
        add_order_info,
        _calculate_graph_stats,
        get_structural_difference_vector,
        create_aig_pyg_data,
        process_aig_pairs_directory,
        eigvec_normalizer,
        get_lap_decomp_stats,
    )
except ImportError as e:
    print(f"Could not find 'data_prep.py' or a dependency. Error: {e}")
    # Define dummy functions to allow the test suite to be defined
    def top_sort(ei, gs): return torch.zeros(gs, dtype=torch.long)
    def add_order_info(g): return g
    def _calculate_graph_stats(d): return torch.zeros(16)
    def get_structural_difference_vector(d1, d2): return torch.zeros(1, 16)
    def create_aig_pyg_data(aig): return None
    def process_aig_pairs_directory(b, o, c, u): pass
    def eigvec_normalizer(EigVecs, EigVals, normalization="L2", eps=1e-12): return EigVecs
    def get_lap_decomp_stats(evals, evects, max_freqs, eigvec_norm='L2'): return torch.randn(len(evals), max_freqs)


class TestDataPreparation(unittest.TestCase):
    """
    Unit tests for the AIG data preparation pipeline.
    """

    def setUp(self):
        """Set up common data for tests."""
        # Simple line graph: 0 -> 1 -> 2
        self.line_graph = Data(
            x=torch.randn(3, 4),
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            num_nodes=3
        )
        # Graph with a fork and join: 0 -> 1 -> 3, 0 -> 2 -> 3
        self.diamond_graph = Data(
            x=torch.randn(4, 4),
            edge_index=torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]], dtype=torch.long),
            num_nodes=4
        )
        self.diamond_graph.x[torch.tensor([1,2]),2] = 1 # Mark nodes 1 and 2 as AND gates for stats

    def test_top_sort(self):
        """Tests the topological sort function."""
        print("\nRunning test: top_sort")
        pe_line = top_sort(self.line_graph.edge_index, self.line_graph.num_nodes)
        self.assertTrue(torch.equal(pe_line, torch.tensor([0, 1, 2])))

        pe_diamond = top_sort(self.diamond_graph.edge_index, self.diamond_graph.num_nodes)
        self.assertTrue(torch.equal(pe_diamond, torch.tensor([0, 1, 1, 2])))
        print("...Passed")

    def test_add_order_info_attributes(self):
        """Tests that `add_order_info` adds all required attributes with correct shapes."""
        print("\nRunning test: add_order_info_attributes")
        processed_graph = add_order_info(self.line_graph.clone())
        
        self.assertTrue(hasattr(processed_graph, 'abs_pe'))
        self.assertTrue(hasattr(processed_graph, 'Eigvecs'))
        self.assertTrue(hasattr(processed_graph, 'dag_rr_edge_index'))
        self.assertTrue(hasattr(processed_graph, 'mask_rc'))
        
        self.assertEqual(processed_graph.abs_pe.shape, (self.line_graph.num_nodes,))
        self.assertEqual(processed_graph.Eigvecs.shape, (self.line_graph.num_nodes, 8))
        self.assertEqual(processed_graph.mask_rc.shape, (8, 8)) # Fixed size mask
        self.assertGreaterEqual(processed_graph.dag_rr_edge_index.shape[1], self.line_graph.num_edges)
        print("...Passed")
        
    def test_graph_with_isolated_node(self):
        """Tests a graph containing a node with no edges."""
        print("\nRunning test: Graph with Isolated Node")
        # Graph: 0 -> 1, with node 2 being isolated
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        data = Data(x=torch.randn(3, 4), edge_index=edge_index, num_nodes=3)
        processed_data = add_order_info(data.clone())
        
        # Node 2 is a root, just like node 0.
        expected_abs_pe = torch.tensor([0, 1, 0], dtype=torch.long)
        self.assertTrue(torch.equal(processed_data.abs_pe, expected_abs_pe))
        print("...Passed")

    def test_calculate_graph_stats(self):
        """Tests the calculation of the 16-dimensional structural vector."""
        print("\nRunning test: _calculate_graph_stats")
        processed_diamond = add_order_info(self.diamond_graph.clone())
        stats_vector = _calculate_graph_stats(processed_diamond)
        
        self.assertEqual(stats_vector.shape, (16,))
        self.assertEqual(stats_vector.dtype, torch.float)
        # Check a few specific stats
        self.assertAlmostEqual(stats_vector[0].item(), 4.0) # num_nodes
        self.assertAlmostEqual(stats_vector[1].item(), 4.0) # num_edges
        self.assertAlmostEqual(stats_vector[2].item(), 2.0) # num_and_gates
        print("...Passed")

    def test_calculate_graph_stats_edge_cases(self):
        """Tests _calculate_graph_stats with edge case graphs."""
        print("\nRunning test: _calculate_graph_stats Edge Cases")
        # Case 1: Graph with no edges
        graph_no_edges = Data(x=torch.randn(3, 4), num_nodes=3)
        graph_no_edges = add_order_info(graph_no_edges)
        stats_no_edges = _calculate_graph_stats(graph_no_edges)
        self.assertEqual(stats_no_edges[1].item(), 0) # num_edges
        self.assertEqual(stats_no_edges[5].item(), 0) # avg_fanout
        self.assertEqual(stats_no_edges[11].item(), 0) # density
        
        # Case 2: Disconnected graph
        edge_index = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
        disconnected_graph = Data(x=torch.randn(4, 4), edge_index=edge_index, num_nodes=4)
        disconnected_graph = add_order_info(disconnected_graph)
        stats_disconnected = _calculate_graph_stats(disconnected_graph)
        self.assertEqual(stats_disconnected[13].item(), -1.0) # diameter
        self.assertEqual(stats_disconnected[14].item(), -1.0) # radius
        self.assertAlmostEqual(stats_disconnected[12].item(), 0.0) # algebraic_connectivity
        print("...Passed")

    def test_get_structural_difference_vector(self):
        """Tests the difference vector calculation."""
        print("\nRunning test: get_structural_difference_vector")
        g1 = add_order_info(self.line_graph.clone())
        g2 = add_order_info(self.diamond_graph.clone())
        
        diff_vec = get_structural_difference_vector(g1, g2)
        self.assertEqual(diff_vec.shape, (1, 16))
        
        zero_diff_vec = get_structural_difference_vector(g1, g1)
        self.assertTrue(torch.all(zero_diff_vec == 0))
        print("...Passed")

    @patch('data_prep.read_aiger_into_aig')
    def test_create_aig_pyg_data(self, mock_read_aig):
        """Tests the end-to-end creation of a PyG data object from a mock AIG."""
        print("\nRunning test: create_aig_pyg_data")
        mock_aig = MagicMock()
        mock_aig.name = "test_aig"
        mock_aig.pis.return_value = [2]; mock_aig.gates.return_value = [4]; mock_aig.pos.return_value = [8]
        mock_aig.get_node.return_value = 4; mock_aig.is_complemented.return_value = True
        
        mock_edge = MagicMock(); mock_edge.source, mock_edge.target, mock_edge.weight = 2, 4, 0
        with patch('data_prep.to_edge_list', return_value=[mock_edge]):
            pyg_data = create_aig_pyg_data(mock_aig)

        self.assertIsNotNone(pyg_data)
        self.assertTrue(isinstance(pyg_data, Data))
        self.assertEqual(pyg_data.num_nodes, 4)
        self.assertEqual(pyg_data.num_edges, 2)
        self.assertTrue(hasattr(pyg_data, 'abs_pe'))
        print("...Passed")

    @patch('data_prep._save_data_chunk')
    @patch('data_prep.create_aig_pyg_data')
    @patch('data_prep.read_aiger_into_aig')
    def test_process_aig_pairs_directory_robust(self, mock_read_aig, mock_create_pyg, mock_save):
        """Tests the main directory processing logic with a more complex file system."""
        print("\nRunning test: process_aig_pairs_directory (Robust)")
        mock_create_pyg.return_value = self.line_graph.clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Set 1: Valid pair
            os.makedirs(os.path.join(tmpdir, "set1"));
            with open(os.path.join(tmpdir, "set1", "a1.aig"), "w") as f: f.write("dummy")
            with open(os.path.join(tmpdir, "set1", "a2.aig"), "w") as f: f.write("dummy")
            # Set 2: Should be skipped (only one file)
            os.makedirs(os.path.join(tmpdir, "set2_skipped"));
            with open(os.path.join(tmpdir, "set2_skipped", "b1.aag"), "w") as f: f.write("dummy")
            # Set 3: Valid pairs, ignoring other files
            os.makedirs(os.path.join(tmpdir, "set3"));
            with open(os.path.join(tmpdir, "set3", "c1.aig"), "w") as f: f.write("dummy")
            with open(os.path.join(tmpdir, "set3", "c2.aig"), "w") as f: f.write("dummy")
            with open(os.path.join(tmpdir, "set3", "c3.aig"), "w") as f: f.write("dummy")
            with open(os.path.join(tmpdir, "set3", "readme.txt"), "w") as f: f.write("ignore")
            # Set 4: Empty, should be skipped
            os.makedirs(os.path.join(tmpdir, "set4_empty"));

            output_file = os.path.join(tmpdir, "processed", "dataset.pkl")
            process_aig_pairs_directory(tmpdir, output_file, chunk_size=100, use_chunking=False)
            
            # Expected pairs: set1 has 1 combo (2 directed), set3 has 3 combos (6 directed) -> 8 total
            self.assertEqual(mock_save.call_count, 1)
            saved_data = mock_save.call_args[0][0]
            self.assertEqual(len(saved_data), 8)
        print("...Passed")

class TestHelperFunctions(unittest.TestCase):
    """Directly tests numerical helper functions."""
    
    def test_eigvec_normalizer(self):
        print("\nRunning test: eigvec_normalizer")
        vecs = torch.tensor([[3.0, 5.0], [4.0, 12.0]], dtype=torch.float)
        vals = torch.tensor([1.0, 2.0], dtype=torch.float)
        
        norm_l2 = eigvec_normalizer(vecs.clone(), vals, "L2")
        # Col 0 norm: sqrt(3^2 + 4^2) = 5. Expected: [0.6, 0.8]
        # Col 1 norm: sqrt(5^2 + 12^2) = 13. Expected: [5/13, 12/13]
        self.assertTrue(torch.allclose(norm_l2[:, 0], torch.tensor([0.6, 0.8])))
        self.assertTrue(torch.allclose(norm_l2[:, 1], torch.tensor([5.0/13.0, 12.0/13.0])))
        print("...Passed")

    def test_get_lap_decomp_stats_padding(self):
        """Tests the padding logic of get_lap_decomp_stats."""
        print("\nRunning test: get_lap_decomp_stats padding")
        evals = np.array([0.1, 0.5, 1.0])
        evects = np.random.rand(3, 3)
        max_freqs = 5
        
        EigVecs = get_lap_decomp_stats(evals, evects, max_freqs)
        self.assertEqual(EigVecs.shape, (3, 5))
        # The padded columns should be NaN
        self.assertTrue(torch.isnan(EigVecs[:, 3:]).all())
        print("...Passed")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
