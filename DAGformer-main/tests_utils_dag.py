import unittest
import torch
from torch_geometric.data import Data

# Import the function to be tested.
# This assumes the simplified 'add_order_info' is in a file named 'utils_dag.py'
# in the same directory.
try:
    from utils_dag_simplified import add_order_info
except ImportError:
    print("Could not find 'utils_dag_simplified.py'. Please ensure the file with the function to test is accessible.")
    # As a fallback, define a dummy function to avoid crashing the test suite definition.
    def add_order_info(data):
        return data

class TestAddOrderInfo(unittest.TestCase):
    """
    Unit tests for the add_order_info function to verify the correctness of
    'abs_pe' (absolute positional encoding) and 'mask_rc' (causality mask).
    """

    def test_simple_line_graph(self):
        """Tests a simple directed line graph: 0 -> 1 -> 2."""
        print("\nRunning test: Simple Line Graph")
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t().contiguous()
        data = Data(edge_index=edge_index, num_nodes=3)
        
        # Run the function
        processed_data = add_order_info(data)
        
        # --- Verification ---
        # Expected absolute positional encoding (longest path from a root)
        expected_abs_pe = torch.tensor([0, 1, 2], dtype=torch.long)
        
        # Expected causality mask (a node can attend to its ancestors and itself)
        # Rows are targets, columns are sources. mask_rc[i, j] is True if j is an ancestor of i.
        expected_mask_rc = torch.tensor([
            [True, False, False],  # Node 0 attends to Node 0
            [True, True,  False],  # Node 1 attends to Nodes 0, 1
            [True, True,  True]   # Node 2 attends to Nodes 0, 1, 2
        ], dtype=torch.bool)
        
        self.assertTrue(torch.equal(processed_data.abs_pe, expected_abs_pe), "abs_pe is incorrect for the line graph.")
        self.assertTrue(torch.equal(processed_data.mask_rc, expected_mask_rc), "mask_rc is incorrect for the line graph.")
        print("...Passed")

    def test_graph_with_multiple_roots(self):
        """Tests a graph with two roots merging into one node: 0 -> 2, 1 -> 2."""
        print("\nRunning test: Graph with Multiple Roots")
        edge_index = torch.tensor([[0, 1], [2, 2]], dtype=torch.long).t().contiguous()
        data = Data(edge_index=edge_index, num_nodes=3)
        
        processed_data = add_order_info(data)
        
        # --- Verification ---
        # Roots 0 and 1 have pe=0. Node 2 is at pe=1.
        expected_abs_pe = torch.tensor([0, 0, 1], dtype=torch.long)
        
        expected_mask_rc = torch.tensor([
            [True, False, False], # Node 0 attends to 0
            [False, True, False], # Node 1 attends to 1
            [True, True, True]    # Node 2 attends to 0, 1, 2
        ], dtype=torch.bool)
        
        self.assertTrue(torch.equal(processed_data.abs_pe, expected_abs_pe), "abs_pe is incorrect for the multi-root graph.")
        self.assertTrue(torch.equal(processed_data.mask_rc, expected_mask_rc), "mask_rc is incorrect for the multi-root graph.")
        print("...Passed")

    def test_complex_dag(self):
        """
        Tests a more complex DAG structure:
          0 -> 1 -> 3
            \-> 2 -> 3 -> 4
        """
        print("\nRunning test: Complex DAG")
        edge_index = torch.tensor([[0, 0, 1, 2, 3], [1, 2, 3, 3, 4]], dtype=torch.long).t().contiguous()
        data = Data(edge_index=edge_index, num_nodes=5)

        processed_data = add_order_info(data)

        # --- Verification ---
        # Longest paths:
        # 0: 0 (root)
        # 1: 1 (0->1)
        # 2: 1 (0->2)
        # 3: 2 (0->1->3 or 0->2->3)
        # 4: 3 (0->2->3->4)
        expected_abs_pe = torch.tensor([0, 1, 1, 2, 3], dtype=torch.long)

        expected_mask_rc = torch.tensor([
            [True, False, False, False, False], # 0 <- {0}
            [True, True,  False, False, False], # 1 <- {0, 1}
            [True, False, True,  False, False], # 2 <- {0, 2}
            [True, True,  True,  True,  False], # 3 <- {0, 1, 2, 3}
            [True, True,  True,  True,  True]   # 4 <- {0, 1, 2, 3, 4}
        ], dtype=torch.bool)

        self.assertTrue(torch.equal(processed_data.abs_pe, expected_abs_pe), "abs_pe is incorrect for the complex DAG.")
        self.assertTrue(torch.equal(processed_data.mask_rc, expected_mask_rc), "mask_rc is incorrect for the complex DAG.")
        print("...Passed")

    def test_disconnected_graph(self):
        """Tests a graph with two disconnected components: 0 -> 1 and 2 -> 3."""
        print("\nRunning test: Disconnected Graph")
        edge_index = torch.tensor([[0, 2], [1, 3]], dtype=torch.long).t().contiguous()
        data = Data(edge_index=edge_index, num_nodes=4)

        processed_data = add_order_info(data)

        # --- Verification ---
        # Component 1: 0->1 -> pe: 0, 1
        # Component 2: 2->3 -> pe: 0, 1
        expected_abs_pe = torch.tensor([0, 1, 0, 1], dtype=torch.long)

        expected_mask_rc = torch.tensor([
            [True, False, False, False], # 0 <- {0}
            [True, True,  False, False], # 1 <- {0, 1}
            [False, False, True, False], # 2 <- {2}
            [False, False, True, True]   # 3 <- {2, 3}
        ], dtype=torch.bool)

        self.assertTrue(torch.equal(processed_data.abs_pe, expected_abs_pe), "abs_pe is incorrect for the disconnected graph.")
        self.assertTrue(torch.equal(processed_data.mask_rc, expected_mask_rc), "mask_rc is incorrect for the disconnected graph.")
        print("...Passed")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
