import unittest
import torch
from torch_geometric.data import Data, Batch

# We assume the GraphTransformer model is in a file named `dag_transformer_model.py`
# Please adjust the import path if your file is named differently.
try:
    from dag_transformer_model import GraphTransformer
except ImportError:
    print("Could not find 'dag_transformer_model.py'.")
    print("Please ensure the file containing the GraphTransformer class is in the same directory or accessible in the python path.")
    # Define a dummy class to allow the test suite to be defined without errors.
    GraphTransformer = torch.nn.Module

class TestGraphTransformer(unittest.TestCase):
    """
    Unit tests for the GraphTransformer model to ensure its robustness
    and correctness across different configurations.
    """

    def setUp(self):
        """
        Set up common data structures to be used across multiple tests.
        This includes a single sample DAG and a batch of two such DAGs.
        """
        self.num_nodes = 5
        self.in_size = 16  # Input node feature dimension
        self.d_model = 32  # Model's hidden dimension
        self.num_edge_features = 4 # Number of possible edge types for embedding

        # --- Create a single graph data object ---
        # Graph structure: 0 -> 1 -> 3; 0 -> 2 -> 3 -> 4
        edge_index = torch.tensor([[0, 0, 1, 2, 3], [1, 2, 3, 3, 4]], dtype=torch.long)
        self.single_data = Data(
            x=torch.randn(self.num_nodes, self.in_size),
            edge_index=edge_index,
            edge_attr=torch.randint(0, self.num_edge_features, (edge_index.shape[1],)),
            abs_pe=torch.tensor([0, 1, 1, 2, 3], dtype=torch.long),
            mask_rc=torch.randint(0, 2, (self.num_nodes, self.num_nodes), dtype=torch.bool),
            # Dummy attributes that the model expects
            dag_rr_edge_index=torch.tensor([[0, 0, 0, 1, 2], [3, 4, 2, 4, 4]]),
            Eigvecs=torch.randn(self.num_nodes, 8),
            degree=torch.randint(1, 4, (self.num_nodes,)),
            num_nodes=self.num_nodes
        )

        # --- Create a batched graph data object ---
        data_list = [self.single_data, self.single_data]
        self.batched_data = Batch.from_data_list(data_list)

    def test_initialization(self):
        """Tests that the model can be initialized with different configurations without errors."""
        print("\nRunning test: Model Initialization")
        try:
            # Test default GPS mode (gps=0)
            GraphTransformer(in_size=self.in_size, d_model=self.d_model)
            # Test GPS mode 1
            GraphTransformer(in_size=self.in_size, d_model=self.d_model, gps=1)
            # Test GPS mode 2 (DAG)
            GraphTransformer(in_size=self.in_size, d_model=self.d_model, gps=2)
            # Test with edge attributes enabled
            GraphTransformer(in_size=self.in_size, d_model=self.d_model, use_edge_attr=True, num_edge_features=self.num_edge_features)
            # Test different positional encodings
            GraphTransformer(in_size=self.in_size, d_model=self.d_model, abs_pe='none')
            GraphTransformer(in_size=self.in_size, d_model=self.d_model, abs_pe='Eigvecs')
        except Exception as e:
            self.fail(f"Model initialization failed with error: {e}")
        print("...Passed")

    def test_forward_pass_and_output_shapes(self):
        """Tests the forward pass for single and batched data, verifying output shapes."""
        print("\nRunning test: Forward Pass and Output Shapes")
        model = GraphTransformer(in_size=self.in_size, d_model=self.d_model, use_edge_attr=True)
        model.eval() # Set to evaluation mode

        # --- Test with a single graph ---
        with torch.no_grad():
            output = model(self.single_data)
        self.assertEqual(output.shape, (self.num_nodes, self.in_size), "Output shape is incorrect for a single graph.")

        # --- Test with a batched graph ---
        with torch.no_grad():
            batched_output = model(self.batched_data)
        total_nodes = self.num_nodes * 2
        self.assertEqual(batched_output.shape, (total_nodes, self.in_size), "Output shape is incorrect for a batched graph.")
        print("...Passed")

    def test_all_configurations_run(self):
        """
        A comprehensive test to ensure the forward pass runs for a matrix of common configurations.
        This checks for runtime errors, which is the primary goal of unit testing models.
        """
        print("\nRunning test: All Configurations Runtime Check")
        configs = []
        for gps_mode in [0, 1, 2]:
            for abs_pe_mode in ['dagpe', 'none', 'Eigvecs']:
                for use_edge_attr_mode in [True, False]:
                    configs.append({
                        'gps': gps_mode,
                        'abs_pe': abs_pe_mode,
                        'use_edge_attr': use_edge_attr_mode
                    })
        
        for i, config in enumerate(configs):
            with self.subTest(f"Config {i+1}/{len(configs)}: {config}"):
                try:
                    model = GraphTransformer(
                        in_size=self.in_size,
                        d_model=self.d_model,
                        num_edge_features=self.num_edge_features,
                        **config
                    )
                    model.eval()
                    with torch.no_grad():
                        model(self.batched_data)
                except Exception as e:
                    self.fail(f"Model forward pass failed for config {config} with error: {e}")
        print("...Passed")


if __name__ == '__main__':
    # This allows the test script to be run directly.
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
