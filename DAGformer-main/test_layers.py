import unittest
import torch
from torch_geometric.data import Data, Batch

# We assume the layers are defined in a file named `layers.py`
try:
    from layers import StructureExtractor, Attention, TransformerEncoderLayer
except ImportError:
    print("Could not find 'layers.py'.")
    print("Please ensure the file containing the layer classes is in the same directory.")
    # Define dummy classes to allow the test suite to be defined without errors.
    StructureExtractor = torch.nn.Module
    Attention = torch.nn.Module
    TransformerEncoderLayer = torch.nn.Module

class TestGraphLayers(unittest.TestCase):
    """
    Unit tests for the custom Transformer layers for graph-structured data.
    """

    def setUp(self):
        """Set up common data and configurations for the tests."""
        self.embed_dim = 32
        self.num_heads = 4
        self.num_nodes = 5
        self.batch_size = 2

        # --- Create a single graph data object ---
        # Structure: 0 -> 1 -> 3; 0 -> 2 -> 3 -> 4
        edge_index = torch.tensor([[0, 0, 1, 2, 3], [1, 2, 3, 3, 4]], dtype=torch.long)
        self.single_data = {
            "x": torch.randn(self.num_nodes, self.embed_dim),
            "edge_index": edge_index,
            "edge_attr": torch.randn(edge_index.shape[1], self.embed_dim),
            "mask_dag_": torch.randint(0, 2, (self.num_nodes, self.num_nodes), dtype=torch.bool),
            "ptr": torch.tensor([0, self.num_nodes]),
            # Dummy args that are passed but might not be used in all layers
            "SAT": True, "abs_pe_type": "dagpe", "abs_pe": None, "dag_rr_edge_index": None,
        }

        # --- Create a batched graph data object ---
        total_nodes = self.num_nodes * self.batch_size
        self.batched_data = {
            "x": torch.randn(total_nodes, self.embed_dim),
            "edge_index": torch.cat([edge_index, edge_index + self.num_nodes], dim=1),
            "edge_attr": torch.randn(edge_index.shape[1] * 2, self.embed_dim),
            "mask_dag_": torch.randint(0, 2, (self.batch_size, self.num_nodes, self.num_nodes), dtype=torch.bool),
            "ptr": torch.tensor([0, self.num_nodes, total_nodes]),
            "SAT": True, "abs_pe_type": "dagpe", "abs_pe": None, "dag_rr_edge_index": None,
        }

    def test_structure_extractor(self):
        """Tests the StructureExtractor layer for correct forward pass and output shape."""
        print("\nRunning test: StructureExtractor")
        try:
            # Test with GIN-like layer
            extractor_gin = StructureExtractor(embed_dim=self.embed_dim, gnn_type='gin', num_layers=2)
            output = extractor_gin(self.single_data['x'], self.single_data['edge_index'])
            self.assertEqual(output.shape, (self.num_nodes, self.embed_dim))

            # Test with concat=True
            extractor_concat = StructureExtractor(embed_dim=self.embed_dim, num_layers=2, concat=True)
            output_concat = extractor_concat(self.single_data['x'], self.single_data['edge_index'])
            self.assertEqual(output_concat.shape, (self.num_nodes, self.embed_dim))

            # Test with an edge-aware GNN
            extractor_edge = StructureExtractor(embed_dim=self.embed_dim, gnn_type='gatedgcn', num_layers=2)
            output_edge = extractor_edge(self.single_data['x'], self.single_data['edge_index'], self.single_data['edge_attr'])
            self.assertEqual(output_edge.shape, (self.num_nodes, self.embed_dim))

        except Exception as e:
            self.fail(f"StructureExtractor failed with error: {e}")
        print("...Passed")

    def test_attention_layer(self):
        """Tests the Attention layer for correct forward pass, shapes, and configurations."""
        print("\nRunning test: Attention Layer")
        try:
            # Test with standard configuration
            attention_layer = Attention(embed_dim=self.embed_dim, num_heads=self.num_heads)
            output, attn_weights = attention_layer(**self.batched_data, return_attn=True)
            
            self.assertEqual(output.shape, (self.num_nodes * self.batch_size, self.embed_dim))
            self.assertEqual(attn_weights.shape, (self.batch_size, self.num_heads, self.num_nodes, self.num_nodes))

            # Test with SAT=False (no structure extraction)
            attention_layer_no_sat = Attention(embed_dim=self.embed_dim, num_heads=self.num_heads)
            data_no_sat = self.batched_data.copy()
            data_no_sat['SAT'] = False
            output_no_sat, _ = attention_layer_no_sat(**data_no_sat)
            self.assertEqual(output_no_sat.shape, (self.num_nodes * self.batch_size, self.embed_dim))

            # Test symmetric attention
            attention_layer_sym = Attention(embed_dim=self.embed_dim, num_heads=self.num_heads, symmetric=True)
            output_sym, _ = attention_layer_sym(**self.batched_data)
            self.assertEqual(output_sym.shape, (self.num_nodes * self.batch_size, self.embed_dim))

        except Exception as e:
            self.fail(f"Attention layer failed with error: {e}")
        print("...Passed")
        
    def test_transformer_encoder_layer(self):
        """Tests the full TransformerEncoderLayer for correct forward pass and shapes."""
        print("\nRunning test: TransformerEncoderLayer")
        try:
            # Test with default (post-norm)
            encoder_layer = TransformerEncoderLayer(d_model=self.embed_dim, nhead=self.num_heads)
            output = encoder_layer(**self.batched_data)
            self.assertEqual(output.shape, (self.num_nodes * self.batch_size, self.embed_dim))

            # Test with pre-norm
            encoder_layer_prenorm = TransformerEncoderLayer(d_model=self.embed_dim, nhead=self.num_heads, pre_norm=True)
            output_prenorm = encoder_layer_prenorm(**self.batched_data)
            self.assertEqual(output_prenorm.shape, (self.num_nodes * self.batch_size, self.embed_dim))

            # Test without batch norm
            encoder_layer_no_bn = TransformerEncoderLayer(d_model=self.embed_dim, nhead=self.num_heads, batch_norm=False)
            output_no_bn = encoder_layer_no_bn(**self.batched_data)
            self.assertEqual(output_no_bn.shape, (self.num_nodes * self.batch_size, self.embed_dim))

        except Exception as e:
            self.fail(f"TransformerEncoderLayer failed with error: {e}")
        print("...Passed")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
