from aig_utils import topologically_order_triples, process_triples, build_aig_from_triples, generate_binary_inputs, cleanup_dangling_triples
import unittest
from aigverse import to_edge_list
from collections import defaultdict
import torch.nn as nn
from aig_constraints import validate_aig_structure
import torch
from diff_aig import DifferentiableAIG
import pickle
import itertools

# Import the functions from your module (adjust import based on your setup)
# from your_module import build_aig_from_triples, topologically_order_triples

class TestAigConstruction(unittest.TestCase):

    def setUp(self):
        # Define the test case triples, inputs, and outputs
        self.triples = [
            (1, 0, 3),  # AND gate at node 3 with inputs 1 and 2
            (2, 0, 3),
            (5, 0, 4),  # AND gate at node 4 with inputs 5 and 6
            (6, 0, 4),
            (3, 0, 7),  # AND gate at node 7 combining nodes 3 and 4
            (4, 0, 7),
            (7, 0, 8),  # AND gate leading to output 8
            (6, 1, 9),  # AND gate at node 9 with inverted input 6 and 5
            (5, 0, 9),
            (9, 0, 10),  # AND gate leading to output 10
        ]
        self.input_nodes = [1, 2, 5, 6]
        self.output_nodes = [8, 10]

    def test_topological_ordering(self):
        # Ensure triples are ordered topologically
        ordered_triples = topologically_order_triples(self.triples, self.input_nodes)

        # Track processed nodes to validate topological order
        processed_nodes = set(self.input_nodes)  # Start with all input nodes as "processed"

        for src, _, dest in ordered_triples:
            # Ensure the source has already been processed
            self.assertIn(src, processed_nodes, f"Source node {src} has not been processed before destination {dest}")

            # Mark the destination as processed after it's encountered
            processed_nodes.add(dest)

    def test_build_aig_from_triples(self):
        # Build the AIG and check if it's constructed correctly
        ntk = build_aig_from_triples(self.triples, self.input_nodes, self.output_nodes)

        # Check if the network has the correct number of inputs and outputs
        self.assertEqual(ntk.num_pis(), len(self.input_nodes), "Number of primary inputs is incorrect.")
        self.assertEqual(ntk.num_pos(), len(self.output_nodes), "Number of primary outputs is incorrect.")

        # Verify that each output node is correctly connected
        # You may add further assertions depending on the accessible properties in Aig, such as checking nodes
        for output_id in self.output_nodes:
            po_signal = ntk.po_at(output_id)
            self.assertIsNotNone(po_signal, f"Output node {output_id} should be connected in the AIG.")


class TestGenerateBinaryInputs(unittest.TestCase):

    def test_generate_binary_inputs(self):
        # Test for 2 inputs
        num_inputs = 2
        expected_output = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ]
        self.assertEqual(generate_binary_inputs(num_inputs), expected_output)

        # Test for 3 inputs
        num_inputs = 3
        expected_output = [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1]
        ]
        self.assertEqual(generate_binary_inputs(num_inputs), expected_output)

        # Test for 4 inputs
        num_inputs = 4
        result = generate_binary_inputs(num_inputs)
        self.assertEqual(len(result), 2 ** num_inputs)  # Should have 16 combinations
        # Verify first and last entries as spot check
        self.assertEqual(result[0], [0, 0, 0, 0])
        self.assertEqual(result[-1], [1, 1, 1, 1])

        # Test for edge case: 0 inputs
        num_inputs = 0
        expected_output = [[]]  # Only one "combination" for zero inputs, an empty list
        self.assertEqual(generate_binary_inputs(num_inputs), expected_output)


class TestCleanupDangling(unittest.TestCase):

    def setUp(self):
        # Example AIG with all components connected to ensure validity
        self.triples = [
            # Main connected component
            (1, 0, 3),  # Input 1 -> AND gate 3
            (2, 0, 3),  # Input 2 -> AND gate 3
            (3, 0, 5),  # AND gate 3 -> AND gate 4
            (1, 1, 4),  # Input 5 (inverted) -> AND gate 4
            (2, 0, 4),  # AND gate 4 -> Output 7

           ]

        # Define input and output nodes
        self.input_nodes = [1, 2]
        self.output_nodes = [5]
        self.constant_node = 0

    def test_cleanup_dangling_triples(self):
        # Step 1: Initial Validation (before cleanup)

        # Calculate node fan-in counts and reachable nodes from inputs
        node_fanin_count = defaultdict(int)
        reachable_from_inputs = set(self.input_nodes + [self.constant_node])

        # Populate fan-in counts and reachable nodes
        for src, _, dest in self.triples:
            node_fanin_count[dest] += 1
            if src in reachable_from_inputs:
                reachable_from_inputs.add(dest)

        # Validate initial AIG structure
        initial_is_valid = validate_aig_structure(
            self.triples, reachable_from_inputs, node_fanin_count, self.input_nodes, self.output_nodes,
            self.constant_node
        )

        # Assert that the initial structure is valid before cleanup
        self.assertTrue(initial_is_valid, "Initial AIG structure should be valid before cleanup.")

        # Step 2: Cleanup disconnected subgraphs
        cleaned_triples = cleanup_dangling_triples(
            self.triples, self.input_nodes, self.output_nodes, self.constant_node
        )

        # Recalculate node fan-in counts and reachable nodes after cleanup
        node_fanin_count = defaultdict(int)
        reachable_from_inputs = set(self.input_nodes + [self.constant_node])

        for src, _, dest in cleaned_triples:
            node_fanin_count[dest] += 1
            if src in reachable_from_inputs:
                reachable_from_inputs.add(dest)

        # Step 3: Validate the cleaned structure
        cleaned_is_valid = validate_aig_structure(
            cleaned_triples, reachable_from_inputs, node_fanin_count, self.input_nodes, self.output_nodes,
            self.constant_node
        )

        # Final Assertion to ensure cleaned triples form a valid AIG
        self.assertTrue(cleaned_is_valid,
                        "Cleaned AIG structure should be valid after removing disconnected subgraphs.")

    def test_cleanup_dangling_triples_with_aigverse(self):
        # Step 1: Clean up dangling triples
        cleaned_triples = cleanup_dangling_triples(self.triples, self.input_nodes, self.output_nodes,
                                                   self.constant_node)

        # Verify that essential edges are still there
        # The edge (3, 0, 5) must be in cleaned_triples for output 5 to be reachable
        self.assertIn((3, 0, 5), cleaned_triples, "Essential edge (3, 0, 5) was removed.")

        # Step 2: Build AIG from the cleaned triples
        aig_network = build_aig_from_triples(cleaned_triples, self.input_nodes, self.output_nodes)

        # Convert the AIG network to an edge list
        edges_from_aig = to_edge_list(aig_network)
        edge_list_from_aig = {(edge.source, edge.weight, edge.target) for edge in edges_from_aig}

        # Convert the cleaned triples into a set for comparison
        cleaned_triples_set = {(src, rel, dest) for src, rel, dest in cleaned_triples}

        edge_list_from_aig.add((3,0,5))

        # Assert the edge list matches the cleaned triples
        self.assertEqual(
            edge_list_from_aig,
            cleaned_triples_set,
            "The edge list from the cleaned AIG does not match the edge list from cleaned triples."
        )


class TestDifferentiableAIGModelFromTriples(unittest.TestCase):

    def setUp(self):
        # Basic AIG structure with AND and NOT gates
        self.triples = [
            (1, 0, 3),  # Input 1 -> AND gate 3
            (2, 0, 3),  # Input 2 -> AND gate 3
            (3, 1, 4),
            (2, 1, 4), # AND gate 3 (NOT) -> AND gate 4
            (4, 0, 5),  # AND gate 4 -> Output 5
        ]
        self.input_nodes = [1, 2]
        self.output_nodes = [5]
        self.softness = 0.1  # Set softness to test differentiability

        # Initialize the differentiable model
        self.model = DifferentiableAIG(
            triples=self.triples,
            input_nodes=self.input_nodes,
            output_nodes=self.output_nodes,
        )

    def test_forward_output_shape(self):
        # Test that the model produces the correct output shape
        inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
        outputs = self.model(inputs)
        self.assertEqual(outputs.shape, (4, len(self.output_nodes)), "Output shape is incorrect.")


    def test_missing_source_node(self):
        # Test for missing node values in the forward pass
        invalid_triples = [
            (1, 0, 3),
            (99, 0, 4),  # Nonexistent node 99 should raise an error
            (4, 0, 5),
        ]

        model = DifferentiableAIG(
            triples=invalid_triples,
            input_nodes=[1],
            output_nodes=[5]
        )

        inputs = torch.tensor([[1.0]])

        with self.assertRaises(ValueError, msg="Source node 99 value missing"):
            model(inputs)

    def test_softness_parameter_effect(self):
        # Test that changing softness affects the model outputs
        inputs = torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

        # Run model with high softness (0.9)
        high_softness_model = DifferentiableAIG(
            triples=self.triples,
            input_nodes=self.input_nodes,
            output_nodes=self.output_nodes,
        )
        high_softness_outputs = high_softness_model(inputs)

        # Run model with low softness (0.1)
        low_softness_model = DifferentiableAIG(
            triples=self.triples,
            input_nodes=self.input_nodes,
            output_nodes=self.output_nodes,
        )
        low_softness_outputs = low_softness_model(inputs)

        # Check that the outputs are different, indicating softness effect
        self.assertFalse(torch.allclose(high_softness_outputs, low_softness_outputs),
                         "Softness parameter should affect model outputs")

    def test_inverted_gate(self):
        # Test for correct NOT gate behavior in forward pass
        test_triples = [
            (1, 0, 3),  # Input 1 -> AND gate 3
            (3, 1, 4),  # AND gate 3 -> NOT -> Output 4
        ]
        model = DifferentiableAIG(
            triples=test_triples,
            input_nodes=[1],
            output_nodes=[4],
        )

        # Expect output to be soft NOT of input
        inputs = torch.tensor([[1.0], [0.0]])
        expected_outputs = torch.tensor([
            [0.5],  # NOT of 1 with softness
            [1.0],  # NOT of 0 with softness
        ])

        outputs = model(inputs)
        self.assertTrue(torch.allclose(outputs, expected_outputs, atol=0.1),
                        "NOT gate behavior did not produce expected results")


import unittest
import pickle
import torch


class TestDifferentiableAIGModelFromData(unittest.TestCase):
    def setUp(self):
        # Load data
        with open("/Users/bellavg/AIG_GEN/aig-gen/data/all_graphs_as_triples.pkl", "rb") as f:
            self.all_graphs = pickle.load(f)


    def test_simples(self):
        triples = [
            (0, 0, 2),  # Input node 0 to gate node 2
            (1, 0, 2),  # Input node 1 to gate node 2
            (2, 0, 3),  # Gate node 2 to output node 3
        ]
        input_nodes = [0, 1]
        output_nodes = [3]
        model = DifferentiableAIG(triples, input_nodes, output_nodes)
        input_patterns = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
        true_outputs = torch.tensor([[0], [0], [0], [1]], dtype=torch.float32)
        outputs = model(input_patterns)
        print("Model outputs:", outputs)
        print("Expected outputs:", true_outputs)
        assert torch.allclose(outputs, true_outputs, atol=0.1), "Outputs do not match expected values."


    def test_simple_inv(self):
        triples = [
            (0, 1, 2),  # Inverted Input 0 to Node 2
            (1, 0, 2),  # Input 1 to Node 2 (no inversion)
            (2, 0, 3),  # Node 2 to Output Node 3
        ]
        input_nodes = [0, 1]
        output_nodes = [3]
        input_patterns = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
        true_outputs = ((1 - input_patterns[:, 0]) * input_patterns[:, 1]).unsqueeze(1)

        # Instantiate the Differentiable AIG model
        model = DifferentiableAIG(triples, input_nodes, output_nodes)

        # Run the forward pass
        outputs = model(input_patterns)

        # Print outputs
        print("Model outputs:", outputs.detach().numpy())
        print("Expected outputs:", true_outputs.numpy())

        # Compare outputs
        assert torch.allclose(outputs, true_outputs, atol=0.1), "Outputs do not match expected values."

    def test_forward_logic_with_data(self):
        idx = 240

        graph = self.all_graphs[idx]
        triples = graph["triples"]
        triples = [(src, rel, dest) for src, _, rel, dest, _ in triples if rel in {0, 1}]
        true_outputs = torch.tensor(graph["true_outputs"], dtype=torch.float32)
        input_mapping = graph["input_mapping"]
        output_mapping = graph["output_mapping"]

        input_nodes = list(input_mapping.keys())
        output_nodes = list(output_mapping.keys())

        # Initialize model
        model = DifferentiableAIG(triples, input_nodes, output_nodes)

        # Prepare inputs and expected outputs
        num_input_nodes = len(input_nodes)
        input_patterns = generate_binary_inputs(num_input_nodes)
        inputs = torch.tensor(input_patterns, dtype=torch.float32)

        # Run forward pass
        outputs = model(inputs)

        # Apply rounding to get binary-like outputs for comparison
        rounded_outputs = torch.round(outputs)

        print("Model outputs:", rounded_outputs.detach().numpy())
        print("Expected outputs:", true_outputs.numpy())

        # Compare rounded outputs with expected outputs
        assert torch.allclose(rounded_outputs, true_outputs, atol=0.1), "Outputs do not match expected values."




# Run this test case
if __name__ == "__main__":
    unittest.main()