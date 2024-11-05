import torch
import torch.nn as nn


# Function to generate a fixed sinusoidal embedding for a given index
def generate_fixed_embedding(embed_dim, index):
    encoding = torch.zeros(embed_dim)
    position = torch.tensor([index])
    div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
    encoding[0::2] = torch.sin(position * div_term)
    encoding[1::2] = torch.cos(position * div_term)
    return encoding


class AIGTripleEmbeddingLayer(nn.Module):
    def __init__(self, embed_dim, num_gate_nodes):
        super(AIGTripleEmbeddingLayer, self).__init__()

        self.embed_dim = embed_dim

        # Embedding layer for gate nodes and edge types
        self.gate_embeddings = nn.Embedding(num_gate_nodes, embed_dim)
        self.edge_type_embeddings = nn.Embedding(2, embed_dim)  # 2 edge types: regular and inverted

        # Linear layer to reduce concatenated embedding back to embed_dim
        self.linear_layer = nn.Linear(3 * embed_dim, embed_dim)

    def generate_input_output_embeddings(self, num_input_nodes, num_output_nodes):
        """Generate embeddings for inputs and outputs based on the graph's input/output count."""
        self.input_encodings = torch.stack(
            [generate_fixed_embedding(self.embed_dim, i) for i in range(num_input_nodes)]
        )
        self.output_encodings = torch.stack(
            [generate_fixed_embedding(self.embed_dim, i) for i in range(num_output_nodes)]
        )

    def set_gate_id_mapping(self, gate_node_ids):
        """Create a mapping for gate node IDs to a sequential index range."""
        self.gate_id_mapping = {node_id: idx for idx, node_id in enumerate(sorted(gate_node_ids))}
        self.num_gate_nodes = len(gate_node_ids)

    def get_node_embedding(self, node_id, node_type, input_mapping=None, output_mapping=None):
        if node_type == "input":
            logical_input_order = input_mapping.get(node_id, None)
            if logical_input_order is None or logical_input_order >= len(self.input_encodings):
                raise IndexError(f"node_id {node_id} is out of bounds for input nodes or not found in input_mapping")
            return self.input_encodings[logical_input_order]
        elif node_type == "output":
            logical_output_order = output_mapping.get(node_id, None)
            if logical_output_order is None or logical_output_order >= len(self.output_encodings):
                raise IndexError(f"node_id {node_id} is out of bounds for output nodes or not found in output_mapping")
            return self.output_encodings[logical_output_order]
        elif node_type == "gate":
            mapped_id = self.gate_id_mapping.get(node_id)
            if mapped_id is None or mapped_id >= self.num_gate_nodes:
                raise IndexError(f"node_id {node_id} is out of bounds or not found in gate id mapping")
            return self.gate_embeddings(torch.tensor([mapped_id], dtype=torch.long)).squeeze(0)
        elif node_type == "constant":
            return torch.zeros_like(self.gate_embeddings.weight[0])

    def forward(self, triples, num_input_nodes, num_output_nodes, input_mapping=None, output_mapping=None,
                gate_id_mapping=None):
        # Generate input and output embeddings for this specific graph
        self.generate_input_output_embeddings(num_input_nodes, num_output_nodes)

        # Assign the gate ID mapping for the graph
        if gate_id_mapping is not None:
            self.set_gate_id_mapping(gate_id_mapping.keys())

        embedded_triplets = []
        for (source_id, source_type, relationship, dest_id, dest_type) in triples:
            source_embedding = self.get_node_embedding(source_id, source_type, input_mapping, output_mapping)
            dest_embedding = self.get_node_embedding(dest_id, dest_type, input_mapping, output_mapping)
            relationship_embedding = self.edge_type_embeddings(torch.tensor([relationship], dtype=torch.long)).squeeze(
                0)
            triplet_embedding = torch.cat([source_embedding, relationship_embedding, dest_embedding], dim=-1)
            triplet_embedding = self.linear_layer(triplet_embedding)  # Reduce to embed_dim
            embedded_triplets.append(triplet_embedding)
        embedded_triplets = torch.stack(embedded_triplets)
        return embedded_triplets  # Shape: [num_triplets, embed_dim]
#
#
# # Dummy Data for Testing with Output Mapping
# embed_dim = 64
# num_input_nodes = 2   # Example: 2 ordered input nodes
# num_output_nodes = 3  # Example: 3 ordered output nodes
# num_gate_nodes = 5  # Example number of gate nodes
#
# # Instantiate the AIGTripleEmbeddingLayer with sinusoidal embeddings for inputs and outputs
# embedding_layer = AIGTripleEmbeddingLayer(num_input_nodes, num_output_nodes, num_gate_nodes, embed_dim)
#
# # Define an output mapping (actual node ID -> logical output order)
# output_mapping = {
#     15: 0,  # Node ID 15 is "Output 1"
#     8: 1,   # Node ID 8 is "Output 2"
#     12: 2   # Node ID 12 is "Output 3"
# }
#
# # Example triples with logical output ordering enforced through output_mapping
# triples = [
#     (0, "constant", 0, 2, "gate"),      # Constant 0 -> gate
#     (1, "input", 1, 3, "gate"),      # Input 1 -> gate
#     (2, "gate", 0, 15, "output"),    # gate -> Output 1 (Node ID 15)
#     (3, "gate", 1, 8, "output"),     # gate -> Output 2 (Node ID 8)
#     (4, "gate", 0, 12, "output")     # gate -> Output 3 (Node ID 12)
# ]
#
# # Run a forward pass through the embedding layer with triples and output_mapping
# triple_embeddings = embedding_layer(triples, output_mapping=output_mapping)
# print("Triple Embeddings with Logical Output Order:", triple_embeddings)
