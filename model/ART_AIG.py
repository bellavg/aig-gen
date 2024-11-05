import torch.nn as nn
from triple_embedding import AIGTripleEmbeddingLayer


# class AIGTransformerEncoder(nn.Module):
#     def __init__(self, embed_dim, num_heads, num_layers, num_input_nodes, num_output_nodes, num_intermediate_nodes):
#         super(AIGTransformerEncoder, self).__init__()
#
#         # Triplet Embedding Layer to handle (source, relationship, destination) with attention
#         self.triple_embedding_layer = AIGTripleEmbeddingLayer(
#             num_input_nodes, num_output_nodes, num_intermediate_nodes, embed_dim
#         )
#
#         # Transformer encoder layers
#         encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#
#         # Output layer to process the encoded triples
#         self.output_layer = nn.Linear(embed_dim, embed_dim)
#
#     def forward(self, triples, output_mapping=None):
#         """
#         Forward method for the AIG Transformer Encoder with triple embeddings.
#
#         Arguments:
#         - triples: List of tuples representing (source_id, source_type, relationship, dest_id, dest_type)
#         - output_mapping: A dictionary mapping output node IDs to their logical order
#
#         Returns:
#         - Encoded representation of the graph as per the joint embedding learned by the model
#         """
#
#         # Generate embedded triples
#         triple_embeddings = self.triple_embedding_layer(triples, output_mapping=output_mapping)
#
#         # Reshape to fit into Transformer (sequence_length, batch_size, embed_dim)
#         triple_embeddings = triple_embeddings.unsqueeze(1)  # [num_triplets, 1, embed_dim]
#
#         # Pass through Transformer encoder
#         x = self.transformer_encoder(triple_embeddings)
#
#         # Apply output layer for completion/scoring tasks
#         x = self.output_layer(x.squeeze(1))  # Remove batch dimension for downstream tasks
#
#         return x


# Define the AIGTransformerEncoder using AIGTripleEmbeddingLayer
class AIGTransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, num_input_nodes, num_output_nodes, num_intermediate_nodes):
        super(AIGTransformerEncoder, self).__init__()

        self.triple_embedding_layer = AIGTripleEmbeddingLayer(
            num_input_nodes, num_output_nodes, num_intermediate_nodes, embed_dim
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(embed_dim, embed_dim)

    def forward(self, triples, output_mapping=None):
        triple_embeddings = self.triple_embedding_layer(triples, output_mapping=output_mapping)
        triple_embeddings = triple_embeddings.view(triple_embeddings.size(0), 1, -1)  # [num_triplets, 1, embed_dim * 3]
        x = self.transformer_encoder(triple_embeddings)
        x = self.output_layer(x.squeeze(1))
        return x

# Dummy Data and Model Setup
embed_dim = 64
num_heads = 4
num_layers = 2
num_input_nodes = 3
num_output_nodes = 3
num_intermediate_nodes = 5

# Instantiate the AIGTransformerEncoder
model = AIGTransformerEncoder(embed_dim, num_heads, num_layers, num_input_nodes, num_output_nodes, num_intermediate_nodes)

# Dummy triples with an output mapping for logical ordering
triples = [
    (0, "input", 0, 2, "intermediate"),      # Input 1 -> Intermediate
    (1, "input", 1, 3, "intermediate"),      # Input 2 -> Intermediate
    (2, "intermediate", 0, 15, "output"),    # Intermediate -> Output 1 (Node ID 15)
    (3, "intermediate", 1, 8, "output"),     # Intermediate -> Output 2 (Node ID 8)
    (4, "intermediate", 0, 12, "output")     # Intermediate -> Output 3 (Node ID 12)
]
output_mapping = {15: 0, 8: 1, 12: 2}

# Run a forward pass through the model
output = model(triples, output_mapping=output_mapping)
print("Transformer Encoder Output:", output)
