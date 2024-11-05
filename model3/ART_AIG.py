from triple_embedding import AIGTripleEmbeddingLayer
import torch.nn as nn

class AIGTransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_intermediate_nodes, num_heads=1, num_layers=1):
        super(AIGTransformerEncoder, self).__init__()

        # Triplet Embedding Layer to handle (source, relationship, destination) embeddings
        self.triple_embedding_layer = AIGTripleEmbeddingLayer(
            embed_dim=embed_dim, num_gate_nodes=num_intermediate_nodes
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer to produce the final embeddings
        self.output_layer = nn.Linear(embed_dim, embed_dim)

    def forward(self, triples, num_input_nodes, num_output_nodes, input_mapping, output_mapping, gate_id_mapping):
        # Generate embedded triples using the mappings and node counts provided in the forward pass
        triple_embeddings = self.triple_embedding_layer(
            triples, num_input_nodes=num_input_nodes, num_output_nodes=num_output_nodes,
            input_mapping=input_mapping, output_mapping=output_mapping, gate_id_mapping=gate_id_mapping
        )

        # Reshape to fit into Transformer (sequence_length, batch_size, embed_dim)
        triple_embeddings = triple_embeddings.view(triple_embeddings.size(0), 1, -1)  # [num_triplets, 1, embed_dim]

        # Pass through Transformer encoder
        x = self.transformer_encoder(triple_embeddings)

        # Output layer for any downstream task
        x = self.output_layer(x.squeeze(1))

        return x  # Final output

# import pickle
# import torch
#
# # Load the .pkl file containing all graphs
# with open("/Users/bellavg/AIG_GEN/aig-gen/data/all_graphs_as_triples.pkl", "rb") as f:
#     all_graphs = pickle.load(f)
#
# # Extract data for the first graph
# first_graph = all_graphs[0]
# triples = first_graph["triples"]
# input_mapping = first_graph["input_mapping"]
# output_mapping = first_graph["output_mapping"]
# gate_id_mapping = first_graph["gate_id_mapping"]
# num_input_nodes = first_graph["num_input_nodes"]
# num_output_nodes = first_graph["num_output_nodes"]
# num_intermediate_nodes = first_graph["num_gates"]
#
# # Model parameters
# embed_dim = 64
# num_heads = 2
# num_layers = 1
#
# # Instantiate the AIGTransformerEncoder model
# model = AIGTransformerEncoder(embed_dim, num_intermediate_nodes, num_heads=num_heads, num_layers=num_layers)
#
# # Convert triples to the required PyTorch format
# triples_tensor = [
#     (src_id, src_type, rel, dst_id, dst_type)
#     for (src_id, src_type, rel, dst_id, dst_type) in triples
# ]
#
# # Run a forward pass through the model with dynamic input/output mappings and gate ID mapping
# output = model(
#     triples_tensor,
#     num_input_nodes=num_input_nodes,
#     num_output_nodes=num_output_nodes,
#     input_mapping=input_mapping,
#     output_mapping=output_mapping,
#     gate_id_mapping=gate_id_mapping
# )
# print(len(triples))
# print("Transformer Encoder Output:", output)
# print(output.size())