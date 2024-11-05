from ART_AIG import AIGTransformerEncoder
from ARC_AIG import AIGARCDecoder
import pickle
import torch

# Load the .pkl file containing all graphs
with open("/Users/bellavg/AIG_GEN/aig-gen/data/all_graphs_as_triples.pkl", "rb") as f:
    all_graphs = pickle.load(f)



# Extract data for the first graph
first_graph = all_graphs[0]
triples = first_graph["triples"]
input_mapping = first_graph["input_mapping"]
output_mapping = first_graph["output_mapping"]
gate_id_mapping = first_graph["gate_id_mapping"]
num_input_nodes = first_graph["num_input_nodes"]
num_output_nodes = first_graph["num_output_nodes"]
num_intermediate_nodes = first_graph["num_gates"]

# Convert triples to the required PyTorch format
triples_tensor = [
    (src_id, src_type, rel, dst_id, dst_type)
    for (src_id, src_type, rel, dst_id, dst_type) in triples
]


# Model parameters
embed_dim = 64
num_heads = 2
num_layers = 1
max_seq_len = len(triples)

# Instantiate the AIGTransformerEncoder model
encoder_model = AIGTransformerEncoder(embed_dim, num_intermediate_nodes, num_heads=num_heads, num_layers=num_layers)

# Run a forward pass through the encoder with dynamic input/output mappings and gate ID mapping
encoder_output = encoder_model(
    triples_tensor,
    num_input_nodes=num_input_nodes,
    num_output_nodes=num_output_nodes,
    input_mapping=input_mapping,
    output_mapping=output_mapping,
    gate_id_mapping=gate_id_mapping
)

print("Testing Encoder:")
print(f"Number of Triples: {len(triples)}")
print("Encoder Output:", encoder_output)
print("Encoder Output Size (before unsqueeze):", encoder_output.size())

# Add a batch dimension to the encoder output to make it compatible with the decoder
encoder_output = encoder_output.unsqueeze(0)  # Shape: [1, seq_length, embed_dim]
print("Encoder Output Size (after unsqueeze):", encoder_output.size())
# Instantiate the AIGARCDecoder model
decoder_model = AIGARCDecoder(embed_dim, num_heads=num_heads, num_layers=num_layers)

# Initialize a dummy target sequence with a start token for autoregressive decoding
target_seq = torch.zeros(1, 1, embed_dim)  # Start sequence for decoding, batch_size=1

# Testing the decoder by generating a series of triples based on the encoder output
decoded_triples = []
print("\nTesting Decoder:")
for i in range(max_seq_len):  # Assuming max_seq_len is the desired decoding length
    tgt_mask = torch.tril(torch.ones((i + 1, i + 1))).type(torch.bool)  # Mask for autoregressive decoding

    # Run decoder model to predict triples
    source_logits, relation_logits, destination_logits, triple_scores = decoder_model(
        encoder_output, target_seq, tgt_mask=tgt_mask
    )

    # Decode the predicted logits to actual triple components
    source_node = torch.argmax(source_logits[:, -1], dim=-1).item()
    relation = torch.argmax(relation_logits[:, -1], dim=-1).item()  # 0: Regular, 1: Inverted
    destination_node = torch.argmax(destination_logits[:, -1], dim=-1).item()

    # Append the predicted triple (source, relation, destination) with score
    decoded_triples.append((source_node, relation, destination_node, triple_scores[:, -1].item()))

    # Update target_seq with the latest predicted embedding for continuity
    new_triple_embedding = encoder_output[:, -1, :]  # Add last output embedding for sequential input
    target_seq = torch.cat((target_seq, new_triple_embedding.unsqueeze(1)), dim=1)

# Print out the results of the decoder
print("Decoded Triples:")
for i, (src, rel, dst, score) in enumerate(decoded_triples):
    print(f"Triple {i + 1}: Source={src}, Relation={'Inverted' if rel else 'Regular'}, Destination={dst}, Score={score}")
