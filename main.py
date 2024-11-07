import pickle
from model3.ARC_AIG import ARCDecoder
from model3.ART_AIG import AIGTransformerEncoder
from model3.aig_utils import  cleanup_dangling_triples, generate_binary_inputs
from model3.model_utils import get_prior, aggregate_source_embeddings
from model3.generator import generate_triples_with_arc
from model3.diff_aig import DifferentiableAIGModelFromTriples
import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F



# Load the .pkl file containing all graphs
with open("/Users/bellavg/AIG_GEN/aig-gen/data/all_graphs_as_triples.pkl", "rb") as f:
    all_graphs = pickle.load(f)

embed_dim = 128
heads = 2
layers = 4

# Training Loop
# Initialize encoder and decoder models
encoder_model = AIGTransformerEncoder(embed_dim=embed_dim, num_heads=heads, num_layers=layers)
arc_decoder = ARCDecoder(embed_dim=embed_dim, num_relationships=4)

# Define optimizer
optimizer = optim.Adam(list(encoder_model.parameters()) + list(arc_decoder.parameters()), lr=0.001)
criterion = nn.BCELoss()

# Compute prior probabilities
p_R = get_prior(all_graphs) # prior for p_R[1], p_R[2], p_R[-1], p_R[-2]

#TODO: add batches

# Loop over each graph in the dataset
for graph in all_graphs:
    triples = graph["triples"]
    input_mapping = graph["input_mapping"]
    output_mapping = graph["output_mapping"]
    gate_id_mapping = graph["gate_id_mapping"]
    num_input_nodes = graph["num_input_nodes"]
    num_output_nodes = graph["num_output_nodes"]
    num_gate_nodes = graph["num_gates"]

    # Assuming `precomputed_outputs` is the dictionary returned by precompute_aig_outputs
    true_outputs = graph["true_outputs"]

    # Prepare input and output nodes
    input_nodes = list(input_mapping.keys())
    output_nodes = list(output_mapping.keys())

    # Encode the graph using ART (AIGTransformerEncoder)
    encoder_output, source_to_triplet_indices = encoder_model(
        triples,
        num_input_nodes=num_input_nodes,
        num_output_nodes=num_output_nodes,
        num_intermediate_nodes=num_gate_nodes,
        input_mapping=input_mapping,
        output_mapping=output_mapping,
        gate_id_mapping=gate_id_mapping
    )

    # Total number of nodes in the graph
    num_nodes = num_input_nodes + num_gate_nodes + num_output_nodes

    # Aggregate source embeddings
    aggregated_source_embeddings = aggregate_source_embeddings(encoder_output, source_to_triplet_indices)

    # Generate triples using ARCDecoder
    generated_triples = generate_triples_with_arc(
        aggregated_source_embeddings=aggregated_source_embeddings,
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        arc_decoder=arc_decoder,
        num_nodes=num_nodes,
        p_R=p_R
    )

    # build into aig object and get rid of disconnected subgraphs
    #gen_aig = build_aig_from_triples(generated_triples, input_nodes, output_nodes)
    # get rid of disconnected subgraphs
    generated_triples = cleanup_dangling_triples(generated_triples, input_nodes, output_nodes)

    # Convert the AIG to a differentiable model
    diff_aig_model = DifferentiableAIGModelFromTriples(generated_triples, input_nodes, output_nodes)

    # Use the differentiable AIG model in your training loop
    input_patterns = torch.tensor(generate_binary_inputs(num_input_nodes), dtype=torch.float32)
    target_outputs = torch.tensor(true_outputs, dtype=torch.float32)

    # Zero gradients
    optimizer.zero_grad()

    # Forward pass through differentiable AIG
    predicted_outputs = diff_aig_model(input_patterns)

    # Compute loss
    loss = criterion(predicted_outputs, target_outputs)

    # Backward pass
    loss.backward()

    # Update parameters
    optimizer.step()


    #TODO:
    # arg parser
    # Every N steps get accuracy
    # binary_outputs = (model_outputs >= 0.5).float()
    # correct = (binary_outputs == target_outputs).all(dim=1).float()
    # accuracy = correct.mean().item()
    # print(f"Functional Equivalence Rate: {accuracy * 100:.2f}%")
    # Structural difference
    # # Compute expected structural size from decoder outputs
    # edge_probs = decoder_output['edge_probs']  # Obtain from your decoder
    # expected_edge_count = torch.sum(edge_probs)
    # # Original edge count (as a constant tensor)
    # orig_edge_count_tensor = torch.tensor([orig_edge_count], dtype=torch.float32)
    # # Structural loss
    # structural_loss = F.relu(expected_edge_count - orig_edge_count_tensor)
    # # Total loss
    # lambda_weight = 1.0  # Hyperparameter to adjust
    # total_loss = functional_loss + lambda_weight * structural_loss


    #TODO: add training and testing loop, batches

