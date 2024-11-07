import torch
from typing import List
from aigverse import Aig
from typing import List, Tuple, Dict
from typing import List, Tuple, Dict
from collections import deque


def topologically_order_triples(
        triples: List[Tuple[int, int, int]],
        input_nodes: List[int]
) -> List[Tuple[int, int, int]]:
    """
    Orders the triples in topological order based on the dependency graph.

    :param triples: List of triples representing (source, relationship, destination).
    :param input_nodes: List of input node IDs (sources in the AIG).
    :return: List of triples sorted in topological order.
    """
    # Step 1: Build the graph and in-degree count from the triples
    graph = {}
    in_degree = {}

    for src, rel, dest in triples:
        if src not in graph:
            graph[src] = []
        if dest not in graph:
            graph[dest] = []
        graph[src].append(dest)

        # Initialize in-degree count for nodes
        if dest not in in_degree:
            in_degree[dest] = 0
        if src not in in_degree:
            in_degree[src] = 0

        # Increase in-degree for destination node
        in_degree[dest] += 1

    # Step 2: Initialize the queue with nodes having zero in-degree (inputs)
    queue = deque([node for node in input_nodes if in_degree[node] == 0])
    topological_order = []

    # Step 3: Process the queue to order nodes
    while queue:
        node = queue.popleft()
        topological_order.append(node)

        # Reduce in-degree for each connected node and add to queue if it reaches zero
        for neighbor in graph.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Step 4: Reorder the triples based on the topological order of nodes
    # We use a dictionary to quickly find the order of each node
    node_order = {node: idx for idx, node in enumerate(topological_order)}
    ordered_triples = sorted(triples, key=lambda x: node_order[x[2]])

    return ordered_triples


def build_aig_from_triples(
        triples: List[Tuple[int, int, int]],
        input_nodes: List[int],
        output_nodes: List[int]
) -> Aig:
    """
    Constructs an AIG from triples (source, relationship, destination).

    :param triples: List of triples representing (source, relationship, destination)
    :param input_nodes: List of node IDs for primary inputs
    :param output_nodes: List of node IDs for primary outputs
    :return: Constructed AIG network
    """
    # Initialize an empty AIG network
    ntk = Aig()
    sigs = {}  # Maps node ID to signal in the AIG

    # Step 1: Add input nodes to the AIG and populate the sigs map
    for input_id in input_nodes:
        input_signal = ntk.create_pi()  # Create primary input node
        sigs[input_id] = input_signal  # Store the signal for this input

    # Step 2: Order the triples topologically
    ordered_triples = topologically_order_triples(triples, input_nodes)

    # Step 3: Process each triple in topological order to create AND gates
    for i in range(0, len(ordered_triples) - len(output_nodes), 2):  # Process pairs of triples
        # Get the destination ID and ensure both fan-ins point to the same destination
        source1, rel1, dest_id = ordered_triples[i]
        source2, rel2, dest_id2 = ordered_triples[i + 1]

        assert dest_id == dest_id2, "Expected pairs of triples with the same destination."

        # Retrieve signals for the two sources, applying inversion if necessary
        signal1 = sigs[source1]
        if rel1 == 1:  # Inverted signal
            signal1 = ntk.create_not(signal1)

        signal2 = sigs[source2]
        if rel2 == 1:  # Inverted signal
            signal2 = ntk.create_not(signal2)

        # Create the AND gate for the destination
        and_signal = ntk.create_and(signal1, signal2)
        sigs[dest_id] = and_signal  # Store the resulting signal for the destination

    # Step 4: Add output nodes using the last `len(output_nodes)` triples
    for idx, (src, rel, dest) in enumerate(ordered_triples[-len(output_nodes):]):
        assert dest in output_nodes, "Mismatch between output node order and triples."
        output_signal = sigs[src]
        if rel == 1:  # Inverted signal for the output
            output_signal = ntk.create_not(output_signal)
        ntk.create_po(output_signal)  # Attach the output signal as a primary output

    size = ntk.size()
    ntk.cleanup_dangling()
    print(f"Removed disconnected subgraphs. Reduced from {size} to {ntk.size()} nodes.")

    return ntk



def get_prior(all_graphs):
    # Initialize counts for four relationship types: 0, 1, 2, 3
    relation_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    total_triples = 0

    for graph in all_graphs:
        triples = graph["triples"]
        for triple in triples:
            rel = triple[2]  # Relationship is at index 2
            if rel not in relation_counts:
                relation_counts[rel] = 0
            relation_counts[rel] += 1
            total_triples += 1

    # Compute prior probabilities: count of each relationship type / total count of relationships
    p_R = {rel: count / total_triples for rel, count in relation_counts.items()}

    print("Prior probabilities p(R):")
    for rel, prob in p_R.items():
        relation_type = f"Relation {rel}"
        print(f"{relation_type}: {prob:.4f}")

    # Convert prior to list form in the order [0, 1, 2, 3]
    p_R_list = [p_R[0], p_R[1], p_R[2], p_R[3]]

    return p_R_list


def aggregate_source_embeddings(embedded_triplets, source_to_triplet_indices):
    """
    Aggregates triplet embeddings for each source node, with a zero embedding for the constant node if missing.

    Args:
        embedded_triplets: Tensor of shape [num_triples, embed_dim], containing embeddings for each triplet.
        source_to_triplet_indices: Dictionary mapping each source node ID to a list of indices in embedded_triplets.

    Returns:
        aggregated_source_embeddings: Dictionary mapping each source node ID to its aggregated embedding.
    """
    aggregated_source_embeddings = {}

    for source_id, indices in source_to_triplet_indices.items():
        # Get all triplet embeddings for this source node
        triplet_embeddings = embedded_triplets[indices]  # Shape: [num_occurrences, embed_dim]

        # Aggregate using mean pooling
        aggregated_embedding = triplet_embeddings.mean(dim=0)  # Shape: [embed_dim]

        aggregated_source_embeddings[source_id] = aggregated_embedding

    # Ensure a zero embedding for the constant node (node ID 0) if it is missing
    if 0 not in aggregated_source_embeddings:
        # Measure embed_dim from any existing embedding in the dictionary
        example_embedding = next(iter(aggregated_source_embeddings.values()))
        embed_dim = example_embedding.shape[0]

        print("Warning: Source embedding for constant node 0 is missing. Adding zero embedding.")
        aggregated_source_embeddings[0] = torch.zeros(embed_dim)

    return aggregated_source_embeddings


def generate_binary_inputs(num_inputs: int) -> List[List[int]]:
    """
    Generates all possible binary input combinations for a given number of inputs.

    :param num_inputs: The number of input signals to the AIG.
    :return: A list of lists, where each inner list is a binary configuration.
    """
    # Use a list comprehension to generate binary numbers from 0 to 2^num_inputs - 1
    return [[(i >> bit) & 1 for bit in range(num_inputs - 1, -1, -1)] for i in range(2 ** num_inputs)]